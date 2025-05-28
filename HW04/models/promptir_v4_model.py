"""
Defines the PromptIR-V4 model architecture and its constituent blocks.

This includes:
- Downsample and Upsample utility classes.
- PromptGenBlockV4: Generates dynamic spatial prompts.
- PromptInteractionBlockV4: Integrates prompts with decoder features.
- PromptIR_V4: The main U-Net based model with Transformer blocks and
  the V4 prompting mechanism.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming transformer_block.py is in the same directory or accessible via PYTHONPATH
from .transformer_block import TransformerBlock
# from .transformer_block import LayerNorm # Not used separately here


class Downsample(nn.Module):
    """
    Downsampling block using a strided convolution.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, C_in, H, W].
        Returns:
            torch.Tensor: Output tensor of shape [B, C_out, H/2, W/2].
        """
        return self.conv(x)


class Upsample(nn.Module):
    """
    Upsampling block using PixelShuffle.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Conv layer outputs 4x channels for PixelShuffle with scale_factor=2
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.PixelShuffle(2)  # Upscales H and W by 2

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, C_in, H, W].
        Returns:
            torch.Tensor: Output tensor of shape [B, C_out, H*2, W*2].
        """
        x = self.conv(x)
        x = self.upsample(x)
        return x


class PromptGenBlockV4(nn.Module):
    """
    Generates a dynamic spatial prompt based on input decoder features.

    The prompt is a weighted sum of learnable spatial prompt components,
    where weights are dynamically predicted from the decoder features.
    The resulting prompt is interpolated to match the decoder feature map size.
    """
    def __init__(self, features_input_dim, num_prompt_components,
                 prompt_channel_dim, base_prompt_hw, bias=False):
        """
        Args:
            features_input_dim (int): Channel dimension of the input decoder features.
            num_prompt_components (int): Number of learnable spatial prompt components.
            prompt_channel_dim (int): Channel dimension for each prompt component and the output prompt.
            base_prompt_hw (int or tuple): Initial (H, W) of the spatial prompt components.
            bias (bool): Whether to use bias in convolutional layers.
        """
        super().__init__()
        if isinstance(base_prompt_hw, int):
            base_prompt_h, base_prompt_w = base_prompt_hw, base_prompt_hw
        else:
            base_prompt_h, base_prompt_w = base_prompt_hw

        # Learnable spatial prompt components
        self.prompt_components = nn.Parameter(
            torch.randn(1, num_prompt_components, prompt_channel_dim, base_prompt_h, base_prompt_w)
        )
        # Linear layer to generate weights for combining prompt components
        self.weight_generator = nn.Linear(features_input_dim, num_prompt_components)
        # Final convolution to refine the generated prompt
        self.final_conv = nn.Conv2d(
            prompt_channel_dim, prompt_channel_dim, kernel_size=3, padding=1, bias=bias
        )

    def forward(self, decoder_features):
        """
        Dynamically generates a spatial prompt.
        Args:
            decoder_features (torch.Tensor): Features from a decoder stage [B, C_feat, H, W].
        Returns:
            torch.Tensor: Generated spatial prompt [B, C_prompt, H, W].
        """
        b, _, h, w = decoder_features.shape  # Use _ for c_feat as it's defined by features_input_dim

        # Generate weights for prompt components
        pooled_features = F.adaptive_avg_pool2d(decoder_features, (1, 1)).view(b, -1)  # [B, C_feat]
        prompt_weights = F.softmax(self.weight_generator(pooled_features), dim=1)  # [B, num_components]

        # Weighted sum of spatial prompt components
        # prompt_weights: [B, N] -> [B, N, 1, 1, 1] for broadcasting
        # self.prompt_components: [1, N, C_prompt, H_base, W_base]
        weighted_prompts = prompt_weights.view(b, -1, 1, 1, 1) * self.prompt_components
        summed_prompt = torch.sum(weighted_prompts, dim=1)  # [B, C_prompt, H_base, W_base]

        # Interpolate to match decoder feature map size and refine
        interpolated_prompt = F.interpolate(summed_prompt, size=(h, w), mode='bilinear', align_corners=False)
        output_prompt = self.final_conv(interpolated_prompt)

        return output_prompt


class PromptInteractionBlockV4(nn.Module):
    """
    Integrates a generated prompt with decoder features using a TransformerBlock.
    """
    def __init__(self, feature_dim, prompt_dim, num_transformer_heads,
                 ffn_expansion_factor, bias):
        """
        Args:
            feature_dim (int): Channel dimension of the decoder features.
            prompt_dim (int): Channel dimension of the generated prompt.
            num_transformer_heads (int): Number of attention heads for the TransformerBlock.
            ffn_expansion_factor (float): Expansion factor for the FFN in TransformerBlock.
            bias (bool): Whether to use bias in convolutional layers.
        """
        super().__init__()
        concat_dim = feature_dim + prompt_dim
        self.transformer = TransformerBlock(
            dim=concat_dim, num_heads=num_transformer_heads,
            ffn_expansion_factor=ffn_expansion_factor, bias=bias
        )
        # Adjust channel dimension back to original feature_dim
        self.channel_adjust_conv = nn.Conv2d(concat_dim, feature_dim, kernel_size=1, bias=bias)

    def forward(self, features, prompt):
        """
        Args:
            features (torch.Tensor): Decoder features [B, C_feat, H, W].
            prompt (torch.Tensor): Generated spatial prompt [B, C_prompt, H, W].
        Returns:
            torch.Tensor: Features after interaction with prompt [B, C_feat, H, W].
        """
        # Concatenate features and prompt along channel dimension
        x = torch.cat([features, prompt], dim=1)
        x = self.transformer(x)
        x = self.channel_adjust_conv(x)
        return x + features  # Residual connection with original features


class PromptIR_V4(nn.Module):
    """
    PromptIR-V4 model architecture.

    A U-Net based architecture with a Transformer backbone. It incorporates
    dynamic spatial prompting at multiple decoder stages for adaptive image restoration.
    """
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 base_dim=48,
                 num_blocks_per_level=(4, 6, 6, 8), # Default from a successful experiment
                 num_refinement_blocks=4,
                 num_prompt_components=5,
                 pg_prompt_dim_map=(64, 128, 256), # C_prompt for PGM (deep, mid, shallow decoder stages)
                 pg_base_hw_map=(16, 32, 64),    # Base H,W for PGM components (deep, mid, shallow)
                 backbone_num_attn_heads=8,
                 prompt_interaction_num_attn_heads=8,
                 ffn_expansion_factor=2.66,
                 bias=False):
        super().__init__()

        self.base_dim = base_dim
        self.num_levels = len(num_blocks_per_level)

        if isinstance(backbone_num_attn_heads, int):
            backbone_num_attn_heads = [backbone_num_attn_heads] * self.num_levels

        # Initial convolution: projects input to base_dim
        self.initial_conv = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1, bias=bias)

        # --- Encoder ---
        self.encoder_levels = nn.ModuleList()
        self.encoder_skip_dims = []  # To store dimensions for skip connections
        current_dim = base_dim
        for i in range(self.num_levels):
            self.encoder_skip_dims.append(current_dim)
            # Transformer blocks for the current encoder level
            level_blocks = nn.Sequential(*[
                TransformerBlock(dim=current_dim, num_heads=backbone_num_attn_heads[i],
                                 ffn_expansion_factor=ffn_expansion_factor, bias=bias)
                for _ in range(num_blocks_per_level[i])
            ])
            self.encoder_levels.append(level_blocks)
            # Downsample if not the last encoder level
            if i < self.num_levels - 1:
                self.encoder_levels.append(Downsample(current_dim, current_dim * 2))
                current_dim *= 2
        
        # --- Latent/Bottleneck ---
        # Transformer blocks at the bottleneck of the U-Net
        self.latent_transformers = nn.Sequential(*[
            TransformerBlock(dim=current_dim, num_heads=backbone_num_attn_heads[-1], # Use heads for deepest level
                             ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for _ in range(num_blocks_per_level[-1]) # Using last element of num_blocks for latent
        ])
        
        # --- Decoder with Prompt Blocks ---
        self.decoder_levels = nn.ModuleList()
        self.prompt_gens = nn.ModuleList()
        self.prompt_interactions = nn.ModuleList()

        # Decoder stages (number of upsampling steps = num_levels - 1)
        for i in range(self.num_levels - 1):
            # Upsampling block
            upsample_out_dim = current_dim // 2
            self.decoder_levels.append(Upsample(current_dim, upsample_out_dim))
            
            # Concatenation with skip connection from corresponding encoder level
            skip_dim_idx = self.num_levels - 2 - i  # Skips are indexed from shallowest encoder level
            merged_dim_after_skip = upsample_out_dim + self.encoder_skip_dims[skip_dim_idx]
            
            # Convolution to merge channels to the target dimension for this decoder level
            target_current_dim_decoder = upsample_out_dim
            self.decoder_levels.append(
                nn.Conv2d(merged_dim_after_skip, target_current_dim_decoder, kernel_size=1, bias=bias)
            )
            current_dim = target_current_dim_decoder  # This is the main feature dimension for this decoder stage

            # Transformer blocks for this decoder level
            num_dec_tf_blocks = num_blocks_per_level[skip_dim_idx]
            dec_tf_heads = backbone_num_attn_heads[skip_dim_idx]
            decoder_transformer_stage = nn.Sequential(*[
                TransformerBlock(dim=current_dim, num_heads=dec_tf_heads,
                                 ffn_expansion_factor=ffn_expansion_factor, bias=bias)
                for _ in range(num_dec_tf_blocks)
            ])
            self.decoder_levels.append(decoder_transformer_stage)

            # Prompt Generation and Interaction for this stage
            # pg_prompt_dim_map/pg_base_hw_map keys 0,1,2 correspond to i=0,1,2 (deep to shallow prompt blocks)
            current_pg_prompt_channel_dim = pg_prompt_dim_map[i]
            current_pg_base_hw = pg_base_hw_map[i]

            self.prompt_gens.append(
                PromptGenBlockV4(features_input_dim=current_dim, # Input from decoder TFs
                                 num_prompt_components=num_prompt_components,
                                 prompt_channel_dim=current_pg_prompt_channel_dim,
                                 base_prompt_hw=current_pg_base_hw,
                                 bias=bias)
            )
            self.prompt_interactions.append(
                PromptInteractionBlockV4(feature_dim=current_dim, # Decoder features
                                         prompt_dim=current_pg_prompt_channel_dim, # PGM output dim
                                         num_transformer_heads=prompt_interaction_num_attn_heads,
                                         ffn_expansion_factor=ffn_expansion_factor,
                                         bias=bias)
            )

        # --- Refinement Stage ---
        # Transformer blocks operating on the output of the shallowest decoder stage
        # current_dim at this point should be equal to base_dim
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=base_dim, num_heads=backbone_num_attn_heads[0], # Use heads for shallowest level
                             ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for _ in range(num_refinement_blocks)
        ])
        
        # Final convolution to produce the output image
        self.final_conv = nn.Conv2d(base_dim, out_channels, kernel_size=3, padding=1, bias=bias)

    def forward(self, x_inp):
        """
        Forward pass of the PromptIR-V4 model.
        Args:
            x_inp (torch.Tensor): Input degraded image tensor [B, C_in, H, W].
        Returns:
            torch.Tensor: Restored image tensor [B, C_out, H, W].
        """
        skip_connections = []
        
        # --- Encoder ---
        x = self.initial_conv(x_inp)
        enc_module_idx = 0
        for i in range(self.num_levels):
            x = self.encoder_levels[enc_module_idx](x)  # Apply Transformer blocks
            enc_module_idx += 1
            if i < self.num_levels - 1: # If not the last (bottleneck) level
                skip_connections.append(x) # Save for skip connection
                x = self.encoder_levels[enc_module_idx](x)  # Apply Downsample
                enc_module_idx += 1
        
        # --- Bottleneck ---
        x = self.latent_transformers(x)

        # --- Decoder ---
        dec_module_idx = 0
        for i in range(self.num_levels - 1): # Iterate through decoder stages (3 stages for 4 levels)
            x = self.decoder_levels[dec_module_idx](x)  # Upsample
            dec_module_idx += 1
            
            skip = skip_connections.pop()  # Get skip connection (from last to first stored)
            x = torch.cat([x, skip], dim=1) # Concatenate
            x = self.decoder_levels[dec_module_idx](x)  # Merge convolution
            dec_module_idx += 1
            
            # Main decoder Transformer blocks for this level
            x_after_tfs = self.decoder_levels[dec_module_idx](x)
            dec_module_idx += 1
            
            # Prompt Generation and Interaction
            generated_prompt = self.prompt_gens[i](x_after_tfs)
            x = self.prompt_interactions[i](x_after_tfs, generated_prompt)
            
        # --- Refinement & Final Output ---
        x = self.refinement(x)
        x_restored = self.final_conv(x)
        
        return x_restored + x_inp # Global residual connection


if __name__ == '__main__':
    # Example usage and test for the PromptIR-V4 model
    bs = 2  # Batch size
    img_size = 256  # Image dimension
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n--- Testing PromptIR_V4 Model ---")
    # Model parameters should match those used in training/prediction scripts
    model_v4 = PromptIR_V4(
        base_dim=48,
        num_blocks_per_level=[3, 4, 4, 6], # Example, adjust to match your config
        num_refinement_blocks=4,
        backbone_num_attn_heads=8,
        prompt_interaction_num_attn_heads=8,
        pg_prompt_dim_map={0: 256, 1: 128, 2: 64}, # Corresponds to 3 decoder prompt stages
        pg_base_hw_map={
            0: img_size // 16, # Deepest prompt stage
            1: img_size // 8,  # Middle prompt stage
            2: img_size // 4   # Shallowest prompt stage
        }
    ).to(device)

    test_input = torch.randn(bs, 3, img_size, img_size).to(device)
    print(f"Input shape: {test_input.shape}")
    
    try:
        output = model_v4(test_input)
        print(f"PromptIR_V4 output shape: {output.shape}")
        assert output.shape == (bs, 3, img_size, img_size), "Output shape mismatch!"
        
        total_params = sum(p.numel() for p in model_v4.parameters() if p.requires_grad)
        print(f"Total trainable parameters for PromptIR_V4: {total_params / 1e6:.2f} M")
        print("PromptIR_V4 forward pass successful.")
        
    except Exception as e:
        print(f"Error during PromptIR_V4 forward pass: {e}")
        import traceback
        traceback.print_exc()
