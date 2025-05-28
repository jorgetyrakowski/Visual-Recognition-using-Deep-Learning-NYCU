"""
Defines modules for prompt generation and interaction, inspired by the
PromptIR paper's mechanisms. These were likely used in earlier model versions
(e.g., V1, V2, V3) before the V4-specific prompt blocks were developed.

Includes:
- PromptGenerationModule: Generates input-conditioned prompts.
- PromptInteractionModule: Interacts features with generated prompts.
- PromptBlock: A combined module of PGM and PIM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming transformer_block.py is in the same directory or accessible via PYTHONPATH
from .transformer_block import TransformerBlock, LayerNorm


class PromptGenerationModule(nn.Module):
    """
    Generates an input-conditioned prompt based on decoder features.
    This version is inspired by the original PromptIR paper's PGM.
    It uses learnable prompt component vectors and dynamically computes weights
    to combine them. The resulting vector is then spatially broadcasted and refined.
    """
    def __init__(self, in_dim_feat, num_prompt_components, prompt_component_dim, bias=False):
        """
        Args:
            in_dim_feat (int): Channel dimension of the input features (Fl).
            num_prompt_components (int): Number of learnable prompt components (N).
            prompt_component_dim (int): Dimension of each prompt component (C_prompt),
                                      also the output prompt's channel dimension.
            bias (bool): Whether to use bias in convolutional layers.
        """
        super().__init__()
        self.num_prompt_components = num_prompt_components
        self.prompt_component_dim = prompt_component_dim

        # Learnable prompt components (vectors)
        # Shape: [N, C_prompt]
        self.prompt_components = nn.Parameter(
            torch.randn(num_prompt_components, prompt_component_dim)
        )

        # Layers to generate weights for prompt components from input features
        # fc1: Channel downscaling (as per PromptIR paper description)
        self.fc1 = nn.Conv2d(in_dim_feat, prompt_component_dim, kernel_size=1, bias=bias)
        # fc2: Generates N weights for the N prompt components
        self.fc2 = nn.Conv2d(prompt_component_dim, num_prompt_components, kernel_size=1, bias=bias)

        # Final convolution layer to refine the spatially broadcasted prompt
        # Output is P of shape [B, C_prompt, H, W]
        self.conv_out = nn.Conv2d(
            prompt_component_dim, prompt_component_dim, kernel_size=3, padding=1, bias=bias
        )

    def forward(self, features):
        """
        Generates a spatial prompt conditioned on input features.
        Args:
            features (torch.Tensor): Input features Fl of shape [B, C_in_feat, H, W].
        Returns:
            torch.Tensor: Generated prompt P of shape [B, C_prompt, H, W].
        """
        b, _, h, w = features.shape  # Use _ for c_in as it's defined by in_dim_feat

        # 1. Generate prompt weights (w_i in PromptIR Eq. 2)
        # Global Average Pooling (GAP) of input features
        pooled_features = F.adaptive_avg_pool2d(features, (1, 1))  # [B, C_in_feat, 1, 1]
        
        # Channel downscaling to create a compact feature vector
        compact_vector = self.fc1(pooled_features)  # [B, C_prompt, 1, 1]
        
        # Generate N weights from the compact vector
        prompt_weights_logits = self.fc2(compact_vector)  # [B, N, 1, 1]
        prompt_weights = F.softmax(prompt_weights_logits, dim=1)  # Softmax over N components

        # 2. Weighted sum of prompt components
        # Reshape weights for broadcasting: [B, N, 1, 1] -> [B, N, 1]
        weights = prompt_weights.squeeze(-1).squeeze(-1).unsqueeze(2)  # Shape: [B, N, 1]
        
        # Reshape prompt_components for broadcasting: [N, C_prompt] -> [1, N, C_prompt]
        components = self.prompt_components.unsqueeze(0)
        
        # Weighted sum: (weights * components) results in [B, N, C_prompt]
        # Sum over N (dim=1) to get [B, C_prompt]
        weighted_sum_vectors = torch.sum(weights * components, dim=1)  # Shape: [B, C_prompt]
        
        # Reshape to [B, C_prompt, 1, 1] to prepare for spatial broadcasting
        prompt_vectors_spatial = weighted_sum_vectors.unsqueeze(-1).unsqueeze(-1)

        # 3. Spatially broadcast and refine
        # Upsample/interpolate the [B, C_prompt, 1, 1] vector to match feature map size [H, W]
        prompt_spatial_broadcast = F.interpolate(
            prompt_vectors_spatial, size=(h, w), mode='bilinear', align_corners=False
        )

        # Final Conv3x3 to get the generated prompt P
        generated_prompt = self.conv_out(prompt_spatial_broadcast)  # [B, C_prompt, H, W]
        
        return generated_prompt


class PromptInteractionModule(nn.Module):
    """
    Interacts input features with a generated prompt using a TransformerBlock.
    This follows the PIM description in the PromptIR paper.
    """
    def __init__(self, feature_dim, prompt_dim, num_heads=8,
                 ffn_expansion_factor=2.66, bias=False):
        """
        Args:
            feature_dim (int): Channel dimension of the input features (Fl).
            prompt_dim (int): Channel dimension of the generated prompt (P).
            num_heads (int): Number of attention heads for the TransformerBlock.
            ffn_expansion_factor (float): Expansion factor for FFN in TransformerBlock.
            bias (bool): Whether to use bias in convolutional layers.
        """
        super().__init__()
        self.transformer_dim = feature_dim + prompt_dim  # Dimension after concatenation
        self.transformer_block = TransformerBlock(
            dim=self.transformer_dim,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias
        )
        # Output convolution to restore original feature dimension
        self.conv_out = nn.Conv2d(
            self.transformer_dim, feature_dim, kernel_size=3, padding=1, bias=bias
        )

    def forward(self, features, prompt):
        """
        Args:
            features (torch.Tensor): Input features Fl of shape [B, C_feat, H, W].
            prompt (torch.Tensor): Generated prompt P of shape [B, C_prompt, H, W].
        Returns:
            torch.Tensor: Modulated features F_hat_l of shape [B, C_feat, H, W].
        """
        # Concatenate features and prompt along the channel dimension
        concat_features = torch.cat([features, prompt], dim=1)
        
        # Pass through TransformerBlock
        transformed_features = self.transformer_block(concat_features)
        
        # Output convolution to project back to original feature dimension
        output_features = self.conv_out(transformed_features)
        
        return output_features


class PromptBlock(nn.Module):
    """
    A combined block consisting of a PromptGenerationModule (PGM)
    and a PromptInteractionModule (PIM).
    """
    def __init__(self, feature_dim, num_prompt_components, prompt_component_dim,
                 num_attn_heads=8, ffn_expansion_factor=2.66, bias=False):
        """
        Args:
            feature_dim (int): Channel dimension of the input features.
            num_prompt_components (int): Number of learnable components for PGM.
            prompt_component_dim (int): Output channel dimension of PGM / prompt input to PIM.
            num_attn_heads (int): Number of attention heads for PIM's TransformerBlock.
            ffn_expansion_factor (float): FFN expansion factor for PIM's TransformerBlock.
            bias (bool): Whether to use bias in conv layers.
        """
        super().__init__()
        self.pgm = PromptGenerationModule(
            in_dim_feat=feature_dim,
            num_prompt_components=num_prompt_components,
            prompt_component_dim=prompt_component_dim,
            bias=bias
        )
        self.pim = PromptInteractionModule(
            feature_dim=feature_dim,
            prompt_dim=prompt_component_dim,
            num_heads=num_attn_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias
        )

    def forward(self, features):
        """
        Args:
            features (torch.Tensor): Input features Fl from a decoder stage.
        Returns:
            torch.Tensor: Modulated features after prompt interaction.
        """
        generated_prompt = self.pgm(features)
        modulated_features = self.pim(features, generated_prompt)
        return modulated_features


if __name__ == '__main__':
    # Example Usage and Tests
    batch_size = 2
    feature_h, feature_w = 32, 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing PromptBlock components on device: {device}")
    
    decoder_feature_dim = 128
    num_components = 5
    prompt_dim = 64
    attention_heads = 4

    print(f"\n--- Testing PromptGenerationModule (PGM) ---")
    pgm_test = PromptGenerationModule(
        in_dim_feat=decoder_feature_dim,
        num_prompt_components=num_components,
        prompt_component_dim=prompt_dim
    ).to(device)
    sample_features_pgm = torch.randn(batch_size, decoder_feature_dim, feature_h, feature_w).to(device)
    try:
        generated_p = pgm_test(sample_features_pgm)
        print(f"PGM Input (Fl) shape: {sample_features_pgm.shape}")
        print(f"PGM Output (P) shape: {generated_p.shape}")
        assert generated_p.shape == (batch_size, prompt_dim, feature_h, feature_w)
        print("PGM test passed.")
    except Exception as e:
        print(f"Error in PGM test: {e}")

    print(f"\n--- Testing PromptInteractionModule (PIM) ---")
    pim_test = PromptInteractionModule(
        feature_dim=decoder_feature_dim,
        prompt_dim=prompt_dim,
        num_heads=attention_heads
    ).to(device)
    # Use a dummy prompt for PIM input, as PGM output is already tested
    sample_prompt_pim = torch.randn(batch_size, prompt_dim, feature_h, feature_w).to(device)
    try:
        modulated_f = pim_test(sample_features_pgm, sample_prompt_pim) # Re-use sample_features_pgm
        print(f"PIM Input (Fl) shape: {sample_features_pgm.shape}")
        print(f"PIM Input (P) shape: {sample_prompt_pim.shape}")
        print(f"PIM Output (F_hat_l) shape: {modulated_f.shape}")
        assert modulated_f.shape == (batch_size, decoder_feature_dim, feature_h, feature_w)
        print("PIM test passed.")
    except Exception as e:
        print(f"Error in PIM test: {e}")

    print(f"\n--- Testing PromptBlock (PGM + PIM) ---")
    prompt_block_combined_test = PromptBlock(
        feature_dim=decoder_feature_dim,
        num_prompt_components=num_components,
        prompt_component_dim=prompt_dim,
        num_attn_heads=attention_heads
    ).to(device)
    try:
        output_pb = prompt_block_combined_test(sample_features_pgm) # Re-use sample_features_pgm
        print(f"PromptBlock Input (Fl) shape: {sample_features_pgm.shape}")
        print(f"PromptBlock Output (F_hat_l) shape: {output_pb.shape}")
        assert output_pb.shape == (batch_size, decoder_feature_dim, feature_h, feature_w)
        print("PromptBlock test passed.")
    except Exception as e:
        print(f"Error in PromptBlock test: {e}")
