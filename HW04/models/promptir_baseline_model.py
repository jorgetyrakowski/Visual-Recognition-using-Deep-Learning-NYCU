import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_block import TransformerBlock
from .prompt_block import PromptBlock

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels): # out_channels is the target number of channels AFTER PixelShuffle
        super().__init__()
        # The conv layer needs to output out_channels * 4 for PixelShuffle(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x

class PromptIRBaseline(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 out_channels=3, 
                 base_dim=48, # Base dimension, PromptIR uses 48
                 num_blocks_per_level=[4, 6, 6, 8], # As per PromptIR for encoder/decoder levels
                 num_prompt_components=5, # N in PromptIR paper
                 prompt_component_dim=64, # C_prompt
                 num_attn_heads=8,
                 ffn_expansion_factor=2.66,
                 bias=False):
        super().__init__()

        self.base_dim = base_dim
        self.num_levels = len(num_blocks_per_level)

        # Initial convolution
        self.initial_conv = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1, bias=bias)

        # Encoder
        self.encoder_levels = nn.ModuleList()
        current_dim = base_dim
        for i in range(self.num_levels):
            level_blocks = [
                TransformerBlock(dim=current_dim, num_heads=num_attn_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
                for _ in range(num_blocks_per_level[i])
            ]
            self.encoder_levels.append(nn.Sequential(*level_blocks))
            if i < self.num_levels - 1: # Add downsample except for the last encoder level
                self.encoder_levels.append(Downsample(current_dim, current_dim * 2))
                current_dim *= 2
        
        # Bottleneck (deepest level of encoder is the start of bottleneck)
        # The last element of encoder_levels is already the deepest transformer blocks.
        # current_dim is now at the bottleneck dimension

        # Decoder
        self.decoder_levels = nn.ModuleList()
        self.prompt_blocks = nn.ModuleList() # Store prompt blocks separately for clarity

        # Decoder levels are num_levels-1 because we start from bottleneck up
        for i in range(self.num_levels - 1): 
            # Upsampling brings channels from current_dim to current_dim // 2
            self.decoder_levels.append(Upsample(current_dim, current_dim // 2))
            current_dim //= 2 # Dimension after upsampling (matches skip connection)
            
            # PromptBlock for this decoder level
            # feature_dim for PGM is current_dim (after upsample, before concat with skip)
            # However, PromptIR inserts prompt block *between* decoder levels.
            # "Prompt blocks are adapter modules that sequentially connect every two levels of the decoder."
            # This means it takes the output of an upsampling+transformer_blocks stage.
            # Let's refine: Upsample -> Concat Skip -> TransformerBlocks -> PromptBlock (if not last stage)

            # The prompt block is applied to the output of the decoder transformer blocks at this level
            # The input feature_dim to PromptBlock will be current_dim
            self.prompt_blocks.append(
                PromptBlock(
                    feature_dim=current_dim, # This will be the dim after concat with skip
                    num_prompt_components=num_prompt_components,
                    prompt_component_dim=prompt_component_dim,
                    num_attn_heads=num_attn_heads,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias
                )
            )
            
            # Transformer blocks for this decoder level (operates on current_dim + skip_dim)
            # The input to these blocks will be current_dim (from upsample) + current_dim (from skip)
            # So, the actual dim for these blocks is current_dim * 2, then reduced.
            # Let's simplify: after upsample, dim is current_dim. After concat with skip, it's current_dim*2.
            # The transformer blocks should operate on current_dim (after skip connection and a conv to merge)
            
            # Number of blocks for decoder level i corresponds to encoder level (num_levels - 1 - i -1)
            # e.g. first decoder stage (i=0) uses blocks from second to last encoder stage
            num_dec_blocks = num_blocks_per_level[self.num_levels - 1 - (i+1)]
            
            # Conv to merge channels after concatenating skip connection
            # Input: current_dim (from upsample) + current_dim (from corresponding encoder level)
            # Output: current_dim
            self.decoder_levels.append(nn.Conv2d(current_dim * 2, current_dim, kernel_size=1, bias=bias))
            
            level_dec_blocks = [
                TransformerBlock(dim=current_dim, num_heads=num_attn_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
                for _ in range(num_dec_blocks)
            ]
            self.decoder_levels.append(nn.Sequential(*level_dec_blocks))

        # Final convolution
        self.final_conv = nn.Conv2d(base_dim, out_channels, kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        skip_connections = []
        
        # Encoder path
        x = self.initial_conv(x)
        
        enc_idx = 0
        for i in range(self.num_levels):
            x = self.encoder_levels[enc_idx](x) # Transformer blocks
            enc_idx += 1
            if i < self.num_levels - 1:
                skip_connections.append(x)
                x = self.encoder_levels[enc_idx](x) # Downsample
                enc_idx += 1
        
        # Bottleneck is 'x' now

        # Decoder path
        dec_idx = 0
        prompt_idx = 0
        for i in range(self.num_levels - 1):
            x = self.decoder_levels[dec_idx](x) # Upsample
            dec_idx += 1
            
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1)
            x = self.decoder_levels[dec_idx](x) # Conv to merge channels
            dec_idx += 1
            
            x = self.decoder_levels[dec_idx](x) # Decoder Transformer blocks
            dec_idx += 1
            
            # Apply PromptBlock after the transformer blocks of this decoder level
            if prompt_idx < len(self.prompt_blocks): # Prompt blocks are between decoder levels
                 x = self.prompt_blocks[prompt_idx](x)
                 prompt_idx +=1

        x = self.final_conv(x)
        return x

if __name__ == '__main__':
    # Example Usage
    img_channels = 3
    base_model_dim = 32 # Smaller for quick test
    levels = 3 # For a 3-level U-Net (2 down/up stages)
    blocks_per_level = [2, 2, 2] # Simplified
    
    # For a 4-level U-Net as in PromptIR (3 down/up stages)
    # levels = 4
    # blocks_per_level = [2,2,2,2] # Simplified for testing

    model = PromptIRBaseline(
        in_channels=img_channels,
        out_channels=img_channels,
        base_dim=base_model_dim,
        num_blocks_per_level=blocks_per_level, # Use simplified for test
        num_prompt_components=3,
        prompt_component_dim=32, # Should be manageable
        num_attn_heads=4 
    )

    # Input: B=1, C=3, H=128, W=128 (smaller for testing)
    # For 256x256, with 3 downsamples, min size is 32x32.
    # With 4 levels (3 downsamples), 256 -> 128 -> 64 -> 32. This is fine.
    # If blocks_per_level = [2,2,2], then num_levels = 3.
    # Encoder: L0_blocks -> DS -> L1_blocks -> DS -> L2_blocks (bottleneck)
    # Decoder: US -> Skip_L1 -> L1_dec_blocks -> Prompt -> US -> Skip_L0 -> L0_dec_blocks -> Prompt (No, last prompt is not needed)
    # PromptIR: "3 prompt blocks in the overall PromptIR network" for 4 levels.
    # This means one after each decoder stage's transformer blocks, except the very last one before final_conv.

    test_input_128 = torch.randn(1, img_channels, 128, 128)
    output_128 = model(test_input_128)
    print(f"PromptIRBaseline with {levels} levels, input 128x128, output shape: {output_128.shape}")
    assert output_128.shape == (1, img_channels, 128, 128)

    # Test with parameters closer to paper for shape compatibility
    model_paper_like_dims = PromptIRBaseline(
        base_dim=48,
        num_blocks_per_level=[4,6,6,8], # 4 levels
        num_prompt_components=5,
        prompt_component_dim=64,
        num_attn_heads=8
    )
    test_input_256 = torch.randn(1, img_channels, 256, 256) # Actual target size
    output_256 = model_paper_like_dims(test_input_256)
    print(f"PromptIRBaseline with paper-like dims, input 256x256, output shape: {output_256.shape}")
    assert output_256.shape == (1, img_channels, 256, 256)
    
    print("PromptIRBaseline model tests passed.")
