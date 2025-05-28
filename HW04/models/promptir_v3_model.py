import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_block import TransformerBlock, LayerNorm, MDTA, GDFN

# --- Re-using Downsample and Upsample from baseline for consistency ---
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.PixelShuffle(2)
    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x

# --- PIP-Style Prompt Modules (Copied from V2) ---

class PromptToPromptInteraction(nn.Module):
    def __init__(self, basic_prompt_dim, degradation_prompt_dim, universal_prompt_dim, num_heads=4, bias=False):
        super().__init__()
        self.basic_prompt_dim = basic_prompt_dim
        self.degradation_prompt_dim = degradation_prompt_dim
        self.universal_prompt_dim = universal_prompt_dim

        self.query_proj = nn.Conv2d(basic_prompt_dim, basic_prompt_dim, 1, bias=bias)
        self.kv_proj_degrad = nn.Linear(degradation_prompt_dim, basic_prompt_dim * 2, bias=bias)
        self.attn = MDTA(dim=basic_prompt_dim, num_heads=num_heads, bias=bias)
        self.ffn = GDFN(dim=basic_prompt_dim, bias=bias)
        
        if basic_prompt_dim != universal_prompt_dim:
            self.out_proj = nn.Conv2d(basic_prompt_dim, universal_prompt_dim, 1, bias=bias)
        else:
            self.out_proj = nn.Identity()

    def forward(self, basic_prompt_b, degradation_prompt_d_vec):
        b, c_basic, h_basic, w_basic = basic_prompt_b.shape
        degrad_conditioning = self.kv_proj_degrad(degradation_prompt_d_vec)
        degrad_scale, degrad_shift = degrad_conditioning.chunk(2, dim=1)
        degrad_scale = degrad_scale.unsqueeze(-1).unsqueeze(-1)
        degrad_shift = degrad_shift.unsqueeze(-1).unsqueeze(-1)

        fused_prompt = basic_prompt_b * (1 + degrad_scale) + degrad_shift
        fused_prompt = self.attn(fused_prompt) 
        fused_prompt = self.ffn(fused_prompt)
        
        universal_prompt_u = self.out_proj(fused_prompt)
        return universal_prompt_u

class SelectivePromptToFeatureInteraction(nn.Module):
    def __init__(self, feature_dim, universal_prompt_dim, num_heads=4, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        self.feature_dim = feature_dim
        self.universal_prompt_dim = universal_prompt_dim

        if feature_dim != universal_prompt_dim:
            self.feat_proj = nn.Conv2d(feature_dim, universal_prompt_dim, kernel_size=1, bias=bias)
        else:
            self.feat_proj = nn.Identity()

        concat_dim = universal_prompt_dim + universal_prompt_dim
        
        self.norm_in = LayerNorm(concat_dim)
        self.attn = MDTA(dim=concat_dim, num_heads=num_heads, bias=bias)
        self.norm_mid = LayerNorm(concat_dim)
        self.ffn = GDFN(dim=concat_dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
        self.out_conv = nn.Conv2d(concat_dim, feature_dim, kernel_size=3, padding=1, bias=bias)

    def forward(self, features_z, universal_prompt_u):
        projected_feat_z = self.feat_proj(features_z)
        concat_input = torch.cat([projected_feat_z, universal_prompt_u], dim=1)
        
        x = concat_input 
        x_res1 = x
        x = self.norm_in(x)
        x = self.attn(x) 
        x = x + x_res1

        x_res2 = x 
        x = self.norm_mid(x)
        x = self.ffn(x)
        x = x + x_res2
        
        modulated_output = self.out_conv(x)
        return modulated_output + features_z


class PromptIR_V3(nn.Module): # Renamed class
    def __init__(self, 
                 in_channels=3, 
                 out_channels=3, 
                 base_dim=48, # Defaulting to larger capacity
                 num_blocks_per_level=[4, 6, 6, 8], # Defaulting to larger capacity
                 num_degradations=2,
                 degradation_prompt_dim=64,
                 basic_prompt_c=64, basic_prompt_hw=16,
                 universal_prompt_dim=64,
                 p2p_heads=4,
                 p2f_heads=4,
                 num_attn_heads=8, # Defaulting to larger capacity for backbone
                 ffn_expansion_factor=2.66,
                 bias=False):
        super().__init__()

        self.base_dim = base_dim
        self.num_levels = len(num_blocks_per_level)

        self.degradation_aware_prompts = nn.Embedding(num_degradations, degradation_prompt_dim)
        self.basic_restoration_prompt_tensor = nn.Parameter(
            torch.randn(1, basic_prompt_c, basic_prompt_hw, basic_prompt_hw)
        )

        self.p2p_interaction = PromptToPromptInteraction(
            basic_prompt_dim=basic_prompt_c,
            degradation_prompt_dim=degradation_prompt_dim,
            universal_prompt_dim=universal_prompt_dim,
            num_heads=p2p_heads,
            bias=bias
        )

        self.initial_conv = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1, bias=bias)

        self.encoder_levels = nn.ModuleList()
        self.encoder_dims = [] 
        current_dim = base_dim
        for i in range(self.num_levels):
            self.encoder_dims.append(current_dim)
            level_blocks = [
                TransformerBlock(dim=current_dim, num_heads=num_attn_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
                for _ in range(num_blocks_per_level[i])
            ]
            self.encoder_levels.append(nn.Sequential(*level_blocks))
            if i < self.num_levels - 1:
                self.encoder_levels.append(Downsample(current_dim, current_dim * 2))
                current_dim *= 2
        
        self.decoder_levels = nn.ModuleList()
        self.p2f_interactions = nn.ModuleList()

        for i in range(self.num_levels - 1):
            skip_connection_dim = self.encoder_dims[self.num_levels - 1 - (i+1)]
            self.p2f_interactions.append(
                SelectivePromptToFeatureInteraction(
                    feature_dim=skip_connection_dim,
                    universal_prompt_dim=universal_prompt_dim,
                    num_heads=p2f_heads,
                    bias=bias,
                    ffn_expansion_factor=ffn_expansion_factor # Pass this along
                )
            )
            self.decoder_levels.append(Upsample(current_dim, current_dim // 2))
            current_dim //= 2
            num_dec_blocks = num_blocks_per_level[self.num_levels - 1 - (i+1)]
            self.decoder_levels.append(nn.Conv2d(current_dim + skip_connection_dim, current_dim, kernel_size=1, bias=bias))
            level_dec_blocks = [
                TransformerBlock(dim=current_dim, num_heads=num_attn_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
                for _ in range(num_dec_blocks)
            ]
            self.decoder_levels.append(nn.Sequential(*level_dec_blocks))

        self.final_conv = nn.Conv2d(base_dim, out_channels, kernel_size=3, padding=1, bias=bias)

    def forward(self, x, degradation_label):
        b = x.shape[0]
        selected_degrad_aware_prompt_vec = self.degradation_aware_prompts(degradation_label)
        batch_basic_prompt = self.basic_restoration_prompt_tensor.repeat(b, 1, 1, 1)
        universal_prompt_u = self.p2p_interaction(batch_basic_prompt, selected_degrad_aware_prompt_vec)

        skip_connections = []
        x = self.initial_conv(x)
        enc_idx = 0
        for i in range(self.num_levels):
            x = self.encoder_levels[enc_idx](x)
            enc_idx += 1
            if i < self.num_levels - 1:
                skip_connections.append(x)
                x = self.encoder_levels[enc_idx](x)
                enc_idx += 1
        
        dec_idx = 0
        p2f_idx = 0
        for i in range(self.num_levels - 1):
            x = self.decoder_levels[dec_idx](x)
            dec_idx += 1
            skip = skip_connections.pop()
            h_skip, w_skip = skip.shape[2], skip.shape[3]
            universal_prompt_u_resized = F.interpolate(universal_prompt_u, size=(h_skip, w_skip), mode='bilinear', align_corners=False)
            modulated_skip = self.p2f_interactions[p2f_idx](skip, universal_prompt_u_resized)
            p2f_idx += 1
            x = torch.cat([x, modulated_skip], dim=1)
            x = self.decoder_levels[dec_idx](x)
            dec_idx += 1
            x = self.decoder_levels[dec_idx](x)
            dec_idx += 1
            
        x = self.final_conv(x)
        return x

if __name__ == '__main__':
    img_channels = 3
    batch_s = 2 # Critical for memory testing

    # Test V3 model with increased capacity
    model_v3_test = PromptIR_V3(
        in_channels=img_channels,
        out_channels=img_channels,
        base_dim=48, 
        num_blocks_per_level=[4,6,6,8], 
        num_degradations=2,
        degradation_prompt_dim=64, # As in V2
        basic_prompt_c=64, basic_prompt_hw=16, # As in V2
        universal_prompt_dim=64, # As in V2
        p2p_heads=4, # As in V2
        p2f_heads=4, # As in V2
        num_attn_heads=8 # For backbone
    ).cuda()

    test_input = torch.randn(batch_s, img_channels, 256, 256).cuda()
    degrad_labels = torch.randint(0, 2, (batch_s,)).cuda() 
    
    print(f"Input shape: {test_input.shape}")
    try:
        output = model_v3_test(test_input, degrad_labels)
        print(f"PromptIR_V3 output shape: {output.shape}")
        assert output.shape == (batch_s, img_channels, 256, 256)
        total_params = sum(p.numel() for p in model_v3_test.parameters() if p.requires_grad)
        print(f"Total trainable parameters for PromptIR_V3 (test config): {total_params/1e6:.2f} M")
        print("PromptIR_V3 forward pass successful with target capacity.")
    except Exception as e:
        print(f"Error during PromptIR_V3 forward pass: {e}")
        import traceback
        traceback.print_exc()
