import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_block import TransformerBlock, LayerNorm, MDTA, GDFN # Re-using MDTA and GDFN

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

# --- PIP-Style Prompt Modules ---

class PromptToPromptInteraction(nn.Module):
    """
    Fuses a degradation-aware prompt (d_t) with a basic restoration prompt (B)
    to create a universal restoration prompt (U).
    Inspired by PIP paper's P2P interaction using cross-attention.
    Query: Basic Restoration Prompt (B)
    Key/Value: Degradation-aware prompt (d_t)
    """
    def __init__(self, basic_prompt_dim, degradation_prompt_dim, universal_prompt_dim, num_heads=4, bias=False):
        super().__init__()
        self.basic_prompt_dim = basic_prompt_dim
        self.degradation_prompt_dim = degradation_prompt_dim
        self.universal_prompt_dim = universal_prompt_dim # Output dim for U

        # Projects basic_prompt to query_dim (which is basic_prompt_dim for MDTA)
        self.query_proj = nn.Conv2d(basic_prompt_dim, basic_prompt_dim, 1, bias=bias)
        
        # Projects degradation_prompt (1D vector) to key/value_dim for spatial interaction
        # Degradation prompt is 1D, needs to be expanded and projected to match spatial basic prompt
        self.kv_proj_degrad = nn.Linear(degradation_prompt_dim, basic_prompt_dim * 2, bias=bias) # Output K and V channels

        self.attn = MDTA(dim=basic_prompt_dim, num_heads=num_heads, bias=bias) # Operates on spatial features
        self.ffn = GDFN(dim=basic_prompt_dim, bias=bias) # Operates on spatial features
        
        # Final projection to universal_prompt_dim if different
        if basic_prompt_dim != universal_prompt_dim:
            self.out_proj = nn.Conv2d(basic_prompt_dim, universal_prompt_dim, 1, bias=bias)
        else:
            self.out_proj = nn.Identity()

    def forward(self, basic_prompt_b, degradation_prompt_d_vec):
        """
        Args:
            basic_prompt_b (torch.Tensor): Basic restoration prompt, shape [Batch, C_basic, H_basic, W_basic]
            degradation_prompt_d_vec (torch.Tensor): Degradation-aware prompt vector, shape [Batch, C_degrad_vec]
        Returns:
            torch.Tensor: Universal restoration prompt U, shape [Batch, C_universal, H_basic, W_basic]
        """
        b, c_basic, h_basic, w_basic = basic_prompt_b.shape
        
        # Query from basic_prompt_b
        q_b = self.query_proj(basic_prompt_b) # [B, C_basic, H_basic, W_basic]

        # Key & Value from degradation_prompt_d_vec
        # Expand d_vec and project to match spatial dimensions of basic_prompt_b for interaction
        kv_degrad = self.kv_proj_degrad(degradation_prompt_d_vec) # [B, C_basic*2]
        k_d_flat, v_d_flat = kv_degrad.chunk(2, dim=1) # Each [B, C_basic]
        
        # Tile/repeat k_d_flat and v_d_flat to spatial dimensions [B, C_basic, H_basic, W_basic]
        k_d = k_d_flat.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h_basic, w_basic)
        v_d = v_d_flat.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h_basic, w_basic)

        # PIP uses cross-attention. MDTA is self-attention.
        # For cross-attention with MDTA structure:
        # Q from basic_prompt_b, K,V from degradation_prompt_d_vec
        # We need to adapt MDTA or use a more standard cross-attention.
        # For simplicity, let's use a modified attention: Q from basic, K,V from degrad.
        # This is a simplification of full cross-attention.
        # A simple way: element-wise modulation then pass through MDTA+GDFN
        
        # Simplified interaction: Modulate basic_prompt_b with k_d, v_d (e.g. adaptive instance norm style or simple scaling)
        # Or, treat basic_prompt_b as X and (k_d, v_d) as context for attention.
        # Let's try a simple feature fusion for now, then refine if needed.
        # Modulate basic_prompt_b with k_d (acting as scale) and v_d (acting as shift)
        
        # Alternative: Use basic_prompt_b as input to MDTA, and use d_vec to condition the MDTA (e.g. FiLM like)
        # For now, let's follow the spirit of Q_b, K_d, V_d for an attention mechanism.
        # The MDTA expects Q,K,V to be derived from a single input X.
        # To do cross attention: Q=Conv(X_q), K=Conv(X_kv), V=Conv(X_kv)
        # Here X_q = basic_prompt_b, X_kv = degradation_prompt_d (spatially tiled)
        
        # Let's use a simpler P2P for now:
        # Project d_vec to match channels of basic_prompt_b, add/concat, then process.
        degrad_conditioning = self.kv_proj_degrad(degradation_prompt_d_vec) # [B, C_basic*2]
        degrad_scale, degrad_shift = degrad_conditioning.chunk(2, dim=1) # [B, C_basic]
        degrad_scale = degrad_scale.unsqueeze(-1).unsqueeze(-1) # [B, C_basic, 1, 1]
        degrad_shift = degrad_shift.unsqueeze(-1).unsqueeze(-1) # [B, C_basic, 1, 1]

        fused_prompt = basic_prompt_b * (1 + degrad_scale) + degrad_shift
        fused_prompt = self.attn(fused_prompt) # Apply self-attention on the conditioned basic prompt
        fused_prompt = self.ffn(fused_prompt)
        
        universal_prompt_u = self.out_proj(fused_prompt)
        return universal_prompt_u

class SelectivePromptToFeatureInteraction(nn.Module):
    """
    Modulates U-Net features (Z) with the universal restoration prompt (U).
    Inspired by PIP paper's P2F interaction using selective cross-attention.
    Query: U-Net features (Z)
    Key/Value: Universal prompt (U)
    """
    def __init__(self, feature_dim, universal_prompt_dim, num_heads=4, ffn_expansion_factor=2.66, bias=False): # top_m_ratio removed for now
        super().__init__()
        self.feature_dim = feature_dim
        self.universal_prompt_dim = universal_prompt_dim

        # If feature_dim is different from universal_prompt_dim, project features_z
        if feature_dim != universal_prompt_dim:
            self.feat_proj = nn.Conv2d(feature_dim, universal_prompt_dim, kernel_size=1, bias=bias)
        else:
            self.feat_proj = nn.Identity()

        concat_dim = universal_prompt_dim + universal_prompt_dim # After concatenating projected_z and prompt_u
        
        self.norm_in = LayerNorm(concat_dim) # Norm after concatenation
        self.attn = MDTA(dim=concat_dim, num_heads=num_heads, bias=bias)
        self.norm_mid = LayerNorm(concat_dim)
        self.ffn = GDFN(dim=concat_dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
        
        # Project back to original feature_dim
        self.out_conv = nn.Conv2d(concat_dim, feature_dim, kernel_size=3, padding=1, bias=bias)

    def forward(self, features_z, universal_prompt_u):
        """
        Args:
            features_z (torch.Tensor): U-Net features, shape [B, C_feat, H, W]
            universal_prompt_u (torch.Tensor): Universal prompt, shape [B, C_univ_prompt, H, W] 
                                              (spatially resized to match features_z)
        Returns:
            torch.Tensor: Modulated features_z_hat, shape [B, C_feat, H, W]
        """
        b, c_feat, h, w = features_z.shape
        
        projected_feat_z = self.feat_proj(features_z) # [B, C_univ_prompt, H, W]
        
        # Concatenate projected features and universal prompt
        concat_input = torch.cat([projected_feat_z, universal_prompt_u], dim=1) # [B, 2*C_univ_prompt, H, W]
        
        # Pass through Transformer-like block (Norm -> MDTA -> Norm -> GDFN)
        # Residual connection is typically around the attention and FFN blocks
        
        x = concat_input # Store for residual
        x = self.norm_in(x)
        x = self.attn(x) 
        x = x + concat_input # First residual connection (around attention)

        x_ffn_in = x # Store for second residual
        x = self.norm_mid(x)
        x = self.ffn(x)
        x = x + x_ffn_in # Second residual connection (around FFN)
        
        # Project back to original feature dimension
        modulated_output = self.out_conv(x) # [B, C_feat, H, W]
        
        return modulated_output + features_z # Final residual connection with original features_z


class PromptIR_V2(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 out_channels=3, 
                 base_dim=40, 
                 num_blocks_per_level=[3, 4, 4, 5], 
                 # PIP-style prompt params
                 num_degradations=2, # rain, snow
                 degradation_prompt_dim=64,
                 basic_prompt_c=64, basic_prompt_hw=16, # C, H, W for basic prompt
                 universal_prompt_dim=64, # Output of P2P, input to P2F
                 p2p_heads=4,
                 p2f_heads=4,
                 # General params
                 num_attn_heads=8, # For backbone transformer blocks
                 ffn_expansion_factor=2.66,
                 bias=False):
        super().__init__()

        self.base_dim = base_dim
        self.num_levels = len(num_blocks_per_level)

        # PIP Prompts
        self.degradation_aware_prompts = nn.Embedding(num_degradations, degradation_prompt_dim)
        self.basic_restoration_prompt_tensor = nn.Parameter(
            torch.randn(1, basic_prompt_c, basic_prompt_hw, basic_prompt_hw)
        ) # Shape [1, C_basic, H_basic, W_basic] for broadcasting

        self.p2p_interaction = PromptToPromptInteraction(
            basic_prompt_dim=basic_prompt_c,
            degradation_prompt_dim=degradation_prompt_dim,
            universal_prompt_dim=universal_prompt_dim,
            num_heads=p2p_heads,
            bias=bias
        )

        # Initial convolution
        self.initial_conv = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1, bias=bias)

        # Encoder
        self.encoder_levels = nn.ModuleList()
        self.encoder_dims = [] # To store output dims of encoder levels for skip connections
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
        
        # Decoder & P2F for skip connections
        self.decoder_levels = nn.ModuleList()
        self.p2f_interactions = nn.ModuleList()

        for i in range(self.num_levels - 1):
            skip_connection_dim = self.encoder_dims[self.num_levels - 1 - (i+1)]
            
            # P2F module for this skip connection
            # Universal prompt spatial dim should match skip feature spatial dim at this point.
            # This implies U needs to be generated once from basic_prompt_hw and then adapted, or P2F handles resizing.
            # For now, assume universal_prompt_dim is channel dim, and P2F will handle spatial.
            # The universal_prompt_u will have spatial H_basic, W_basic. Skip features have varying H,W.
            # P2F needs to handle this. Let P2F take feature_dim and prompt_channel_dim.
            self.p2f_interactions.append(
                SelectivePromptToFeatureInteraction(
                    feature_dim=skip_connection_dim, # Skip feature dim
                    universal_prompt_dim=universal_prompt_dim, # Channel dim of U
                    num_heads=p2f_heads,
                    bias=bias
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

    def forward(self, x, degradation_label): # degradation_label: [B], tensor of 0s or 1s
        b = x.shape[0]
        
        # 1. Generate Universal Prompt U using P2P
        selected_degrad_aware_prompt_vec = self.degradation_aware_prompts(degradation_label) # [B, degrad_prompt_dim]
        # Broadcast basic_restoration_prompt_tensor to batch size
        batch_basic_prompt = self.basic_restoration_prompt_tensor.repeat(b, 1, 1, 1)
        universal_prompt_u = self.p2p_interaction(batch_basic_prompt, selected_degrad_aware_prompt_vec)
        # universal_prompt_u has shape [B, C_universal, H_basic, W_basic]

        skip_connections = []
        # Encoder path
        x = self.initial_conv(x)
        enc_idx = 0
        for i in range(self.num_levels):
            x = self.encoder_levels[enc_idx](x)
            enc_idx += 1
            if i < self.num_levels - 1:
                skip_connections.append(x)
                x = self.encoder_levels[enc_idx](x)
                enc_idx += 1
        
        # Decoder path
        dec_idx = 0
        p2f_idx = 0
        for i in range(self.num_levels - 1):
            x = self.decoder_levels[dec_idx](x) # Upsample
            dec_idx += 1
            
            skip = skip_connections.pop()
            
            # Modulate skip connection with P2F
            # Need to ensure universal_prompt_u is spatially compatible with skip
            # Resize U to match skip's H, W
            h_skip, w_skip = skip.shape[2], skip.shape[3]
            universal_prompt_u_resized = F.interpolate(universal_prompt_u, size=(h_skip, w_skip), mode='bilinear', align_corners=False)
            
            modulated_skip = self.p2f_interactions[p2f_idx](skip, universal_prompt_u_resized)
            p2f_idx += 1
            
            x = torch.cat([x, modulated_skip], dim=1)
            x = self.decoder_levels[dec_idx](x) # Conv to merge
            dec_idx += 1
            
            x = self.decoder_levels[dec_idx](x) # Decoder Transformer blocks
            dec_idx += 1
            
        x = self.final_conv(x)
        return x

if __name__ == '__main__':
    img_channels = 3
    batch_s = 2

    # Test V2 model
    model_v2 = PromptIR_V2(
        in_channels=img_channels,
        out_channels=img_channels,
        base_dim=40, # Increased
        num_blocks_per_level=[2,2,2,2], # Simplified for test
        num_degradations=2,
        degradation_prompt_dim=64,
        basic_prompt_c=64, basic_prompt_hw=8, # Smaller basic prompt H,W for test
        universal_prompt_dim=64,
        p2p_heads=4,
        p2f_heads=4,
        num_attn_heads=4 # For backbone
    ).cuda()

    test_input = torch.randn(batch_s, img_channels, 256, 256).cuda()
    # 0 for rain, 1 for snow
    degrad_labels = torch.randint(0, 2, (batch_s,)).cuda() 
    
    print(f"Input shape: {test_input.shape}")
    print(f"Degradation labels: {degrad_labels}")

    # Test P2P
    p2p_module = model_v2.p2p_interaction
    basic_p_test = model_v2.basic_restoration_prompt_tensor.repeat(batch_s,1,1,1).cuda()
    degrad_p_vec_test = model_v2.degradation_aware_prompts(degrad_labels)
    print(f"Basic P shape for P2P: {basic_p_test.shape}")
    print(f"Degrad P vec shape for P2P: {degrad_p_vec_test.shape}")
    u_prompt_test = p2p_module(basic_p_test, degrad_p_vec_test)
    print(f"P2P Universal Prompt U shape: {u_prompt_test.shape}")
    assert u_prompt_test.shape == (batch_s, model_v2.p2p_interaction.universal_prompt_dim, model_v2.basic_restoration_prompt_tensor.shape[2], model_v2.basic_restoration_prompt_tensor.shape[3])

    # Test P2F (using one from the list)
    if model_v2.p2f_interactions:
        p2f_module = model_v2.p2f_interactions[0] # Test the first P2F
        # Dummy skip feature (e.g., from first skip connection)
        # Encoder dim for first skip: base_dim = 40
        dummy_skip_feat_c = model_v2.encoder_dims[model_v2.num_levels-2] # Matching the first skip connection
        dummy_skip_h, dummy_skip_w = 256//2, 256//2 # Example size for first skip
        dummy_skip_feat = torch.randn(batch_s, dummy_skip_feat_c, dummy_skip_h, dummy_skip_w).cuda()
        
        u_prompt_resized_for_p2f = F.interpolate(u_prompt_test, size=(dummy_skip_h, dummy_skip_w), mode='bilinear', align_corners=False)
        print(f"P2F input feature Z shape: {dummy_skip_feat.shape}")
        print(f"P2F input prompt U (resized) shape: {u_prompt_resized_for_p2f.shape}")
        
        modulated_feat_test = p2f_module(dummy_skip_feat, u_prompt_resized_for_p2f)
        print(f"P2F Modulated feature shape: {modulated_feat_test.shape}")
        assert modulated_feat_test.shape == dummy_skip_feat.shape
    
    # Full model forward pass
    output = model_v2(test_input, degrad_labels)
    print(f"PromptIR_V2 output shape: {output.shape}")
    assert output.shape == (batch_s, img_channels, 256, 256)
    
    print("PromptIR_V2 model and sub-module tests seem to pass shape checks.")

    # Calculate parameters
    total_params = sum(p.numel() for p in model_v2.parameters() if p.requires_grad)
    print(f"Total trainable parameters for PromptIR_V2 (test config): {total_params/1e6:.2f} M")
