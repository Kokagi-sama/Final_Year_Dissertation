# models/ALPETNet.py
"""
ALPETNet (Attention-reinforced Light Patch Embedding Transformer Network) for lip reading.

This module implements the ALPETNet architecture, which enhances the LPETNet model with
multi-scale patch embedding, gated transformer blocks, and cross-attention fusion.
The model combines an enhanced CNN pathway with channel attention and a transformer
pathway with efficient linear attention, optimized for video-based speech recognition.
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from einops import rearrange

class EfficientMultiScalePatchEmbedding(nn.Module):
    """Multi-scale patch embedding with three different patch sizes.
    
    This module extracts patches at different scales from input video frames,
    creating a multi-scale representation that captures both fine and coarse details.
    It uses three separate convolutional pathways with different kernel sizes and strides.
    
    Args:
        in_channels (int): Number of input channels (typically 3 for RGB).
        embed_dim (int): Total embedding dimension, split across the three scales.
    """
    def __init__(self, in_channels=3, embed_dim=64):
        super().__init__()

        # Split embedding dimension across scales
        small_dim = embed_dim // 3
        medium_dim = embed_dim // 3
        large_dim = embed_dim - small_dim - medium_dim
        
        # Three different patch scales with different receptive fields
        self.embed_small = nn.Conv3d(in_channels, small_dim,
                                    kernel_size=(1, 4, 4), stride=(1, 4, 4))
        self.embed_medium = nn.Conv3d(in_channels, medium_dim,
                                     kernel_size=(1, 8, 8), stride=(1, 8, 8))
        self.embed_large = nn.Conv3d(in_channels, large_dim,
                                    kernel_size=(1, 12, 12), stride=(1, 12, 12))
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """Process input through multi-scale patch embedding.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T, H, W].
                B: batch size, C: channels, T: time/frames, H: height, W: width
                
        Returns:
            tuple: (embedded_patches, T)
                - embedded_patches: Tensor of shape [(B*T), (H*W), embed_dim]
                - T: Number of frames
        """
        # x shape: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        
        # Extract multi-scale patches
        x_small = self.embed_small(x) # Smallest patch size (4×4)
        x_medium = self.embed_medium(x) # Medium patch size (8×8)
        x_large = self.embed_large(x) # Largest patch size (12×12)
        
        # Resize to match dimensions
        target_h, target_w = x_large.shape[3], x_large.shape[4]  # Use largest stride as target
        if x_small.shape[3:] != (target_h, target_w):
            x_small = F.interpolate(x_small, size=(T, target_h, target_w), mode='nearest')
        if x_medium.shape[3:] != (target_h, target_w):
            x_medium = F.interpolate(x_medium, size=(T, target_h, target_w), mode='nearest')
        
        # Concatenate along embedding dimension
        x_concat = torch.cat([x_small, x_medium, x_large], dim=1)
        
        # Reshape to sequence format
        x_concat = rearrange(x_concat, 'b e t h w -> (b t) (h w) e')
        x_concat = self.norm(x_concat)
        
        return x_concat, T

class LinearAttention(nn.Module):
    """Linear attention mechanism with O(N) complexity.
    
    Implements an efficient attention mechanism that scales linearly with sequence length
    instead of quadratically, using a kernel trick approximation with ELU+1 feature maps.
    
    Args:
        dim (int): Input feature dimension.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
    """
    def __init__(self, dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """Apply linear attention to input.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, C].
                B: batch size, N: sequence length, C: channels
                
        Returns:
            torch.Tensor: Output after attention of shape [B, N, C]
        """
        B, N, C = x.shape
        
        # Create query, key, value projections
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply ELU+1 feature map for numerical stability
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # Linear attention computation (O(N) complexity)
        kv = torch.einsum('bhnd,bhne->bhde', k, v) # First aggregate key-value pairs
        out = torch.einsum('bhnd,bhde->bhne', q, kv) # Then apply query
        
        # Normalize
        z = 1.0 / (torch.einsum('bhnd,bhd->bhn', q, k.sum(dim=2)) + 1e-8)
        out = out * z.unsqueeze(-1)
        
        # Reshape and project
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj(out)
        out = self.dropout(out)
        
        return out

class GatedTransformerBlock(nn.Module):
    """Enhanced transformer block with built-in gating mechanism.
    
    Extends the standard transformer block with a learnable gating parameter
    that controls information flow from the MLP path, allowing the model to
    adaptively balance feature retention and transformation.
    
    Args:
        dim (int): Input feature dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Multiplier for MLP hidden dimension.
        dropout (float): Dropout probability.
    """
    def __init__(self, dim, num_heads=2, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LinearAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP with gating mechanism
        hidden_dim = int(dim * mlp_ratio)
        self.mlp_gate = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Learnable gate parameter
        self.gate = nn.Parameter(torch.ones(1, 1, dim) * 0.5)  # Initialize at 0.5
        self.gate_activation = nn.Sigmoid()
        
    def forward(self, x):
        """Process input through transformer block with gating.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, C].
                B: batch size, N: sequence length, C: channels
                
        Returns:
            torch.Tensor: Transformed output of shape [B, N, C]
        """
        # Attention with residual connection
        attn_output = self.attn(self.norm1(x))
        x = x + attn_output
        
        # Gated MLP
        mlp_output = self.mlp_gate(self.norm2(x))
        gate_value = self.gate_activation(self.gate) # Dynamic gate value
        x = x + gate_value * mlp_output # Apply gating to MLP output
        
        return x

class ChannelAttention(nn.Module):
    """Squeeze-and-excitation module for channel attention.
    
    Applies channel-wise attention to emphasize important feature channels
    and suppress less relevant ones, using a squeeze-and-excitation approach.
    
    Args:
        in_channels (int): Number of input channels.
        reduction_ratio (int): Ratio for reducing channel dimension in the bottleneck.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1) # Global average pooling
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """Apply channel attention to input.
        
        Args:
            x (torch.Tensor): Input tensor.
                
        Returns:
            torch.Tensor: Channel-weighted output tensor.
        """
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # Squeeze operation
        y = self.fc(y).view(b, c, 1, 1, 1) # Excitation operation
        return x * y.expand_as(x) # Scale input features

class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion that lets transformer features guide CNN features.
    
    Uses transformer features to compute attention weights that are applied to
    CNN features, enabling the model to focus on the most relevant spatial regions
    based on semantic understanding from the transformer pathway.
    
    Args:
        cnn_dim (int): Dimension of CNN features.
        transformer_dim (int): Dimension of transformer features.
        fusion_dim (int): Output dimension after fusion.
    """
    def __init__(self, cnn_dim, transformer_dim, fusion_dim=256):
        super().__init__()
        self.cnn_proj = nn.Linear(cnn_dim, fusion_dim)
        self.transformer_q = nn.Linear(transformer_dim, fusion_dim)
        self.transformer_k = nn.Linear(transformer_dim, fusion_dim)
        self.scale = fusion_dim ** -0.5  # Scaling factor for attention
        self.out_proj = nn.Linear(fusion_dim, fusion_dim)
        
    def forward(self, cnn_features, transformer_features):
        """Fuse CNN and transformer features using cross-attention.
        
        Args:
            cnn_features (torch.Tensor): CNN pathway features of shape [T, B, cnn_dim].
            transformer_features (torch.Tensor): Transformer pathway features of shape [T, B, transformer_dim].
                
        Returns:
            torch.Tensor: Fused features of shape [T, B, fusion_dim].
        """
        # Shapes: cnn_features [T, B, cnn_dim], transformer_features [T, B, transformer_dim]
        T, B, _ = cnn_features.shape
        
        # Project CNN features
        cnn_proj = self.cnn_proj(cnn_features)  # [T, B, fusion_dim]
        
        # Create query and key from transformer features
        q = self.transformer_q(transformer_features)  # [T, B, fusion_dim]
        k = self.transformer_k(transformer_features)  # [T, B, fusion_dim]
        
        # Reshape for batch matrix multiplication
        q = q.view(T * B, 1, -1)  # [T*B, 1, fusion_dim]
        k = k.view(T * B, 1, -1)  # [T*B, 1, fusion_dim]
        cnn_flat = cnn_proj.view(T * B, 1, -1)  # [T*B, 1, fusion_dim]
        
        # Compute attention weights
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale  # [T*B, 1, 1]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to CNN features
        out = cnn_flat * attn  # [T*B, 1, fusion_dim]
        out = out.view(T, B, -1)  # [T, B, fusion_dim]
        
        return self.out_proj(out) # Final projection

class DepthwiseSeparableConv3d(nn.Module):
    """Depthwise separable 3D convolution for efficiency.
    
    Implements 3D convolution as a depthwise convolution (per-channel) followed 
    by a pointwise (1x1x1) convolution, significantly reducing parameters and
    computational cost compared to standard 3D convolutions.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (tuple): Size of the convolution kernel.
        stride (int or tuple): Stride of the convolution.
        padding (int or tuple): Padding added to all sides of the input.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        """Apply depthwise separable convolution to input.
        
        Args:
            x (torch.Tensor): Input tensor.
                
        Returns:
            torch.Tensor: Output after convolution.
        """
        x = self.depthwise(x) # Apply depthwise convolution
        x = self.pointwise(x) # Apply pointwise convolution
        return x

class ALPETNet(nn.Module):
    """Attention-reinforced Light Patch Embedding Transformer Network for lip reading.
    
    This network combines an enhanced CNN pathway using depthwise separable 3D convolutions
    with channel attention, and a transformer pathway using multi-scale patch embedding
    and gated transformer blocks. Features from both pathways are fused using cross-attention
    and processed through bidirectional GRUs for sequence modeling.
    
    Args:
        dropout_p (float): Dropout probability for CNN and sequence models.
        transformer_dropout (float): Dropout probability for transformer components.
        num_classes (int): Number of output classes (typically characters+blank).
    """
    def __init__(self, dropout_p=0.5, transformer_dropout=0.1, num_classes=28):
        super(ALPETNet, self).__init__()
        
        # CNN pathway
        self.conv1 = DepthwiseSeparableConv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.channel_attn1 = ChannelAttention(32)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
        self.conv2 = DepthwiseSeparableConv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.channel_attn2 = ChannelAttention(64)
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
        self.conv3 = DepthwiseSeparableConv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.channel_attn3 = ChannelAttention(96)
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
        # Enhanced transformer pathway
        self.transformer_embed_dim = 64
        
        # Multi-scale patch embedding
        self.patch_embed = EfficientMultiScalePatchEmbedding(
            in_channels=3, embed_dim=self.transformer_embed_dim
        )
        
        # Global token for feature aggregation
        self.global_token = nn.Parameter(torch.zeros(1, 1, self.transformer_embed_dim))
        
        # Gated transformer block
        self.transformer = GatedTransformerBlock(
            dim=self.transformer_embed_dim,
            num_heads=2,
            mlp_ratio=2.0,
            dropout=transformer_dropout
        )
        
        # Feature dimensions
        self.cnn_feature_dim = 96 * 4 * 8
        
        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(
            cnn_dim=self.cnn_feature_dim,
            transformer_dim=self.transformer_embed_dim,
            fusion_dim=256
        )
        
        # Sequence modeling with bidirectional GRUs
        self.gru1 = nn.GRU(256, 256, 1, bidirectional=True)
        self.gru2 = nn.GRU(512, 256, 1, bidirectional=True)

        # Classification head
        self.FC = nn.Linear(512, num_classes)
        
        # Utility layers
        self.dropout_p = dropout_p
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout3d = nn.Dropout3d(self.dropout_p)
        
        # Initialize parameters
        self._init()
    
    def _init(self):
        """Initialize model parameters with appropriate weight initialization schemes.
        
        Applies Kaiming initialization to CNN layers, normal initialization to the
        global token, and specialized initialization for GRU layers to improve
        training stability and convergence.
        """
        # Initialize CNN layers
        for m in [self.conv1.depthwise, self.conv1.pointwise,
                 self.conv2.depthwise, self.conv2.pointwise,
                 self.conv3.depthwise, self.conv3.pointwise]:
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        
        # Initialize global token
        init.normal_(self.global_token, std=0.02)
        
        # Initialize classification head
        init.kaiming_normal_(self.FC.weight, nonlinearity='sigmoid')
        init.constant_(self.FC.bias, 0)
        
        # Initialize GRU layers
        for m in (self.gru1, self.gru2):
            stdv = math.sqrt(2 / (256 + 256))
            for i in range(0, 256 * 3, 256):
                init.uniform_(m.weight_ih_l0[i: i + 256],
                            -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0[i: i + 256])
                init.constant_(m.bias_ih_l0[i: i + 256], 0)
                init.uniform_(m.weight_ih_l0_reverse[i: i + 256],
                            -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0_reverse[i: i + 256])
                init.constant_(m.bias_ih_l0_reverse[i: i + 256], 0)
    
    def forward(self, x):
        """Forward pass through the network.
        
        Processes input through CNN and transformer pathways, fuses the features
        with cross-attention, and passes through sequence models to generate predictions.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T, H, W].
                B: batch size, C: channels, T: time/frames, H: height, W: width
                
        Returns:
            torch.Tensor: Output logits of shape [B, T, num_classes]
                formatted for CTC loss
        """
        # Process through CNN pathway with channel attention
        cnn_features = self.conv1(x)
        cnn_features = self.channel_attn1(cnn_features) # Apply channel attention
        cnn_features = self.relu(cnn_features)
        cnn_features = self.dropout3d(cnn_features)
        cnn_features = self.pool1(cnn_features)
        
        cnn_features = self.conv2(cnn_features)
        cnn_features = self.channel_attn2(cnn_features) # Apply channel attention
        cnn_features = self.relu(cnn_features)
        cnn_features = self.dropout3d(cnn_features)
        cnn_features = self.pool2(cnn_features)
        
        cnn_features = self.conv3(cnn_features)
        cnn_features = self.channel_attn3(cnn_features) # Apply channel attention
        cnn_features = self.relu(cnn_features)
        cnn_features = self.dropout3d(cnn_features)
        cnn_features = self.pool3(cnn_features)
        
        # Process through transformer pathway
        patch_features, T = self.patch_embed(x) # Extract multi-scale patches
        
        # Add global token
        B = patch_features.shape[0] // T
        global_tokens = self.global_token.expand(B * T, -1, -1)
        patch_features = torch.cat((global_tokens, patch_features), dim=1)
        
        # Apply transformer block
        patch_features = self.transformer(patch_features)
        
        # Extract global token features
        global_features = patch_features[:, 0]  # [B*T, embed_dim]
        global_features = rearrange(global_features, '(b t) d -> t b d', t=T)
        
        # Reshape CNN features for sequence modeling
        cnn_features = cnn_features.permute(2, 0, 1, 3, 4).contiguous()  # [T, B, C, H, W]
        cnn_features = cnn_features.view(cnn_features.size(0), cnn_features.size(1), -1)  # [T, B, C*H*W]
        
        # Apply cross-attention fusion where transformer guides CNN
        fused_features = self.fusion(cnn_features, global_features)
        fused_features = self.relu(fused_features)
        fused_features = self.dropout(fused_features)
        
        # Process through BiGRU layers
        self.gru1.flatten_parameters() # Optimize memory layout
        self.gru2.flatten_parameters() # Optimize memory layout
        
        gru_out, _ = self.gru1(fused_features)
        gru_out = self.dropout(gru_out)
        
        gru_out, _ = self.gru2(gru_out)
        gru_out = self.dropout(gru_out)
        
        # Final classification
        output = self.FC(gru_out)
        
        # Reshape for CTC loss
        output = output.permute(1, 0, 2).contiguous()
        
        return output
