# models/LPETNet.py
"""
LPETNet (Light Patch Embedding Transformer Network) for lip reading.

This module implements the LPETNet architecture, which combines CNN and transformer pathways
for efficient video-based speech recognition. The model uses depthwise separable 3D convolutions
for the CNN pathway and a lightweight transformer with linear attention for the transformer
pathway, achieving computational efficiency while maintaining performance.
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange

class EfficientPatchEmbedding(nn.Module):
    """Efficient patch embedding with larger patches and stride.
    
    Creates patch embeddings from the input video frames using 3D convolution with
    larger patch sizes and strides to reduce computational overhead.
    
    Args:
        in_channels (int): Number of input channels (typically 3 for RGB).
        patch_size (int): Size of each patch (height and width).
        stride (int): Stride between patches.
        embed_dim (int): Embedding dimension for each patch.
    """
    def __init__(self, in_channels=3, patch_size=12, stride=12, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # Larger stride reduces overlapping computation
        self.projection = nn.Conv3d(in_channels, embed_dim, 
                                   kernel_size=(1, patch_size, patch_size), 
                                   stride=(1, stride, stride))
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """Process input through patch embedding.
        
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
        
        # Project patches
        x = self.projection(x)  # Fewer patches due to larger stride
        
        # Rearrange to sequence of patches
        x = rearrange(x, 'b e t h w -> (b t) (h w) e')
        x = self.norm(x)
        
        return x, T

class LinearAttention(nn.Module):
    """Linear attention mechanism - more efficient than quadratic attention.
    
    Implements attention with linear complexity O(N) instead of quadratic O(N²)
    with respect to sequence length, using kernel trick approximation.
    
    Args:
        dim (int): Input dimension.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
    """
    def __init__(self, dim, num_heads=4, dropout=0.1):
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
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, num_heads, N, head_dim]
        
        # Linear attention (avoid O(N²) complexity)
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # Key-Value aggregation: O(N) instead of O(N²)
        kv = torch.einsum('bhnd,bhne->bhde', k, v)

        # Query multiplication: still O(N) 
        out = torch.einsum('bhnd,bhde->bhne', q, kv)
        
        # Normalize
        z = 1.0 / (torch.einsum('bhnd,bhd->bhn', q, k.sum(dim=2)) + 1e-8)
        out = out * z.unsqueeze(-1)
        
        # Reshape and project
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj(out)
        out = self.dropout(out)
        
        return out

class LightTransformerBlock(nn.Module):
    """Lightweight transformer block with linear attention.
    
    A simplified and more efficient transformer block that uses linear attention
    and a smaller MLP ratio for reduced computation.
    
    Args:
        dim (int): Input dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Multiplier for MLP hidden dimension.
        dropout (float): Dropout probability.
    """
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LinearAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        
        # Smaller MLP ratio (2 instead of 4)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """Process input through transformer block.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, C].
                B: batch size, N: sequence length, C: channels
                
        Returns:
            torch.Tensor: Transformed output of shape [B, N, C]
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class DepthwiseSeparableConv3d(nn.Module):
    """Depthwise separable 3D convolution for efficiency.
    
    Implements 3D convolution as a depthwise convolution (per-channel) followed 
    by a pointwise (1x1x1) convolution, significantly reducing parameters.
    
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

class LPETNet(nn.Module):
    """Light Patch Embedding Transformer Network for lip reading.
    
    This network combines a CNN pathway using depthwise separable 3D convolutions
    and a transformer pathway using lightweight linear attention. The features
    from both pathways are fused and processed through bidirectional GRUs for
    sequence modeling, optimized for video-based speech recognition.
    
    Args:
        dropout_p (float): Dropout probability for CNN and sequence models.
        transformer_dropout (float): Dropout probability for transformer components.
        num_classes (int): Number of output classes (typically characters+blank).
    """
    def __init__(self, dropout_p=0.5, transformer_dropout=0.1, num_classes=28):
        super(LPETNet, self).__init__()
        
        # Depthwise separable convolutions for CNN pathway
        self.conv1 = DepthwiseSeparableConv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))       
        self.conv2 = DepthwiseSeparableConv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))        
        self.conv3 = DepthwiseSeparableConv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
        # Efficient transformer pathway
        self.patch_embed = EfficientPatchEmbedding(in_channels=3, patch_size=12, stride=12, embed_dim=64)
        
        # Use transformer_dropout instead of dropout_p for transformer blocks
        self.transformer = LightTransformerBlock(dim=64, num_heads=2, dropout=transformer_dropout)
        
       # Global token for transformer pathway
        self.global_token = nn.Parameter(torch.zeros(1, 1, 64))

        # Feature dimensions
        self.cnn_feature_dim = 96 * 4 * 8
        self.transformer_feature_dim = 64

        # Feature fusion
        self.fusion = nn.Linear(self.cnn_feature_dim + self.transformer_feature_dim, 256)

        # Sequence modeling with bidirectional GRUs
        self.gru1 = nn.GRU(256, 256, 1, bidirectional=True)
        self.gru2 = nn.GRU(512, 256, 1, bidirectional=True)

        # Classification head
        self.FC = nn.Linear(512, num_classes)
        
        # Store dropout rates
        self.dropout_p = dropout_p
        self.transformer_dropout = transformer_dropout
        
        # Utility layers
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout3d = nn.Dropout3d(self.dropout_p)
        
        # Initialize parameters
        self._init()
    
    def _init(self):
        """Initialize model parameters.
        
        Applies appropriate weight initialization techniques for different components:
        - Kaiming initialization for CNN and projection layers
        - Normal initialization for global token
        - Uniform and orthogonal initialization for GRU layers
        """
        # Initialize CNN layers
        for m in [self.conv1.depthwise, self.conv1.pointwise, 
                  self.conv2.depthwise, self.conv2.pointwise,
                  self.conv3.depthwise, self.conv3.pointwise]:
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        
        # Initialize projection layer
        init.kaiming_normal_(self.patch_embed.projection.weight)
        init.constant_(self.patch_embed.projection.bias, 0)
        
        # Initialize global token
        init.normal_(self.global_token, std=0.02)
        
        # Initialize fusion layer
        init.kaiming_normal_(self.fusion.weight)
        init.constant_(self.fusion.bias, 0)
        
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
        
        Processes input through CNN and transformer pathways, fuses the features,
        and passes through sequence models to generate predictions.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T, H, W].
                B: batch size, C: channels, T: time/frames, H: height, W: width
                
        Returns:
            torch.Tensor: Output logits of shape [B, T, num_classes]
                formatted for CTC loss
        """
        # Process through 3D CNN pathway
        cnn_features = self.conv1(x)
        cnn_features = self.relu(cnn_features)
        cnn_features = self.dropout3d(cnn_features)
        cnn_features = self.pool1(cnn_features)
        
        cnn_features = self.conv2(cnn_features)
        cnn_features = self.relu(cnn_features)
        cnn_features = self.dropout3d(cnn_features)
        cnn_features = self.pool2(cnn_features)
        
        cnn_features = self.conv3(cnn_features)
        cnn_features = self.relu(cnn_features)
        cnn_features = self.dropout3d(cnn_features)
        cnn_features = self.pool3(cnn_features)
        
        # Process through transformer pathway (fewer patches)
        patch_features, T = self.patch_embed(x)
        
        # Add global token 
        B = patch_features.shape[0] // T
        global_tokens = self.global_token.expand(B * T, -1, -1)
        patch_features = torch.cat((global_tokens, patch_features), dim=1)
        
        # Apply single lightweight transformer block
        patch_features = self.transformer(patch_features)
        
        # Extract global token features
        global_features = patch_features[:, 0]  # [B*T, embed_dim]
        global_features = rearrange(global_features, '(b t) d -> t b d', t=T)
        
        # Reshape CNN features for sequence modeling
        cnn_features = cnn_features.permute(2, 0, 1, 3, 4).contiguous()  # [T, B, C, H, W]
        cnn_features = cnn_features.view(cnn_features.size(0), cnn_features.size(1), -1)  # [T, B, C*H*W]
        
        # Combine CNN and transformer features
        combined_features = torch.cat([cnn_features, global_features], dim=2)
        fused_features = self.fusion(combined_features)
        fused_features = self.relu(fused_features)
        fused_features = self.dropout(fused_features)
        
        # Process through BiGRU layers (sequence modeling)
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        
        gru_out, _ = self.gru1(fused_features)
        gru_out = self.dropout(gru_out)
        
        gru_out, _ = self.gru2(gru_out)
        gru_out = self.dropout(gru_out)
        
        # Final classification
        output = self.FC(gru_out)
        
        # Reshape for CTC loss: [B, T, C]
        output = output.permute(1, 0, 2).contiguous()
        
        return output
