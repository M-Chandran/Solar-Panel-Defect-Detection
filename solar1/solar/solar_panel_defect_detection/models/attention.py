import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    Focuses on 'where' the defect is located in the image.
    """
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        spatial = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        spatial = self.conv(spatial)  # [B, 1, H, W]
        return self.sigmoid(spatial)  # [B, 1, H, W]

class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    Focuses on 'what' defect features are important.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        avg_out = self.fc(self.avg_pool(x))  # [B, C, 1, 1]
        max_out = self.fc(self.max_pool(x))  # [B, C, 1, 1]
        out = avg_out + max_out  # [B, C, 1, 1]
        return self.sigmoid(out)  # [B, C, 1, 1]

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    Combines spatial and channel attention for enhanced feature representation.
    """
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention(in_channels)

    def forward(self, x):
        # x: [B, C, H, W]
        # Apply channel attention
        ca_out = self.channel_attention(x) * x  # [B, C, H, W]

        # Apply spatial attention
        sa_out = self.spatial_attention(ca_out) * ca_out  # [B, C, H, W]

        return sa_out

class CrossLayerAttentionFusion(nn.Module):
    """
    Cross-layer attention fusion for multi-scale feature integration.
    """
    def __init__(self, in_channels_list):
        super(CrossLayerAttentionFusion, self).__init__()
        self.attention_layers = nn.ModuleList([
            CBAM(channels) for channels in in_channels_list
        ])

        # Fusion layer
        total_channels = sum(in_channels_list)
        self.fusion_conv = nn.Conv2d(total_channels, in_channels_list[-1], kernel_size=1)

    def forward(self, feature_maps):
        """
        feature_maps: list of feature maps from different layers
        """
        # Apply attention to each feature map
        attended_features = []
        for i, features in enumerate(feature_maps):
            attended = self.attention_layers[i](features)
            attended_features.append(attended)

        # Upsample all features to the same size (largest feature map)
        target_size = attended_features[-1].shape[-2:]
        upsampled_features = []
        for features in attended_features[:-1]:
            upsampled = F.interpolate(features, size=target_size, mode='bilinear', align_corners=False)
            upsampled_features.append(upsampled)
        upsampled_features.append(attended_features[-1])

        # Concatenate and fuse
        concatenated = torch.cat(upsampled_features, dim=1)
        fused = self.fusion_conv(concatenated)

        return fused

class DefectAttentionEncoder(nn.Module):
    """
    Enhanced ResNet encoder with attention mechanisms for defect detection.
    """
    def __init__(self, backbone='resnet50', pretrained=False):
        super(DefectAttentionEncoder, self).__init__()

        if backbone == 'resnet50':
            from torchvision.models import resnet50
            self.backbone = resnet50(pretrained=pretrained)

            # Extract intermediate layers
            self.layer1 = nn.Sequential(*list(self.backbone.children())[:5])  # conv1 + bn1 + relu + maxpool + layer1
            self.layer2 = list(self.backbone.children())[5]  # layer2
            self.layer3 = list(self.backbone.children())[6]  # layer3
            self.layer4 = list(self.backbone.children())[7]  # layer4

            # Attention modules for each layer
            self.attn1 = CBAM(256)   # layer1 output channels
            self.attn2 = CBAM(512)   # layer2 output channels
            self.attn3 = CBAM(1024)  # layer3 output channels
            self.attn4 = CBAM(2048)  # layer4 output channels

            # Cross-layer fusion
            self.cross_fusion = CrossLayerAttentionFusion([256, 512, 1024, 2048])

        else:
            raise NotImplementedError(f"Backbone {backbone} not supported")

    def forward(self, x):
        # Forward through layers with attention
        x1 = self.attn1(self.layer1(x))      # [B, 256, H/4, W/4]
        x2 = self.attn2(self.layer2(x1))     # [B, 512, H/8, W/8]
        x3 = self.attn3(self.layer3(x2))     # [B, 1024, H/16, W/16]
        x4 = self.attn4(self.layer4(x3))     # [B, 2048, H/32, W/32]

        # Cross-layer fusion
        feature_maps = [x1, x2, x3, x4]
        fused_features = self.cross_fusion(feature_maps)

        # Generate attention maps for explainability
        attention_maps = {
            'layer1': torch.mean(x1, dim=1, keepdim=True),  # [B, 1, H/4, W/4]
            'layer2': torch.mean(x2, dim=1, keepdim=True),  # [B, 1, H/8, W/8]
            'layer3': torch.mean(x3, dim=1, keepdim=True),  # [B, 1, H/16, W/16]
            'layer4': torch.mean(x4, dim=1, keepdim=True),  # [B, 1, H/32, W/32]
            'fused': torch.mean(fused_features, dim=1, keepdim=True)  # [B, 1, H/32, W/32]
        }

        return fused_features, attention_maps

class MultiHeadAttentionFusion(nn.Module):
    """
    Multi-head attention for fusing different feature representations.
    """
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadAttentionFusion, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value):
        # query, key, value: [B, N, C] where N is sequence length, C is embed_dim
        attn_output, _ = self.multihead_attn(query, key, value)
        output = self.norm(attn_output + query)  # Residual connection
        output = self.dropout(output)
        return output

if __name__ == "__main__":
    # Test attention modules
    print("Testing attention modules...")

    # Test CBAM
    cbam = CBAM(512)
    x = torch.randn(2, 512, 14, 14)
    out = cbam(x)
    print(f"CBAM input: {x.shape}, output: {out.shape}")

    # Test CrossLayerAttentionFusion
    fusion = CrossLayerAttentionFusion([256, 512, 1024, 2048])
    features = [
        torch.randn(2, 256, 56, 56),
        torch.randn(2, 512, 28, 28),
        torch.randn(2, 1024, 14, 14),
        torch.randn(2, 2048, 7, 7)
    ]
    fused = fusion(features)
    print(f"Cross-layer fusion output: {fused.shape}")

    # Test DefectAttentionEncoder
    encoder = DefectAttentionEncoder()
    x = torch.randn(2, 3, 224, 224)
    features, attn_maps = encoder(x)
    print(f"Encoder output: {features.shape}")
    print(f"Attention maps keys: {list(attn_maps.keys())}")

    print("All attention modules tested successfully!")
