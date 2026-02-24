import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import ResNet, Bottleneck
import math


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation layer for channel attention.
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBottleneck(Bottleneck):
    """
    Bottleneck block with Squeeze-and-Excitation.
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction=16):
        super(SEBottleneck, self).__init__(inplanes, planes, stride, downsample,
                                           groups, base_width, dilation, norm_layer)
        self.se = SELayer(planes * 4, reduction)


class SEResNet50(ResNet):
    """
    SE-ResNet50 with Squeeze-and-Excitation blocks.
    """
    def __init__(self, pretrained=True, num_classes=1000):
        super(SEResNet50, self).__init__(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
        
        if pretrained:
            # Load pretrained ResNet50 weights
            try:
                pretrained_model = models.resnet50(pretrained=True)
                self.load_state_dict(pretrained_model.state_dict(), strict=False)
                print("Loaded pretrained ResNet50 weights")
            except Exception as e:
                print(f"Could not load pretrained weights: {e}")


class MoCo(nn.Module):
    """
    Momentum Contrast (MoCo v2) for self-supervised learning.
    Implements the framework from the research paper with SE-ResNet50 encoder.
    """
    def __init__(self, dim=128, K=4096, m=0.999, T=0.2, arch='se_resnet50', pretrained=True):
        """
        dim: feature dimension (default: 128, as per paper)
        K: queue size; number of negative keys (default: 4096, as per paper)
        m: moco momentum of updating key encoder (default: 0.999, as per paper)
        T: softmax temperature (default: 0.2, as per paper)
        arch: encoder architecture (default: 'se_resnet50')
        pretrained: use pretrained ImageNet weights
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # Create the encoders
        self.encoder_q = self._build_encoder(arch, dim, pretrained)
        self.encoder_k = self._build_encoder(arch, dim, pretrained)

        # Initialize key encoder with query encoder parameters
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # Create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _build_encoder(self, arch, dim, pretrained):
        """Build the encoder with projection head (2-layer MLP with ReLU)."""
        if arch == 'se_resnet50':
            # Use SE-ResNet50 as backbone (as per paper)
            backbone = SEResNet50(pretrained=pretrained, num_classes=0)
            # Remove the classification layer, keep feature extraction
            backbone = nn.Sequential(*list(backbone.children())[:-1])
            feat_dim = 2048
        elif arch == 'resnet50':
            # Standard ResNet50
            backbone = models.resnet50(pretrained=pretrained)
            backbone = nn.Sequential(*list(backbone.children())[:-1])
            feat_dim = 2048
        else:
            raise NotImplementedError(f"Architecture {arch} not supported")

        # Projection head: 2-layer MLP with ReLU (as per MoCo v2)
        projection_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, dim)
        )

        encoder = nn.Sequential(
            backbone,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            projection_head
        )

        return encoder


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN (only in distributed mode)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits using InfoNCE loss formulation
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature (tau=0.2 as per paper)
        logits /= self.T

        # labels: positive key indicators (0 for all, since first column is positive)
        device = logits.device
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)


        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

    def extract_features(self, x):
        """
        Extract features for KNN classification (without projection head).
        Uses the backbone features before the projection head.
        """
        # Get backbone features (before projection head)
        for name, module in self.encoder_q.named_children():
            if name == '0':  # backbone
                features = module(x)
            elif name == '1':  # AdaptiveAvgPool2d
                features = module(features)
            elif name == '2':  # Flatten
                features = module(features)
            # Skip projection head (3)
        
        return nn.functional.normalize(features, dim=1)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
    else:
        output = tensor
    return output


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for contrastive learning.
    As described in the paper with temperature scaling.
    """
    def __init__(self, temperature=0.2):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        """
        Args:
            logits: [N, K+1] where first column is positive pair
            labels: [N] all zeros (positive is always at index 0)
        """
        return self.criterion(logits / self.temperature, labels)


class AttentionEncoder(nn.Module):
    """
    ViT with attention mechanisms for defect-sensitive features.
    """
    def __init__(self, dim=768):
        super(AttentionEncoder, self).__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)

        # Spatial Attention (adapted for ViT patch tokens)
        self.spatial_attention = nn.Sequential(
            nn.Linear(768, 197),  # 197 = 14*14 + 1 (CLS token)
            nn.Sigmoid()
        )

        # Channel Attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Linear(768, 768 // 16),
            nn.ReLU(),
            nn.Linear(768 // 16, 768),
            nn.Sigmoid()
        )

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(768, 2048),
            nn.ReLU(),
            nn.Linear(2048, dim)
        )

    def forward(self, x):
        # Extract features (patch tokens + CLS)
        features = self.backbone.forward_features(x)  # [B, 197, 768]

        # Apply spatial attention to patch tokens
        spatial_attn = self.spatial_attention(features.mean(dim=-1))  # [B, 197]
        spatial_attn = spatial_attn.unsqueeze(-1)  # [B, 197, 1]

        # Apply channel attention
        channel_attn = self.channel_attention(features.transpose(1, 2))  # [B, 768, 1]
        channel_attn = channel_attn.transpose(1, 2)  # [B, 1, 768]

        # Combine attentions
        attended_features = features * spatial_attn * channel_attn

        # Use CLS token for embedding
        cls_embedding = attended_features[:, 0, :]  # [B, 768]

        # Project to embedding space
        embedding = self.projection(cls_embedding)
        embedding = nn.functional.normalize(embedding, dim=1)

        return embedding, attended_features, spatial_attn.squeeze(-1), channel_attn.squeeze(1)

class DINO(nn.Module):
    """
    DINO: Emerging Properties in Self-Supervised Vision Transformers
    """
    def __init__(self, student, teacher, dim=768, momentum_teacher=0.996):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.momentum_teacher = momentum_teacher
        self.dim = dim

        # Initialize teacher with student parameters
        self._init_teacher()

    def _init_teacher(self):
        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False

    @torch.no_grad()
    def update_teacher(self, momentum):
        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data = param_t.data * momentum + param_s.data * (1. - momentum)

    def forward(self, student_crops, teacher_crops):
        """
        student_crops: list of augmented crops for student
        teacher_crops: list of augmented crops for teacher (usually 2 global crops)
        """
        # Student forward
        student_outputs = [self.student(crop) for crop in student_crops]
        student_outputs = torch.stack(student_outputs, dim=0)  # [n_crops, B, dim]

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_outputs = [self.teacher(crop) for crop in teacher_crops]
            teacher_outputs = torch.stack(teacher_outputs, dim=0)  # [n_global_crops, B, dim]

        return student_outputs, teacher_outputs

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        torch.distributed.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * torch.distributed.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)

class DINOHead(nn.Sequential):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    # Test MoCo model with SE-ResNet50
    print("Testing MoCo with SE-ResNet50...")
    model = MoCo(dim=128, K=4096, m=0.999, T=0.2, arch='se_resnet50', pretrained=False)
    print("MoCo model created successfully")

    # Test forward pass
    im_q = torch.randn(4, 3, 224, 224)
    im_k = torch.randn(4, 3, 224, 224)
    
    logits, labels = model(im_q, im_k)
    print(f"Logits shape: {logits.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Test feature extraction for KNN
    features = model.extract_features(im_q)
    print(f"Extracted features shape: {features.shape}")
    
    print("All tests passed!")
