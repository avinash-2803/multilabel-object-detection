"""
mlcgcn.py
=========
Exact architecture matching the saved checkpoint.
Confirmed keys: backbone.stem/block*, cp2/3/4, cfm2/3/4,
fusion, lsm (A_hat/embed/gcn1.W/gcn2.W), dgn (pool_bn),
cls_r, cls_m
Fix: replaced F.softmax with torch.softmax for PyTorch 2.6 compatibility.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torch.nn import Parameter


def gen_adj(A):
    D = torch.pow(A.sum(1).float().clamp(min=1e-6), -0.5)
    D = torch.diag(D)
    return torch.matmul(torch.matmul(A, D).t(), D)


def _disable_inplace(module):
    for name, child in module.named_children():
        if isinstance(child, (nn.ReLU, nn.ReLU6)):
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            _disable_inplace(child)


# ── Backbone ──────────────────────────────────────────────────────────────────

class DenseNet169Backbone(nn.Module):
    C2 = 128
    C3 = 256
    C4 = 1664

    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import DenseNet169_Weights
        w  = DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None
        ft = tvm.densenet169(weights=w).features
        _disable_inplace(ft)
        self.stem   = nn.Sequential(ft.conv0, ft.norm0,
                                    nn.ReLU(inplace=False), ft.pool0)
        self.block1 = ft.denseblock1
        self.trans1 = ft.transition1
        self.block2 = ft.denseblock2
        self.trans2 = ft.transition2
        self.block3 = ft.denseblock3
        self.trans3 = ft.transition3
        self.block4 = ft.denseblock4
        self.norm5  = ft.norm5

    def forward(self, x):
        x  = self.stem(x)
        x  = self.block1(x);  F2 = self.trans1(x)
        x  = self.block2(F2); F3 = self.trans2(x)
        x  = self.block3(F3); x  = self.trans3(x)
        x  = self.block4(x)
        F4 = F.relu(self.norm5(x), inplace=False)
        return F2, F3, F4


# ── ContextInfoModule (cp2, cp3, cp4) ────────────────────────────────────────

class DilatedConvBranch(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        self.dconv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.pos_conv = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)

    def forward(self, x):
        E = self.dconv(x)
        return self.pos_conv(E) + E


class MaskedSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = math.sqrt(self.head_dim)
        self.W_Q = nn.Linear(dim, dim, bias=False)
        self.W_K = nn.Linear(dim, dim, bias=False)
        self.W_V = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N   = H * W
        tok = x.flatten(2).transpose(1, 2)
        Q   = self.W_Q(tok)
        K   = self.W_K(tok)
        V   = self.W_V(tok)

        def split_h(t):
            return t.view(B, N, self.num_heads,
                          self.head_dim).transpose(1, 2)

        Q, K, V = split_h(Q), split_h(K), split_h(V)

        # ✓ Fixed: torch.softmax instead of F.softmax
        attn = torch.softmax(
            torch.matmul(Q, K.transpose(-2, -1)) / self.scale, dim=-1)
        out  = torch.matmul(attn, V)
        out  = out.transpose(1, 2).contiguous().view(B, N, C)
        return self.out(out).transpose(1, 2).view(B, C, H, W) + x


class ContextInfoModule(nn.Module):
    def __init__(self, in_ch, out_ch, num_heads=4):
        super().__init__()
        bch        = out_ch // 4
        self.d1    = DilatedConvBranch(in_ch, bch, 1)
        self.d2    = DilatedConvBranch(in_ch, bch, 2)
        self.d3    = DilatedConvBranch(in_ch, bch, 3)
        self.d4    = DilatedConvBranch(in_ch, bch, 4)
        self.ln    = nn.GroupNorm(1, out_ch)
        self.skip  = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                      if in_ch != out_ch else nn.Identity())
        self.msa   = MaskedSelfAttention(out_ch, num_heads)
        self.final = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=False),
        )

    def forward(self, F):
        E = self.ln(
            torch.cat([self.d1(F), self.d2(F), self.d3(F), self.d4(F)], 1)
        ) + self.skip(F)
        return self.final(self.msa(E))


# ── CategoryFeatureExtractor (cfm2, cfm3, cfm4) ──────────────────────────────

class CategoryFeatureExtractor(nn.Module):
    def __init__(self, in_ch, out_ch, num_classes):
        super().__init__()
        self.C         = num_classes
        self.mask_conv = nn.Conv2d(in_ch, num_classes, 1, bias=False)
        self.feat_conv = nn.Conv2d(in_ch, out_ch,      1, bias=False)

    def forward(self, F):
        B, _, H, W = F.shape
        N  = H * W
        # ✓ Fixed: torch.softmax instead of F.softmax
        M  = torch.softmax(self.mask_conv(F).view(B, self.C, N), dim=2)
        Fv = self.feat_conv(F).view(B, -1, N)
        return torch.bmm(Fv, M.transpose(1, 2))


# ── CategoryFusionModule (fusion) ─────────────────────────────────────────────

class CategoryFusionModule(nn.Module):
    def __init__(self, d2, d3, d4, num_classes):
        super().__init__()
        self.scale  = math.sqrt(d4)
        self.align2 = nn.Linear(d2, d4, bias=False)
        self.align3 = nn.Linear(d3, d4, bias=False)
        self.W_Q    = nn.Linear(d4, d4, bias=False)
        self.W_K    = nn.Linear(d4, d4, bias=False)
        self.W_V    = nn.Linear(d4, d4, bias=False)

    def forward(self, V2, V3, V4):
        V2t  = self.align2(V2.transpose(1, 2))
        V3t  = self.align3(V3.transpose(1, 2))
        V4t  = V4.transpose(1, 2)
        Q    = self.W_Q(V2t)
        K    = self.W_K(V3t)
        Vv   = self.W_V(V3t)
        # ✓ Fixed: torch.softmax instead of F.softmax
        attn = torch.softmax(
            torch.bmm(Q, K.transpose(1, 2)) / self.scale, dim=-1)
        return (V4t + torch.bmm(attn, Vv)).transpose(1, 2)


# ── LabelSemanticMining (lsm) ─────────────────────────────────────────────────

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, H, A_hat):
        return F.leaky_relu(self.W(torch.mm(A_hat, H)), 0.2)


class LabelSemanticMining(nn.Module):
    def __init__(self, num_classes, embed_dim, hidden_dim, out_dim):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)
        nn.init.xavier_uniform_(self.embed.weight)
        self.gcn1  = GCNLayer(embed_dim,  hidden_dim)
        self.gcn2  = GCNLayer(hidden_dim, out_dim)
        self.register_buffer('A_hat', torch.eye(num_classes))

    def set_adjacency(self, A_np):
        import numpy as np
        A       = torch.from_numpy(np.array(A_np, dtype='float32'))
        A_tilde = A + torch.eye(A.size(0))
        D       = A_tilde.sum(1)
        D_inv   = torch.diag(D.pow(-0.5))
        self.A_hat = D_inv @ A_tilde @ D_inv

    def forward(self):
        ids = torch.arange(self.embed.num_embeddings,
                           device=self.embed.weight.device)
        H   = self.embed(ids)
        A   = self.A_hat.to(H.device)
        H   = self.gcn1(H, A)
        H   = self.gcn2(H, A)
        return H


# ── DualGraphNetwork (dgn) ────────────────────────────────────────────────────

class DualGraphNetwork(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.static1   = nn.Linear(feat_dim, feat_dim, bias=False)
        self.static2   = nn.Linear(feat_dim, feat_dim, bias=False)
        self.pool_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.pool_bn   = nn.BatchNorm1d(feat_dim)
        self.adj_proj  = nn.Linear(feat_dim * 2, 1,    bias=False)
        self.dyn1      = nn.Linear(feat_dim, feat_dim, bias=False)
        self.dyn2      = nn.Linear(feat_dim, feat_dim, bias=False)

    def forward(self, V, H2):
        B, D, C   = V.shape
        H2t       = H2.t().unsqueeze(0).expand(B, -1, -1)
        Y         = V * H2t
        Yt        = Y.transpose(1, 2)
        T         = F.leaky_relu(self.static1(Yt), 0.2)
        T         = F.leaky_relu(self.static2(T),  0.2)
        T         = T.transpose(1, 2)
        TY        = (T + Y).transpose(1, 2)
        delta     = TY.mean(dim=1)
        delta     = F.leaky_relu(self.pool_bn(self.pool_proj(delta)), 0.2)
        delta_exp = delta.unsqueeze(1).expand(B, C, D)
        pair      = torch.cat([delta_exp, TY], dim=-1)
        Ad        = torch.sigmoid(self.adj_proj(pair))
        Ad        = Ad * Ad.transpose(1, 2)
        Z         = torch.bmm(Ad, TY)
        Z         = F.leaky_relu(self.dyn1(Z), 0.2)
        Z         = F.leaky_relu(self.dyn2(Z), 0.2)
        return Z.transpose(1, 2)


# ── Full model ────────────────────────────────────────────────────────────────

class MLCGCN_DN169(nn.Module):
    def __init__(self, num_classes, embed_dim=300,
                 feat_dim=1664, pretrained=True,
                 adj_matrix=None, t=0.4):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim    = feat_dim

        _d2, _d3, _d4 = 128, 256, 1664

        self.backbone = DenseNet169Backbone(pretrained=pretrained)
        self.cp2 = ContextInfoModule(_d2, _d2, num_heads=4)
        self.cp3 = ContextInfoModule(_d3, _d3, num_heads=4)
        self.cp4 = ContextInfoModule(_d4, _d4, num_heads=8)

        self.cfm2   = CategoryFeatureExtractor(_d2, _d2, num_classes)
        self.cfm3   = CategoryFeatureExtractor(_d3, _d3, num_classes)
        self.cfm4   = CategoryFeatureExtractor(_d4, _d4, num_classes)
        self.fusion = CategoryFusionModule(_d2, _d3, _d4, num_classes)

        self.lsm = LabelSemanticMining(num_classes, embed_dim,
                                       1024, feat_dim)
        if adj_matrix is not None:
            self.lsm.set_adjacency(adj_matrix)

        self.dgn   = DualGraphNetwork(feat_dim, num_classes)
        self.cls_r = nn.Conv1d(feat_dim, 1, kernel_size=1, bias=True)
        self.cls_m = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        F2, F3, F4 = self.backbone(x)
        F2p = self.cp2(F2)
        F3p = self.cp3(F3)
        F4p = self.cp4(F4)
        V2  = self.cfm2(F2p)
        V3  = self.cfm3(F3p)
        V4  = self.cfm4(F4p)
        VM  = self.fusion(V2, V3, V4)
        H2  = self.lsm()
        Z   = self.dgn(VM, H2)
        Sr  = self.cls_r(Z).squeeze(1)
        Sm  = self.cls_m(F4p)
        return 0.5 * (Sr + Sm)