# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union
from einops import rearrange

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

# from ..transformer.cat_seg_predictor import CATSegPredictor
from ..new.PPG import PseudoMaskGenerator,PseudoPointGenerator


@SEM_SEG_HEADS_REGISTRY.register()
class AAASegHead(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        num_clusters=5,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            num_clusters: The number of clusters for PseudoMaskGenerator.
        """
        super().__init__()
        # self.ignore_value = ignore_value # 移除了未使用的参数
        # self.num_classes = num_classes # 移除了未使用的参数
        # self.feature_resolution = feature_resolution # 移除了未使用的参数

        self.softmax0 = nn.Softmax(dim=2)
        self.PMG = PseudoMaskGenerator(num_clusters=num_clusters)
        self.PPG = PseudoPointGenerator()
        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
        }

    def correlation(self, img_feats, text_feats):
        """
        CAT-SEG原始形状
        img_feats torch.Size([1, 768, 24, 24])
        text_feats torch.Size([1, 150, 1, 768])

        基于MAFT的形状
        text_feats shape: torch.Size([1, 171, 1, 768])

        """
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        # print("text_feats shape:", text_feats.shape)
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr


    def forward(self, features, raw_text_feature):
        """
        Arguments:
            features: (B, C, H, W) or (B, HW, C)
            raw_text_feature: (B, T, P, C)
        """
        # print("features shape:", features.shape)
        """
        VIT features shape: torch.Size([1, 577, 768])
        CONVNEXT features shape: torch.Size([1, 768, 36, 25])
        """
        # img_feat = rearrange(features[:, 1:, :], "b (h w) c->b c h w", h=self.feature_resolution[0], w=self.feature_resolution[1])
        img_feat = features
        corr = self.correlation(img_feat, raw_text_feature)
        # print("corr shape:", corr.shape) # corr shape: torch.Size([B, 1, 171, 36, 25])
<<<<<<< HEAD
        # corr_prob = self.softmax0(corr)
        """
        PPG applies a softmax operati onto the correlation map Cn v&l∈R1×H×W for each n-th class to generate class specific object probability masks
        """
        b, _, t, h, w = corr.shape
        corr_prob = F.softmax(corr.view(b, 1, t, -1), dim=-1).view(b, 1, t, h, w)
        # corr_prob = corr
        
        alpha = 1.0 / (h * w)
=======
        corr_prob = self.softmax0(corr)
        
        alpha = 0.1
>>>>>>> feabc4013b61fef153ad34f36e70956f1345eb0d
        binary_mask = (corr_prob > alpha).float()
        PseudoMask = self.PMG(binary_mask) # out shape: torch.Size([1, 171, 5, 36, 25])
        PseudoPoint = self.PPG(PseudoMask , corr_prob)
        # print("PseudoPoint shape:", PseudoPoint.shape) # PseudoPoint shape: torch.Size([1, 171, 5, 2])
        
        out = {
            "segmentation": corr,
            "PseudoMask": PseudoMask,
            "PseudoPoint": PseudoPoint
        }
        return out