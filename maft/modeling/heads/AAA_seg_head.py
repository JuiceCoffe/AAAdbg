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
        if text_feats.dim() == 2:
            text_feats = text_feats.view(1,text_feats.size(0),1, text_feats.size(1))
            text_feats = text_feats.expand(img_feats.size(0), -1, -1, -1)
        print("text_feats shape:", text_feats.shape)
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr


    def forward(self, features, text_feature,batched_inputs=None):

        
        logit_scale = 40
        alpha = 0.2
        area_thd = 0.001
        prob_thd = 0.05
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
        corr = self.correlation(img_feat, text_feature)
        b, _, t, h, w = corr.shape
        # print("corr shape:", corr.shape) # corr shape: torch.Size([B, 1, 171, 36, 25])



        pred_cls = torch.argmax(corr, dim=2) 
        pred_cls = pred_cls.squeeze(1)   # shape: torch.Size([B, h, w]) 每个位置为最大概率类别

        pred_mask = F.one_hot(pred_cls, num_classes=t)
        pred_mask = pred_mask.permute(0, 3, 1, 2).contiguous()  # shape: torch.Size([B, num_classes, h, w])
        area = torch.sum(pred_mask, dim=(2,3))  # shape: torch.Size([B, num_classes])
        valid_area_cls = (area > area_thd * h * w).float()  # shape: torch.Size([B, num_classes])
        pred_result = pred_mask * valid_area_cls.view(b, t, 1, 1)  # shape: torch.Size([B, num_classes, h, w])



        # print("pred_result shape:", pred_result.shape) # pred_result shape: torch.Size([1, 36, 25])

        # corr_prob = F.softmax(corr.view(b, 1, t, -1) * logit_scale, dim=-1).view(b, 1, t, h, w)  # 该版本是ESC-NET的做法，对类内SoftMAX而非类间

        corr_prob = F.softmax(corr * logit_scale, dim=2)  # ResCLIP的做法，对类间SoftMAX [B,1, T, H, W]
        corr_prob = corr_prob.permute(0, 3, 4, 2, 1).contiguous() # shape: torch.Size([B, H, W, T, 1])
        corr_prob = corr_prob.squeeze(-1) # shape: torch.Size([B, H, W, T])
        max_probs, _ = corr_prob.max(dim=-1) 
        valid_prob_mask = max_probs > prob_thd # shape: torch.Size([B, H, W])



        pred_result = pred_result.permute(0, 2, 3, 1).contiguous() # shape: torch.Size([B, h, w, T])

        pred_result = pred_result * valid_prob_mask.unsqueeze(-1) # shape: torch.Size([B, h, w, T])
        pred_result = torch.cat((pred_result, 1 - pred_result.sum(dim=-1, keepdim=True)), dim=-1) # shape: torch.Size([B, h, w, T+1])
        pred_result = torch.argmax(pred_result, dim=-1) # shape: torch.Size([B, h, w])
        
        
        binary_mask = (corr_prob > alpha).float()
        PseudoMask = self.PMG(binary_mask) # out shape: torch.Size([1, 171, 5, 36, 25])
        PseudoPoint = self.PPG(PseudoMask , corr_prob)
        # print("PseudoPoint shape:", PseudoPoint.shape) # PseudoPoint shape: torch.Size([1, 171, 5, 2])
        
        out = {
            "segmentation": corr,
            "PseudoMask": PseudoMask,
            "PseudoPoint": PseudoPoint,
            "pred_result": pred_result,
            # "upsampled_pred_result": upsampled_pred_result
        }
        return out