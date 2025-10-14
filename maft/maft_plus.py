"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py
"""
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .modeling.transformer_decoder.fcclip_transformer_decoder import MaskPooling, get_classification_logits

from .modeling.maft.mask_aware_loss import  MA_Loss
from .modeling.maft.representation_compensation import  Representation_Compensation
from .modeling.maft.content_dependent_transfer import ContentDependentTransfer

from .utils.text_templetes import VILD_PROMPT

import matplotlib.pyplot as plt
import os
import numpy as np


@META_ARCH_REGISTRY.register()
class MAFT_Plus(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        # backbone_t,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        train_metadata,
        test_metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        # MAFT
        rc_weights,
        cdt_params,
    ):

        super().__init__()
        self.backbone = backbone
        # self.backbone_t = backbone_t
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.train_metadata = train_metadata
        self.test_metadata = test_metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        # FC-CLIP args
        # self.mask_pooling = MaskPooling()
        self.train_text_classifier = None
        self.test_text_classifier = None
        self.void_embedding = nn.Embedding(1, backbone.dim_latent) # use this for void

        _, self.train_num_templates, self.train_class_names = self.prepare_class_names_from_metadata(train_metadata, train_metadata)
        _, _, self.raw_class_names = self.prepare_raw_class_names_from_metadata(train_metadata, train_metadata)
        # print(f"Test dataset has {len(self.raw_class_names)} classes:\n", self.raw_class_names)

        # self.cdt = ContentDependentTransfer(d_model = cdt_params[0], nhead = cdt_params[1], panoptic_on = panoptic_on)
        # self.ma_loss = MA_Loss()  # BCELoss BCEWithLogitsLoss SmoothL1Loss
        # self.rc_loss = Representation_Compensation()
        self.rc_weights = rc_weights

        self._freeze()
        self.train_dataname = None
        self.test_dataname = None

        self.cache = None # for caching RAW text embeds in inference

    def _freeze(self, ):
        for name, param in self.named_parameters():
            if 'backbone_t' in name:
                param.requires_grad = False

            elif 'backbone' in name:
                if 'clip_model.visual.trunk.stem' in name:
                    param.requires_grad = True
                if 'clip_model.visual.trunk.stages' in name:
                    param.requires_grad = True
                if 'clip_model.visual.trunk.norm_pre' in name:
                    param.requires_grad = True

                if 'clip_model.visual.trunk.head.norm.' in name:
                    param.requires_grad = False
                if 'clip_model.visual.head.mlp.' in name:
                    param.requires_grad = False

        for name, param in self.named_parameters():
            if param.requires_grad == True and 'sem_seg_head' not in name:
                print(name, param.requires_grad)

    def prepare_class_names_from_metadata(self, metadata, train_metadata):
        def split_labels(x):
            res = []
            for x_ in x:
                x_ = x_.replace(', ', ',')
                x_ = x_.split(',') # there can be multiple synonyms for single class
                res.append(x_)
            return res
        # get text classifier
        try:
            class_names = split_labels(metadata.stuff_classes) # it includes both thing and stuff
            train_class_names = split_labels(train_metadata.stuff_classes)
        except:
            # this could be for insseg, where only thing_classes are available
            class_names = split_labels(metadata.thing_classes)
            train_class_names = split_labels(train_metadata.thing_classes)
        train_class_names = {l for label in train_class_names for l in label}
        category_overlapping_list = []
        self.vis_class_names = class_names
        for test_class_names in class_names:
            is_overlapping = not set(train_class_names).isdisjoint(set(test_class_names)) 
            category_overlapping_list.append(is_overlapping)
        category_overlapping_mask = torch.tensor(
            category_overlapping_list, dtype=torch.long)
        
        def fill_all_templates_ensemble(x_=''):
            res = []
            for x in x_:
                for template in VILD_PROMPT:
                    res.append(template.format(x))
            return res, len(res) // len(VILD_PROMPT)
       
        num_templates = []
        templated_class_names = []
        # print('class_names: ',len(class_names)) 171
        for x in class_names:
            templated_classes, templated_classes_num = fill_all_templates_ensemble(x)
            templated_class_names += templated_classes
            num_templates.append(templated_classes_num) # how many templates for current classes
        class_names = templated_class_names        
        return category_overlapping_mask, num_templates, class_names

    def prepare_raw_class_names_from_metadata(self, metadata, train_metadata):
        def split_labels(x):
            res = []
            for x_ in x:
                x_ = x_.replace(', ', ',')
                x_ = x_.split(',') # there can be multiple synonyms for single class
                res.append(x_)
            return res
        # get text classifier
        try:
            class_names = split_labels(metadata.stuff_classes) # it includes both thing and stuff
            train_class_names = split_labels(train_metadata.stuff_classes)
        except:
            # this could be for insseg, where only thing_classes are available
            class_names = split_labels(metadata.thing_classes)
            train_class_names = split_labels(train_metadata.thing_classes)
        train_class_names = {l for label in train_class_names for l in label}
        category_overlapping_list = []
        for test_class_names in class_names:
            is_overlapping = not set(train_class_names).isdisjoint(set(test_class_names)) 
            category_overlapping_list.append(is_overlapping)
        category_overlapping_mask = torch.tensor(
            category_overlapping_list, dtype=torch.long)
        
        # --- 修改开始 ---
        # 1. 删除了内部函数 fill_all_templates_ensemble 和相关的 VILD_PROMPT 模板。
        #    因为我们不再需要用模板来扩展类别名称。

        # 2. 删除了遍历 class_names 并调用模板函数的循环。

        # 3. 直接处理 `class_names` 变量以提取原始类别名。
        #    此时的 `class_names` 是一个列表的列表，例如: [['wall'], ['building'], ['chest of drawers']]
        #    我们通过列表推导式提取每个子列表的第一个元素，得到一个扁平的字符串列表。
        original_class_names = [x[0] for x in class_names]
        # print(f"Original class names: {len(original_class_names)}:\n", original_class_names)
        """
        Original class names: 171:
        ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'branch', 'bridge', 'building', 'bush', 'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain', 'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble', 'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow', 'plant-other', 'plastic', 'platform', 'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other', 'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops', 'window-blind', 'window-other', 'wood']
        """
       
        # 4. 由于我们不再使用模板，第二个返回参数（原为 num_templates）需要调整。
        #    为了保持函数返回三个值的结构，我们创建一个占位符。
        #    这个列表记录每个类别我们只使用了1个名称（即原始名称）。
        num_names_per_category = [1] * len(original_class_names)
        # --- 修改结束 ---

        # 函数现在返回：
        # 1. 类别是否重叠的掩码 (不变)
        # 2. 每个类别使用的名称数量 (现在恒为1)
        # 3. 干净的、未经模板化的原始类别名列表
        return category_overlapping_mask, num_names_per_category, original_class_names

    def set_metadata(self, metadata):
        self.test_metadata = metadata
        self.category_overlapping_mask, self.test_num_templates, self.test_class_names = self.prepare_class_names_from_metadata(metadata, self.train_metadata)
        self.test_text_classifier = None
        return

    def get_text_classifier(self, dataname):
        if self.training:
            if self.train_dataname != dataname:
                text_classifier = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                # print("train_class_names len: ",len(self.train_class_names)) 4592
                # print("train_class_names: ",self.train_class_names) 带模板的类别名
                # exit()
                for idx in range(0, len(self.train_class_names), bs):
                    text_classifier.append(self.backbone.get_text_classifier(self.train_class_names[idx:idx+bs], self.device).detach())
                text_classifier = torch.cat(text_classifier, dim=0)

                # average across templates and normalization.
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-1]).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.train_text_classifier = text_classifier
                self.train_dataname = dataname
                self.class_name_of_classifier = [element for index, element in enumerate(self.train_class_names) if index % len(VILD_PROMPT) == 0]
                # print('train_class_names: ',len(self.class_name_of_classifier))
                # exit()
            return self.train_text_classifier, self.train_num_templates
        else:
            if self.test_dataname != dataname:
                self.category_overlapping_mask, self.test_num_templates, self.test_class_names = self.prepare_class_names_from_metadata(self.test_metadata[dataname], self.train_metadata)
                text_classifier = []
                bs = 128
                for idx in range(0, len(self.test_class_names), bs):
                    text_classifier.append(self.backbone.get_text_classifier(self.test_class_names[idx:idx+bs], self.device).detach())
                text_classifier = torch.cat(text_classifier, dim=0)

                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-1]).mean(1) 
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.test_text_classifier = text_classifier
                self.test_dataname = dataname
                self.class_name_of_classifier = [element for index, element in enumerate(self.train_class_names) if index % len(VILD_PROMPT) == 0]
            return self.test_text_classifier, self.test_num_templates

    def get_text_embeds(self, dataname): # 来自CAT-SEG，用于计算代价体
        if self.cache is not None and not self.training:
            return self.cache

        # print("classnames: ", classnames)
        # print("templates: ", templates)
        # print("prompt: ", prompt)
        """
        CAT-SEG原始打印结果
        classnames:  ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag']
        templates:  ['A photo of a {} in the scene']
        prompt:  None
        """
        prompt = None
        templates = ['A photo of a {} in the scene']
        tokens = []

        if self.training:
            # 在训练期间，使用训练元数据
            metadata = self.train_metadata
        else:
            # 在评估期间，使用特定的测试元数据
            metadata = self.test_metadata[dataname]

        _,_,classnames = self.prepare_raw_class_names_from_metadata(metadata, self.train_metadata)
        
        self.raw_class_names = classnames

        for classname in classnames:
            if ', ' in classname:
                classname_splits = classname.split(', ')
                texts = [template.format(classname_splits[0]) for template in templates]
            else:
                texts = [template.format(classname) for template in templates]  # format with class
            texts =  self.backbone.tokenize_text(texts).cuda()
            tokens.append(texts)
        tokens = torch.stack(tokens, dim=0).squeeze(1)
       
        self.tokens = tokens


        # class_embeddings = clip_model.encode_text(tokens, prompt)
        class_embeddings =  self.backbone.encode_text(tokens)
        # print("class_embeddings shape:", class_embeddings.shape)
        '''
        class_embeddings shape: torch.Size([171, 768])
        '''
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        
        
        class_embeddings = class_embeddings.unsqueeze(1)
        class_embeddings = class_embeddings.unsqueeze(0)
        
        if not self.training:
            self.cache = class_embeddings
        # print("class_embeddings:", class_embeddings.shape) # class_embeddings: torch.Size([171, 1, 768])

        return class_embeddings

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg) # cfg.MODEL.BACKBONE.NAME : CLIP
        # backbone_t = build_backbone(cfg)


        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        test_metadata = {i: MetadataCatalog.get(i) for i in cfg.DATASETS.TEST}
        return {
            "backbone": backbone,
            # "backbone_t":backbone_t,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "train_metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "test_metadata": test_metadata, # MetadataCatalog.get(cfg.DATASETS.TEST[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "rc_weights": cfg.MODEL.rc_weights,
            "cdt_params": cfg.MODEL.cdt_params,
        }

    @property
    def device(self):
        return self.pixel_mean.device


    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]

        original_image = images[0].clone() # for visualization only

        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        file_names = [x["file_name"] for x in batched_inputs] # 可去变量
        file_names = [x.split('/')[-1].split('.')[0] for x in file_names] # 可去变量

        meta = batched_inputs[0]["meta"]
        # text_classifier, num_templates = self.get_text_classifier('openvocab_ade20k_panoptic_val')
        text_classifier, num_templates = self.get_text_classifier(meta['dataname'])
        # print("meta['dataname']:",meta['dataname'])
        text_classifier = torch.cat([text_classifier, F.normalize(self.void_embedding.weight, dim=-1)], dim=0)
        # print("text_classifier:", text_classifier.shape) # text_classifier: torch.Size([329, 768]) 328个类别+1个void
       

        features = self.backbone.extract_features(images.tensor) # 多尺度特征图,不包括用于与文本匹配的（该层在self.backbone.visual_prediction_forward中调用）
    
        
        for k in features.keys():
            features[k] = features[k].detach()

        features['text_classifier'] = text_classifier
        features['num_templates'] = num_templates

        clip_feature = features['clip_vis_dense'] # torch.Size([1, 1536, 38, 25])
        img_feat = self.visual_prediction_forward_convnext_2d(clip_feature) # 输出可以在CLIP空间中直接理解的语义特征图
        
        img_feats = F.normalize(img_feat, dim=1) # B C H W
        text_feats = text_classifier# T C 带模板
        # print("text_feats shape:", text_feats.shape)
        logit_scale = torch.clamp(self.backbone.clip_model.logit_scale.exp(), max=100)
        logits = self.backbone.clip_model.logit_scale * torch.einsum('bchw, tc -> bthw', img_feats, text_feats)
        
        # ade20K: seg_logits:[1,150,336,448] --> [1,150,512,683]
        seg_logits = logits

        

        final_seg_logits = []
        cur_idx = 0
        for num_t in num_templates: 
            final_seg_logits.append(seg_logits[:, cur_idx: cur_idx + num_t,:,:].max(1).values)
            cur_idx += num_t
        final_seg_logits.append(seg_logits[:, -1,:,:]) # the last classifier is for void
        final_seg_logits = torch.stack(final_seg_logits, dim=1)
        #final_seg_logits = nn.functional.interpolate(final_seg_logits, size=original_image.shape[-2:], mode='bilinear', align_corners=False)
        seg_probs = torch.softmax(final_seg_logits, dim=1) # B T(纯类别)+1 H W

        # pred_result = torch.argmax(seg_probs, dim=1) # B H W

        def post_process(seg_probs):
        
            area_thd = 7.5 # 当前最佳 7.5 -> mIoU=7.3

            corr_prob = seg_probs[:, :-1, :, :].clone()  # B T H W 去除void
            pred_cls = corr_prob.argmax(dim=1) # B H W 最大索引为T-1
            pred_mask = F.one_hot(pred_cls, num_classes=corr_prob.size(1)) # B H W T
            area = pred_mask.sum(dim=(1, 2))  # [B, T]
            valid_area_cls = area > area_thd
            valid_area_mask = torch.einsum('bhwt, bt -> bhwt', pred_mask, valid_area_cls)

            corr_prob = corr_prob * valid_area_mask.permute(0, 3, 1, 2).contiguous() # B T H W
            corr_prob = F.softmax(corr_prob, dim=1)

            original_h, original_w = batched_inputs[0]["height"], batched_inputs[0]["width"]
            corr_prob = F.interpolate(corr_prob, size=(original_h, original_w), mode='bilinear', align_corners=False)
            
            max_prob, pred_result = corr_prob.max(dim=1) # B H W 最大索引为T-1

            return pred_result

        pred_result = post_process(seg_probs)
        

        # visualize_segmentation(pred_result, self.vis_class_names+['void'],batched_inputs[0]["image"],f"./show/{file_names[0]}_")



        mask_results = F.one_hot(pred_result, num_classes=seg_probs.shape[1]).permute(0, 3, 1, 2).float() # B T H W
        # mask_results = mask_results[0].detach() # T H W

        if self.training:
            pass
        else:
            original_h, original_w = batched_inputs[0]["height"], batched_inputs[0]["width"]
            # mask_results = F.interpolate(mask_results, size=(original_h, original_w), mode='bilinear', align_corners=False)[0,:-1]     
            # print("mask_results:", mask_results.shape) # mask_results: torch.Size([150, 512, 683]) ade20k
            mask_results = retry_if_cuda_oom(sem_seg_postprocess)(mask_results, images.image_sizes[0], original_h, original_w)
            return [{"sem_seg": mask_results}] # 去除void



    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred, dataname):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()
        num_classes = len(self.test_metadata[dataname].stuff_classes)
        keep = labels.ne(num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.test_metadata[dataname].thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, dataname):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        # if this is panoptic segmentation
        if self.panoptic_on:
            num_classes = len(self.test_metadata[dataname].stuff_classes)
        else:
            num_classes = len(self.test_metadata[dataname].thing_classes)
        labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.test_metadata[dataname].thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result


    #错误的函数
    def visual_prediction_forward_convnext(self, x):
        batch, channel, h, w = x.shape
        x = x.reshape(batch*h*w, channel).unsqueeze(-1).unsqueeze(-1) # fake 2D input
        x = self.backbone.clip_model.visual.trunk.head(x)
        x = self.backbone.clip_model.visual.head(x)
        return x.reshape(batch, h, w, x.shape[-1]).permute(0,3,1,2) 
    
    def visual_prediction_forward_convnext_2d(self, x):
        
        clip_vis_dense = self.backbone.clip_model.visual.trunk.head.norm(x)
        clip_vis_dense = self.backbone.clip_model.visual.trunk.head.drop(clip_vis_dense.permute(0, 2, 3, 1))
        clip_vis_dense = self.backbone.clip_model.visual.head(clip_vis_dense).permute(0, 3, 1, 2)
        
        return clip_vis_dense

def visualize_segmentation(pred_result, class_names,original_image_tensor, save_path="./show/",fig_size=(10, 10)):
    """
    可视化初步分割结果并将其保存到文件。
    图例会根据每个类别占有的像素数从多到少进行排序。

    Arguments:
        pred_result (torch.Tensor): 模型预测的分割结果，形状为 (H, W)，值为类别索引。
        class_names (list): 一个包含分类器所有类别实际名称的列表。
        save_path (str): 可视化结果的保存路径。
    """
    print("类别数：", len(class_names))

   
    # 确保pred_result在CPU上并且是numpy数组
    if isinstance(pred_result, torch.Tensor):
        pred_result = pred_result.cpu().numpy()

    # 检查是否是批处理的结果，如果是，则只取第一个样本
    if len(pred_result.shape) == 3 and pred_result.shape[0] == 1:
        pred_result = pred_result[0]
    
    height, width = pred_result.shape
    num_classes = len(class_names)

    # 1. 为所有可能的类别生成一个固定的随机颜色调色板
    np.random.seed(0) # 使用固定的种子以确保每次颜色一致
    palette = np.random.randint(0, 255, size=(num_classes, 3))

    # 2. 创建一个彩色的图像（与之前相同）
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    for class_index in range(num_classes):
        color_image[pred_result == class_index] = palette[class_index]

    # 3. 统计每个类别的像素数
    # np.unique 返回图像中实际出现过的类别索引和它们对应的像素数
    unique_classes, pixel_counts = np.unique(pred_result, return_counts=True)
    
    # 4. 将统计结果与类名结合，并按像素数降序排序
    class_statistics = []
    for i, class_index in enumerate(unique_classes):
        if class_index < num_classes:
            class_statistics.append({
                "index": class_index,
                "name": class_names[class_index],
                "count": pixel_counts[i]
            })
    sorted_class_statistics = sorted(class_statistics, key=lambda x: x['count'], reverse=True)

    # 创建目录
    os.makedirs(os.path.dirname(save_path + "prediction.png"), exist_ok=True)

    # 保存原图
    original_image = original_image_tensor.permute(1, 2, 0).numpy().astype(np.uint8).copy()
    plt.imsave(save_path + 'original_image.png', original_image)

    # 绘制分割结果
    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(color_image)
    ax.axis('off')

    # 图例放在底部
    legend_elements = []
    for stats in sorted_class_statistics:
        class_index = stats["index"]
        class_name = stats["name"]
        pixel_count = stats["count"]
        color = palette[class_index] / 255.0
        label = f"{class_name} ({pixel_count:,} px)"
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=color, label=label))

    # 使用 fig.legend 放置在底部
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=min(4, len(legend_elements)),  # 一行最多显示4个类别
        frameon=False
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # 给图例留空间

    # 保存
    pred_save_path = save_path + "prediction.png"
    try:
        plt.savefig(pred_save_path, bbox_inches='tight')
        print(f"可视化结果已保存至: {pred_save_path}")
    except Exception as e:
        print(f"保存文件时出错: {e}")

    plt.close(fig)

