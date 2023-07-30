import math
import fvcore
import logging
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.logger import log_first_n

from adet.modeling.psr.attention import (NestedTensor, PositionEmbeddingSine, DPT, DPTLayer)
from .utils import (imrescale, center_of_mass, point_nms)
from .loss import dice_loss

__all__ = ["PSR"]


@META_ARCH_REGISTRY.register()
class PSR(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # get the device of the model
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.scale_ranges = cfg.MODEL.PSR.FPN_SCALE_RANGES
        self.strides = cfg.MODEL.PSR.FPN_PAR_STRIDES
        self.sigma = cfg.MODEL.PSR.SIGMA

        self.output_dir = cfg.OUTPUT_DIR
        # Partition parameters.
        self.num_ranks = cfg.MODEL.PSR.NUM_RANKS
        self.num_kernels = cfg.MODEL.PSR.NUM_KERNELS
        self.num_grids = cfg.MODEL.PSR.NUM_GRIDS

        self.par_in_features = cfg.MODEL.PSR.PAR_IN_FEATURES
        self.par_strides = cfg.MODEL.PSR.FPN_PAR_STRIDES
        self.par_in_channels = cfg.MODEL.PSR.PAR_IN_CHANNELS
        self.par_channels = cfg.MODEL.PSR.PAR_CHANNELS

        # Mask parameters.
        self.mask_in_features = cfg.MODEL.PSR.MASK_IN_FEATURES
        self.mask_in_channels = cfg.MODEL.PSR.MASK_IN_CHANNELS
        self.mask_channels = cfg.MODEL.PSR.MASK_CHANNELS
        self.num_masks = cfg.MODEL.PSR.NUM_MASKS

        # Inference parameters.
        self.score_threshold = cfg.MODEL.PSR.SCORE_THR
        self.mask_threshold = cfg.MODEL.PSR.MASK_THR

        # build the backbone.
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()

        # build the partition head.
        partition_shapes = [backbone_shape[f] for f in self.par_in_features]
        self.partition_head = PSRPartitionHead(cfg, partition_shapes)

        # build the mask head.
        mask_shapes = [backbone_shape[f] for f in self.mask_in_features]
        self.mask_head = PSRMaskHead(cfg, mask_shapes)

        # loss
        self.dice_loss_weight = cfg.MODEL.PSR.LOSS.DICE_WEIGHT
        self.partition_loss_alpha = cfg.MODEL.PSR.LOSS.FOCAL_ALPHA
        self.partition_loss_gamma = cfg.MODEL.PSR.LOSS.FOCAL_GAMMA
        self.partition_loss_weight = cfg.MODEL.PSR.LOSS.PARTITION_WEIGHT

        # image transform
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DetectionTransform` .
                Each item in the list contains the inputs for one image.
            For now, each item in the list is a dict that contains:
                image: Tensor, image in (C, H, W) format.
                instances: Instances
                Other information that's included in the original dicts, such as:
                    "height", "width" (int): the output resolution of the model, used in inference.
                        See :meth:`postprocess` for details.
         Returns:
            losses (dict[str: Tensor]): mapping from a named loss to a tensor
                storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        fpn_features = self.backbone(images.tensor)

        # partition branch
        par_features = [fpn_features[f] for f in self.par_in_features]
        par_features = self.split_feats(par_features)
        par_pred, kernel_pred = self.partition_head(par_features)

        # mask branch
        mask_features = [fpn_features[f] for f in self.mask_in_features]
        mask_pred = self.mask_head(mask_features)

        if self.training:
            mask_feat_size = mask_pred.size()[-2:]
            targets = self.get_ground_truth(gt_instances, mask_feat_size)
            losses = self.loss(par_pred, kernel_pred, mask_pred, targets)
            return losses
        else:
            par_pred = [point_nms(par_p.sigmoid(), kernel=2).permute(0, 2, 3, 1)
                            for par_p in par_pred]
            results = self.inference(par_pred, kernel_pred, mask_pred, images.image_sizes, batched_inputs)
            return results


    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @torch.no_grad()
    def get_ground_truth(self, gt_instances, mask_feat_size=None):
        ins_label_list, rank_label_list, ins_ind_label_list, grid_order_list, gt_mask_list, gt_rank_list = [], [], [], [], [], []
        for img_idx in range(len(gt_instances)):
            cur_ins_label_list, cur_rank_label_list, \
            cur_ins_ind_label_list, cur_grid_order_list, gt_mask, gt_rank = \
                self.get_ground_truth_single(img_idx, gt_instances,
                                             mask_feat_size=mask_feat_size)
            ins_label_list.append(cur_ins_label_list)
            rank_label_list.append(cur_rank_label_list)
            ins_ind_label_list.append(cur_ins_ind_label_list)
            grid_order_list.append(cur_grid_order_list)
            gt_mask_list.append(gt_mask)
            gt_rank_list.append(gt_rank)
        return ins_label_list, rank_label_list, ins_ind_label_list, grid_order_list, gt_mask_list, gt_rank_list
        
    def get_ground_truth_single(self, img_idx, gt_instances, mask_feat_size):
        gt_bboxes_raw = gt_instances[img_idx].gt_boxes.tensor
        gt_masks_raw = gt_instances[img_idx].gt_masks.tensor
        gt_ranks_raw = gt_instances[img_idx].gt_ranks
        device = gt_ranks_raw[0].device

        gt_mask_list = (gt_masks_raw * 1).permute(1, 2, 0).to(dtype=torch.uint8).cpu().numpy()
        gt_mask_list = imrescale(gt_mask_list, scale=1./4)
        gt_mask_list = torch.from_numpy(gt_mask_list).to(dtype=torch.float, device=device).permute(2, 0, 1)

        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        ins_label_list = []
        rank_label_list = []
        ins_ind_label_list = []
        grid_order_list = []

        
        for (lower_bound, upper_bound), stride, num_grid \
                in zip(self.scale_ranges, self.strides, self.num_grids):

            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            num_ins = len(hit_indices)

            ins_label = []
            grid_order = []
            rank_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
            rank_label = torch.fill_(rank_label, -1)
            ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)

            if num_ins == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                ins_label_list.append(ins_label)
                rank_label_list.append(rank_label)
                ins_ind_label_list.append(ins_ind_label)
                grid_order_list.append([])
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_ranks = gt_ranks_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices, ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            # mask center
            center_ws, center_hs = center_of_mass(gt_masks)
            valid_mask_flags = gt_masks.sum(dim=-1).sum(dim=-1) > 0

            output_stride = 4
            gt_masks = gt_masks.permute(1, 2, 0).to(dtype=torch.uint8).cpu().numpy()
            gt_masks = imrescale(gt_masks, scale=1./output_stride)
            if len(gt_masks.shape) == 2:
                gt_masks = gt_masks[..., None]
            gt_masks = torch.from_numpy(gt_masks).to(dtype=torch.uint8, device=device).permute(2, 0, 1)

            for seg_mask, gt_rank, half_h, half_w, center_h, center_w, valid_mask_flag in zip(gt_masks, gt_ranks, half_hs, half_ws, center_hs, center_ws, valid_mask_flags):
                if not valid_mask_flag:
                    continue
                upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

                top = max(top_box, coord_h-1)
                down = min(down_box, coord_h+1)
                left = max(coord_w-1, left_box)
                right = min(right_box, coord_w+1)

                rank_label[top:(down+1), left:(right+1)] = gt_rank
                for i in range(top, down+1):
                    for j in range(left, right+1):
                        label = int(i * num_grid + j)

                        cur_ins_label = torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8,
                                                    device=device)
                        cur_ins_label[:seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        ins_label.append(cur_ins_label)
                        ins_ind_label[label] = True
                        grid_order.append(label)
            if len(ins_label) == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
            else:
                ins_label = torch.stack(ins_label, 0)
            ins_label_list.append(ins_label)         # Mask GT of levels
            rank_label_list.append(rank_label)       # Rank GT of levels
            ins_ind_label_list.append(ins_ind_label) # Whether the grids in the mask GT at different levels match
            grid_order_list.append(grid_order)       # The grid indices that match in the mask GT at different levels
        return ins_label_list, rank_label_list, ins_ind_label_list, grid_order_list, gt_mask_list, gt_ranks_raw

    def loss(self, rank_preds, kernel_preds, ins_pred, targets):
        ins_label_list, rank_label_list, ins_ind_label_list, grid_order_list, gt_mask_list, gt_rank_list= targets
        ins_labels = [torch.cat([ins_labels_level_img
                                 for ins_labels_level_img in ins_labels_level], 0)
                      for ins_labels_level in zip(*ins_label_list)]

        kernel_preds = [[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:, grid_orders_level_img]
                         for kernel_preds_level_img, grid_orders_level_img in
                         zip(kernel_preds_level, grid_orders_level)]
                        for kernel_preds_level, grid_orders_level in zip(kernel_preds, zip(*grid_order_list))]
        
        # generate masks
        ins_pred_list = []
        for idy, b_kernel_pred in enumerate(kernel_preds):
            b_mask_pred = []
            for idx, kernel_pred in enumerate(b_kernel_pred):

                if kernel_pred.size()[-1] == 0:
                    cur_ins_pred = torch.Tensor([]).to(kernel_pred.device)
                    b_mask_pred.append(cur_ins_pred)
                    continue
                cur_ins_pred = ins_pred[idx, ...]
                H, W = cur_ins_pred.shape[-2:]
                N, I = kernel_pred.shape
                cur_ins_pred = cur_ins_pred.unsqueeze(0)
                kernel_pred = kernel_pred.permute(1, 0).view(I, -1, 1, 1)
                cur_ins_pred = F.conv2d(cur_ins_pred, kernel_pred, stride=1).view(-1, H, W) # (I, H, W)
                b_mask_pred.append(cur_ins_pred)

            if len(b_mask_pred) == 0:
                b_mask_pred = None
            else:
                b_mask_pred = torch.cat(b_mask_pred, 0)
            ins_pred_list.append(b_mask_pred)

        ins_ind_labels = [
            torch.cat([ins_ind_labels_level_img.flatten()
                       for ins_ind_labels_level_img in ins_ind_labels_level])
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        num_ins = flatten_ins_ind_labels.sum()

        # dice loss
        loss_dice = []
        for input, target in zip(ins_pred_list, ins_labels):
            if input is None:
                continue
            input = torch.sigmoid(input)
            loss_dice.append(dice_loss(input, target))

        loss_dice_mean = torch.cat(loss_dice).mean()
        loss_dice = loss_dice_mean * self.dice_loss_weight

        # saliency
        rank_labels = [
            torch.cat([rank_labels_level_img.flatten()
                       for rank_labels_level_img in rank_labels_level])
            for rank_labels_level in zip(*rank_label_list)
        ]
        flatten_rank_labels = torch.cat(rank_labels)

        rank_preds = [
            rank_pred.permute(0, 2, 3, 1).reshape(-1, self.num_ranks)
            for rank_pred in rank_preds
        ]
        flatten_rank_preds = torch.cat(rank_preds)

        # prepare one_hot
        pos_inds = torch.nonzero(flatten_rank_labels != -1).squeeze(1)
        flatten_rank_labels_oh = torch.zeros_like(flatten_rank_preds)
        flatten_rank_labels_inds = flatten_rank_labels[pos_inds]
        mark = True
        while mark:
            flatten_rank_labels_oh[pos_inds, flatten_rank_labels_inds] = 1
            mark = (flatten_rank_labels_inds.sum() > 0)
            flatten_rank_labels_inds = flatten_rank_labels_inds - 1
            flatten_rank_labels_inds = torch.where(flatten_rank_labels_inds  > 0, flatten_rank_labels_inds, 0)
        loss_partition = self.partition_loss_weight * fvcore.nn.sigmoid_focal_loss_jit(flatten_rank_preds, flatten_rank_labels_oh,
                        gamma=self.partition_loss_gamma,
                        alpha=self.partition_loss_alpha,
                        reduction="sum") / (num_ins + 1)

        return {'loss_mask': loss_dice,
                'loss_partition': loss_partition}


    @staticmethod
    def split_feats(feats):
        return (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear'),
                feats[1],
                feats[2],
                feats[3],
                F.interpolate(feats[4], size=feats[3].shape[-2:], mode='bilinear'))


    def inference(self, pred_pars, pred_kernels, pred_masks, cur_sizes, images):
        assert len(pred_pars) == len(pred_kernels)

        results = []
        num_ins_levels = len(pred_pars)
        for img_idx in range(len(images)):
            # image size.
            ori_img = images[img_idx]
            height, width = ori_img["height"], ori_img["width"]
            ori_size = (height, width)

            # prediction.
            pred_par = [pred_pars[i][img_idx].view(-1, self.num_ranks).detach()
                          for i in range(num_ins_levels)]
            pred_kernel = [pred_kernels[i][img_idx].permute(1, 2, 0).view(-1, self.num_kernels).detach()
                            for i in range(num_ins_levels)]
            pred_mask = pred_masks[img_idx, ...].unsqueeze(0)

            pred_par = torch.cat(pred_par, dim=0)
            pred_kernel = torch.cat(pred_kernel, dim=0)

            # inference for single image.
            result = self.inference_single_image(pred_par, pred_kernel, pred_mask,
                                    cur_sizes[img_idx], ori_size)
                        
            results.append({"instances": result})
        return results
    
    def inference_single_image(
            self, par_preds, kernel_preds, seg_preds, cur_size, ori_size
    ):
        """
        Inference for single image.
        Input:
            par_preds: Tensor, (num_grids, num_ranks)
            kernel_preds: Tensor, (num_grids, num_kernels)
            seg_preds: Tensor, (1, num_kernels, H, W)
            cur_size: tuple, (H, W)
            ori_size: tuple, (H, W)
        Output:
            results: Instances(
                pred_masks: Tensor, (H, W)
                pred_ranks: Tensor, (H, W)
                pred_boxes: Boxes, (num_instances, 4)
                scores: Tensor, (num_instances)
            )
        """
        h, w = cur_size
        f_h, f_w = seg_preds.size()[-2:]
        ratio = math.ceil(h/f_h)
        upsampled_size_out = (int(f_h*ratio), int(f_w*ratio))

        # Filter out instances that do not conform to the partitioning paradigm rules
        idx = (par_preds > self.score_threshold).int()
        indicator = idx[:,0]
        for i in range(idx.shape[1]-1):
            sub_num = idx[:, i] - idx[:, i+1]
            change = torch.where(sub_num < 0, 0, 1)
            indicator *= change
        keep = torch.where(indicator==1, True, False)
        inds = keep.nonzero()
        if len(par_preds)==0 or len(inds)==0:
            results = Instances(ori_size)
            results.scores = torch.tensor([])
            results.pred_ranks = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            return results
        
        kernel_preds = kernel_preds[inds[:, 0]]
        par_preds = par_preds[inds[:, 0]]

        size_trans = inds[:, 0].new_tensor(self.num_grids).pow(2).cumsum(0)
        strides = kernel_preds.new_ones(size_trans[-1])

        n_stage = len(self.num_grids)
        strides[:size_trans[0]] *= self.par_strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_ - 1]:size_trans[ind_]] *= self.par_strides[ind_]
        strides = strides[inds[:, 0]]            

        # mask encoding.
        N, I = kernel_preds.shape
        kernel_preds = kernel_preds.view(N, I, 1, 1)
        seg_preds = F.conv2d(seg_preds, kernel_preds, stride=1).squeeze(0).sigmoid()
        
        # mask filter
        seg_masks = seg_preds > self.mask_threshold
        sum_masks = seg_masks.sum((1, 2)).float()
        keep = sum_masks > strides
        if keep.sum() == 0:
            results = Instances(ori_size)
            results.scores = torch.tensor([])
            results.pred_ranks = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            return results            
        
        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        par_preds = par_preds[keep, ...]
        sum_masks = sum_masks[keep]          
        
        # maskness.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        par_preds *= seg_scores.unsqueeze(1)            
        
        keep = seg_masks.new_ones(par_preds.shape[0])
        seg_masks = seg_masks.float()
        rank_scores = seg_masks.new_zeros(par_preds.shape[0])
        rank_labels = -seg_masks.new_ones(par_preds.shape[0])
        for i in range(par_preds.shape[1], 0, -1):
            ind_i = (par_preds[:, i-1] * keep.float()).argmax()
            rank_scores[ind_i] = par_preds[ind_i, i-1]
            if  rank_scores[ind_i] < 0.3:
                continue
            mask_i = seg_masks[ind_i]
            keep[ind_i] = False
            rank_labels[ind_i] = i-1
            # Mask IoU filtering
            for j in range(par_preds.shape[0]):
                if ~keep[j]:
                    continue
                mask_j = seg_masks[j]
                # overlaps
                inter = (mask_i * mask_j).sum()
                union = sum_masks[ind_i] + sum_masks[j] - inter
                if union > 0:
                    if inter / union > self.mask_threshold:
                        keep[j] = False
                else:
                    keep[j] = False
        keep = rank_labels >= 0
        if keep.sum() == 0:
            results = Instances(ori_size)
            results.scores = torch.tensor([])
            results.pred_ranks = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            return results
            
        seg_preds = seg_preds[keep, :, :]
        par_preds = par_preds[keep, :]
        rank_scores = rank_scores[keep]
        rank_labels = rank_labels[keep]           
        rank_labels = self.num_ranks - 1 - rank_labels.sort(descending=True)[1].sort()[1]

        # reshape to original size.
        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                  size=upsampled_size_out,
                                  mode='bilinear')[:, :, :h, :w]
        seg_masks = F.interpolate(seg_preds,
                                  size=ori_size,
                                  mode='bilinear').squeeze(0)
        seg_masks = seg_masks > self.mask_threshold

        results = Instances(ori_size)
        results.pred_ranks = rank_labels
        results.scores = rank_scores
        results.pred_masks = seg_masks
        pred_boxes = torch.zeros(seg_masks.size(0), 4)
        results.pred_boxes = Boxes(pred_boxes)

        return results

class PSRPartitionHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        self.num_ranks = cfg.MODEL.PSR.NUM_RANKS
        self.num_kernels = cfg.MODEL.PSR.NUM_KERNELS
        self.num_grids = cfg.MODEL.PSR.NUM_GRIDS
        self.partition_in_features = cfg.MODEL.PSR.PAR_IN_FEATURES
        self.partition_strides = cfg.MODEL.PSR.FPN_PAR_STRIDES
        self.partition_in_channels = cfg.MODEL.PSR.PAR_IN_CHANNELS
        self.partition_channels = cfg.MODEL.PSR.PAR_CHANNELS
        # Convolutions to use in the towers
        self.type_dcn = cfg.MODEL.PSR.TYPE_DCN
        self.num_levels = len(self.partition_in_features)
        self.d_model = 256
        assert self.num_levels == len(self.partition_strides), \
            print("Strides should match the features.")
        # fmt: on

        head_configs = {"par": (cfg.MODEL.PSR.NUM_PAR_CONVS,
                                 cfg.MODEL.PSR.USE_DCN_IN_PAR,
                                 cfg.MODEL.PSR.USE_COORD_CONV),
                        "kernel": (cfg.MODEL.PSR.NUM_PAR_CONVS,
                                   cfg.MODEL.PSR.USE_DCN_IN_PAR,
                                   cfg.MODEL.PSR.USE_COORD_CONV)
                        }

        norm = None if cfg.MODEL.PSR.NORM == "none" else cfg.MODEL.PSR.NORM
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, \
            print("Each level must have the same channel!")
        in_channels = in_channels[0]
        assert in_channels == cfg.MODEL.PSR.PAR_IN_CHANNELS, \
            print("In channels should equal to tower in channels!")

        for head in head_configs:
            tower = []
            num_convs, use_dcn, use_coord = head_configs[head]
            for i in range(num_convs):
                conv_func = nn.Conv2d
                if i == 0:
                    if use_coord:
                        chn = self.partition_in_channels + 2
                    else:
                        chn = self.partition_in_channels
                else:
                    chn = self.partition_channels

                if (i == 3) & (head == 'par'):
                    tower.append(conv_func(
                        chn, self.d_model,
                        kernel_size=3, stride=1,
                        padding=1, bias=norm is None
                    ))
                    if norm == "GN":
                        tower.append(nn.GroupNorm(32, self.d_model))
                else:                    
                    tower.append(conv_func(
                            chn, self.partition_channels,
                            kernel_size=3, stride=1,
                            padding=1, bias=norm is None
                    ))
                    if norm == "GN":
                        tower.append(nn.GroupNorm(32, self.partition_channels))
                tower.append(nn.ReLU(inplace=True))

            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        self.par_pred = nn.Conv2d(
            self.d_model, self.num_ranks,
            kernel_size=3, stride=1, padding=1
        )
        self.kernel_pred = nn.Conv2d(
            self.partition_channels, self.num_kernels,
            kernel_size=3, stride=1, padding=1
        )

        for modules in [
            self.par_tower, self.kernel_tower,
            self.par_pred, self.kernel_pred
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.PSR.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.par_pred.bias, bias_value)  

        self.position_encoding = PositionEmbeddingSine(self.d_model/2, normalize=True)
        self.level_embed = nn.Parameter(torch.Tensor(self.num_levels, self.d_model))
        self.Attention_depth = cfg.MODEL.PSR.ATTENTION_DEPTH
        dpt_layer = DPTLayer(d_model=self.d_model, heads=8)
        self.DPT = DPT(dpt_layer, depth=self.Attention_depth)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor                                                                                                                                                                                               in the list correspond to different feature levels.

        Returns:
            pass
        """
        kernel_pred = []
        par_pred = []
        par_feats = []
        masks = []
        spatial_shapes = []
        pos_embeds = []
        level_lens = []

        for idx, feature in enumerate(features):
            ins_kernel_feat = feature
            # concat coord
            x_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-1], device=ins_kernel_feat.device)
            y_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-2], device=ins_kernel_feat.device)
            y, x = torch.meshgrid(y_range, x_range)
            y = y.expand([ins_kernel_feat.shape[0], 1, -1, -1])
            x = x.expand([ins_kernel_feat.shape[0], 1, -1, -1])
            coord_feat = torch.cat([x, y], 1)
            ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)

            # individual feature.
            kernel_feat = ins_kernel_feat
            seg_num_grid = self.num_grids[idx]
            kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode='bilinear')
            par_feat = kernel_feat

            # kernel
            kernel_feat = self.kernel_tower(kernel_feat)
            kernel_pred.append(self.kernel_pred(kernel_feat))

            # partition
            par_feat = self.par_tower(par_feat)
            bs, c, h, w = par_feat.shape
            par_feats.append(par_feat)

            # lens
            level_lens.append(h*w)
            
            # prepare for transformer
            mask = torch.zeros([bs, h, w], dtype=bool, device=par_feat.device)
            pos = self.position_encoding(NestedTensor(par_feat, mask))
            pos =  pos + self.level_embed[idx].view(-1, 1, 1)
            spatial_shapes.append((h, w))
            masks.append(mask)
            pos_embeds.append(pos)

        par_feats = self.DPT(par_feats, pos_embeds, self.num_grids)
        par_pred = [self.par_pred(par_feat) for par_feat in par_feats]

        return par_pred, kernel_pred


class PSRMaskHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        self.num_masks = cfg.MODEL.PSR.NUM_MASKS
        self.mask_in_features = cfg.MODEL.PSR.MASK_IN_FEATURES
        self.mask_in_channels = cfg.MODEL.PSR.MASK_IN_CHANNELS
        self.mask_channels = cfg.MODEL.PSR.MASK_CHANNELS
        self.num_levels = len(input_shape)
        assert self.num_levels == len(self.mask_in_features), \
            print("Input shape should match the features.")
        norm = None if cfg.MODEL.PSR.NORM == "none" else cfg.MODEL.PSR.NORM

        self.convs_all_levels = nn.ModuleList()
        for i in range(self.num_levels):
            convs_per_level = nn.Sequential()
            if i == 0:
                conv_tower = list()
                conv_tower.append(nn.Conv2d(
                    self.mask_in_channels, self.mask_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=norm is None
                ))
                if norm == "GN":
                    conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                conv_tower.append(nn.ReLU(inplace=False))
                convs_per_level.add_module('conv' + str(i), nn.Sequential(*conv_tower))
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    chn = self.mask_in_channels + 2 if i == 3 else self.mask_in_channels
                    conv_tower = list()
                    conv_tower.append(nn.Conv2d(
                        chn, self.mask_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=norm is None
                    ))
                    if norm == "GN":
                        conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                    conv_tower.append(nn.ReLU(inplace=False))
                    convs_per_level.add_module('conv' + str(j), nn.Sequential(*conv_tower))
                    upsample_tower = nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_module(
                        'upsample' + str(j), upsample_tower)
                    continue
                conv_tower = list()
                conv_tower.append(nn.Conv2d(
                    self.mask_channels, self.mask_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=norm is None
                ))
                if norm == "GN":
                    conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                conv_tower.append(nn.ReLU(inplace=False))
                convs_per_level.add_module('conv' + str(j), nn.Sequential(*conv_tower))
                upsample_tower = nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=False)
                convs_per_level.add_module('upsample' + str(j), upsample_tower)

            self.convs_all_levels.append(convs_per_level)

        self.conv_pred = nn.Sequential(
            nn.Conv2d(
                self.mask_channels, self.num_masks,
                kernel_size=1, stride=1,
                padding=0, bias=norm is None),
            nn.GroupNorm(32, self.num_masks),
            nn.ReLU(inplace=True)
        )

        for modules in [self.convs_all_levels, self.conv_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            pass
        """
        assert len(features) == self.num_levels, \
            print("The number of input features should be equal to the supposed level.")

        # bottom features first.
        feature_add_all_level = self.convs_all_levels[0](features[0])
        for i in range(1, self.num_levels):
            mask_feat = features[i]
            if i == 3:  # add for coord.
                x_range = torch.linspace(-1, 1, mask_feat.shape[-1], device=mask_feat.device)
                y_range = torch.linspace(-1, 1, mask_feat.shape[-2], device=mask_feat.device)
                y, x = torch.meshgrid(y_range, x_range)
                y = y.expand([mask_feat.shape[0], 1, -1, -1])
                x = x.expand([mask_feat.shape[0], 1, -1, -1])
                coord_feat = torch.cat([x, y], 1)
                mask_feat = torch.cat([mask_feat, coord_feat], 1)
            # add for top features.
            feature_add_all_level = feature_add_all_level + self.convs_all_levels[i](mask_feat)

        mask_pred = self.conv_pred(feature_add_all_level)

        return mask_pred
    