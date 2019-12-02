from __future__ import division

import torch
import torch.nn as nn

from mmdet.core import (bbox2result, bbox2roi, build_assigner, build_sampler,
                        merge_aug_masks)
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin
from mmdet.models.utils import Scale, Scale_channel
from icecream import ic
import mmcv
@DETECTORS.register_module
class CascadeRCNN_pair(BaseDetector, RPNTestMixin, BBoxTestMixin):

    def __init__(self,
                 num_stages,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 pair_train=False,
                 normal_train=False,
                 style=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        super(CascadeRCNN_pair, self).__init__()

        self.num_stages = num_stages
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = nn.ModuleList()
            self.bbox_head = nn.ModuleList()
            if not isinstance(bbox_roi_extractor, list):
                bbox_roi_extractor = [
                    bbox_roi_extractor for _ in range(num_stages)
                ]
            if not isinstance(bbox_head, list):
                bbox_head = [bbox_head for _ in range(num_stages)]
            assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
            for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
                self.bbox_roi_extractor.append(
                    builder.build_roi_extractor(roi_extractor))
                self.bbox_head.append(builder.build_head(head))

        if mask_head is not None:
            self.mask_head = nn.ModuleList()
            if not isinstance(mask_head, list):
                mask_head = [mask_head for _ in range(num_stages)]
            assert len(mask_head) == self.num_stages
            for head in mask_head:
                self.mask_head.append(builder.build_head(head))
            if mask_roi_extractor is not None:
                self.share_roi_extractor = False
                self.mask_roi_extractor = nn.ModuleList()
                if not isinstance(mask_roi_extractor, list):
                    mask_roi_extractor = [
                        mask_roi_extractor for _ in range(num_stages)
                    ]
                assert len(mask_roi_extractor) == self.num_stages
                for roi_extractor in mask_roi_extractor:
                    self.mask_roi_extractor.append(
                        builder.build_roi_extractor(roi_extractor))
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

        self.pair_train = pair_train
        self.normal_train = normal_train
        self.style = style
        if self.style == 'sub_feat':
            self.scale_a = Scale(0.5)
            self.scale_b = Scale(0.5)
        elif self.style == 'vector_add_feat':
            self.scale_a = Scale_channel(1, 256)
            self.scale_b = Scale_channel(1, 256)
    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(CascadeRCNN_pair, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        for i in range(self.num_stages):
            if self.with_bbox:
                self.bbox_roi_extractor[i].init_weights()
                self.bbox_head[i].init_weights()
            if self.with_mask:
                if not self.share_roi_extractor:
                    self.mask_roi_extractor[i].init_weights()
                self.mask_head[i].init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_feat_pair(self, img):
        if self.pair_train is True and self.style is not None:

            b, c, h, w = img.shape
            img = img.reshape(-1, c // 2, h, w)
            if self.style == 'sub_img':
                img = img[0::2, :, :, :] - img[1::2, :, :, :]
                x = self.extract_feat(img)
            elif self.style == 'add_img':
                img = img[0::2, :, :, :] + img[1::2, :, :, :]
                x = self.extract_feat(img)
            elif self.style == 'sub_feat':
                x = self.extract_feat(img)
                x_ = []

                for i, lvl_feat in enumerate(x):
                    x_.append(self.scale_a(lvl_feat[0::2, :, :, :]) + self.scale_b(lvl_feat[1::2, :, :, :]))
                x = tuple(x_)
            elif self.style == 'add_feat':
                x = self.extract_feat(img)
                x_ = []
                for lvl_feat in x:
                    x_.append(lvl_feat[0::2, :, :, :] + lvl_feat[1::2, :, :, :])
                x = tuple(x_)
            elif self.style == 'vector_add_feat':
                x = self.extract_feat(img)
                x_ = []
                for i, lvl_feat in enumerate(x):
                    x_.append(self.scale_a(lvl_feat[0::2, :, :, :]) + self.scale_b(lvl_feat[1::2, :, :, :]))
                x = tuple(x_)
        else:
            x = self.extract_feat(img)

        return x

    def extract_feats_pair(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat_pair(img)

    def extract_feat_pair_test(self, img, temp_feat=None):
        if self.pair_train is True and self.style is not None:
            if temp_feat is None:
                b, c, h, w = img.shape
                img = img.reshape(-1, c // 2, h, w)
                if self.style == 'sub_feat' or self.style == 'vector_add_feat':
                    x = self.extract_feat(img)
                    x_ = []
                    y = []
                    for i, lvl_feat in enumerate(x):
                        x_.append(self.scale_a(lvl_feat[0::2, :, :, :]) + self.scale_b(lvl_feat[1::2, :, :, :]))
                        y.append(self.scale_b(lvl_feat[1::2, :, :, :]))
                    x = tuple(x_)
                else:
                    raise ValueError('unvalid style')
                return x, y
            else:
                if self.style == 'sub_feat' or self.style == 'vector_add_feat':
                    x = self.extract_feat(img)
                    x_ = []
                    y_ = []
                    for i, (lvl_feat, lvl_temp_feat) in enumerate(zip(x, temp_feat)):
                        x_.append(self.scale_a(lvl_feat) + lvl_temp_feat)

                    x = tuple(x_)
                else:
                    raise ValueError('unvalid style')
                return x
        else:
            x = self.extract_feat(img)
            return x

    def extract_feats_pair_test(self, imgs, temp_feats=None):
        assert isinstance(imgs, list)
        x = []
        if temp_feats is None:
            y = []
            for img in imgs:
                x.append(self.extract_feat_pair_test(img, temp_feat=None)[0])
                y.append(self.extract_feat_pair_test(img, temp_feat=None)[1])
            return x, y
        else:
            for img, temp_feat in zip(imgs, temp_feats):
                x.append(self.extract_feat_pair_test(img, temp_feat=temp_feat))
            return x

    def forward_dummy(self, img):
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).cuda()
        # bbox heads
        rois = bbox2roi([proposals])
        if self.with_bbox:
            for i in range(self.num_stages):
                bbox_feats = self.bbox_roi_extractor[i](
                    x[:self.bbox_roi_extractor[i].num_inputs], rois)
                if self.with_shared_head:
                    bbox_feats = self.shared_head(bbox_feats)
                cls_score, bbox_pred = self.bbox_head[i](bbox_feats)
                outs = outs + (cls_score, bbox_pred)
        # mask heads
        if self.with_mask:
            mask_rois = rois[:100]
            for i in range(self.num_stages):
                mask_feats = self.mask_roi_extractor[i](
                    x[:self.mask_roi_extractor[i].num_inputs], mask_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
                mask_pred = self.mask_head[i](mask_feats)
                outs = outs + (mask_pred, )
        return outs


    def forward_train_single(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):

        x = self.extract_feat_pair(img)
        losses = dict()

        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)

            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg.rcnn[i]
            lw = self.train_cfg.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            if self.with_bbox or self.with_mask:
                bbox_assigner = build_assigner(rcnn_train_cfg.assigner)
                bbox_sampler = build_sampler(
                    rcnn_train_cfg.sampler, context=self)
                num_imgs = img.size(0)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]

                for j in range(num_imgs):
                    assign_result = bbox_assigner.assign(
                        proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[j],
                        gt_bboxes[j],
                        gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_roi_extractor = self.bbox_roi_extractor[i]
            bbox_head = self.bbox_head[i]

            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                            rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = bbox_head(bbox_feats)

            bbox_targets = bbox_head.get_target(sampling_results, gt_bboxes,
                                                gt_labels, rcnn_train_cfg)
            loss_bbox = bbox_head.loss(cls_score, bbox_pred, *bbox_targets)
            for name, value in loss_bbox.items():
                losses['s{}.{}'.format(i, name)] = (
                    value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                if not self.share_roi_extractor:
                    mask_roi_extractor = self.mask_roi_extractor[i]
                    pos_rois = bbox2roi(
                        [res.pos_bboxes for res in sampling_results])
                    mask_feats = mask_roi_extractor(
                        x[:mask_roi_extractor.num_inputs], pos_rois)
                    if self.with_shared_head:
                        mask_feats = self.shared_head(mask_feats)
                else:
                    # reuse positive bbox feats
                    pos_inds = []
                    device = bbox_feats.device
                    for res in sampling_results:
                        pos_inds.append(
                            torch.ones(
                                res.pos_bboxes.shape[0],
                                device=device,
                                dtype=torch.uint8))
                        pos_inds.append(
                            torch.zeros(
                                res.neg_bboxes.shape[0],
                                device=device,
                                dtype=torch.uint8))
                    pos_inds = torch.cat(pos_inds)
                    mask_feats = bbox_feats[pos_inds]
                mask_head = self.mask_head[i]
                mask_pred = mask_head(mask_feats)
                mask_targets = mask_head.get_target(sampling_results, gt_masks,
                                                    rcnn_train_cfg)
                pos_labels = torch.cat(
                    [res.pos_gt_labels for res in sampling_results])
                loss_mask = mask_head.loss(mask_pred, mask_targets, pos_labels)
                for name, value in loss_mask.items():
                    losses['s{}.{}'.format(i, name)] = (
                        value * lw if 'loss' in name else value)

            # refine bboxes
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                roi_labels = bbox_targets[0]  # bbox_targets is a tuple
                with torch.no_grad():
                    proposal_list = bbox_head.refine_bboxes(
                        rois, roi_labels, bbox_pred, pos_is_gts, img_meta)
        return losses

    def forward_train_pair(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        # img: b, 2*c = 6, h, w
        b, c, h, w = img.shape
        img = img.reshape(-1, c // 2, h, w)
        # img: 2*b, c, h, w [0,1] , [2,3] , ......
        x = self.extract_feat(img)
        # x : tuple,5 layer
        losses = dict()

        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_outs_half = []
            for outs_0 in rpn_outs:
                tmp = []
                for outs_1 in outs_0:
                    tmp.append(outs_1[::2, :, :, :])
                rpn_outs_half.append(tmp)

            rpn_loss_inputs = tuple(rpn_outs_half) + (gt_bboxes, img_meta,
                                                      self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)


            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)

            # copy train img_meta to pair. 1,2,3->1,1,2,2,3,3
            img_meta = [img_meta[i // 2] for i in range(2 * len(img_meta))]

            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # generate blank gt for normal img
        gt_bboxes_ = []
        gt_labels_ = []
        for i in range(len(gt_bboxes)):
            gt_bboxes_.append(gt_bboxes[i])
            gt_bboxes_.append(torch.Tensor([[1, 1, 1, 1]]).to(gt_bboxes[i].device))
            gt_labels_.append(gt_labels[i])
            gt_labels_.append(torch.Tensor([[0]]).to(gt_labels[i].device))

        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg.rcnn[i]
            lw = self.train_cfg.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []

            if self.with_bbox or self.with_mask:
                bbox_assigner = build_assigner(rcnn_train_cfg.assigner)
                bbox_sampler = build_sampler(
                    rcnn_train_cfg.sampler, context=self)
                assert img.size(0) % 2 == 0
                num_pairs = img.size(0) // 2
                num_imgs = img.size(0)

                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]
                sampling_results = []

                for j in range(num_pairs):
                    i_train = 2 * j
                    i_normal = i_train + 1

                    assign_result_train = bbox_assigner.assign(proposal_list[i_train],
                                                               gt_bboxes_[i_train],
                                                               gt_bboxes_ignore[i_train],
                                                               gt_labels_[i_train])

                    assign_result_normal = bbox_assigner.assign(proposal_list[i_normal],
                                                                gt_bboxes_[i_normal],
                                                                gt_bboxes_ignore[i_normal],
                                                                gt_labels_[i_normal])

                    sampling_result_train, sampling_results_normal = bbox_sampler.pair_sample(
                        assign_result_train,
                        assign_result_normal,
                        proposal_list[i_train],
                        proposal_list[i_normal],
                        gt_bboxes_[i_train],
                        gt_labels_[i_train],
                        feats_train=[lvl_feat[i_train][None] for lvl_feat in x],
                        feats_normal=[lvl_feat[i_normal][None] for lvl_feat in x])
                    sampling_results.append(sampling_result_train)
                    sampling_results.append(sampling_results_normal)



            # bbox head forward and loss
            bbox_roi_extractor = self.bbox_roi_extractor[i]
            bbox_head = self.bbox_head[i]

            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                            rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = bbox_head(bbox_feats)

            bbox_targets = bbox_head.get_target(sampling_results, gt_bboxes,
                                                gt_labels, rcnn_train_cfg)
            loss_bbox = bbox_head.loss(cls_score, bbox_pred, *bbox_targets)
            for name, value in loss_bbox.items():
                losses['s{}.{}'.format(i, name)] = (
                    value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                if not self.share_roi_extractor:
                    mask_roi_extractor = self.mask_roi_extractor[i]
                    pos_rois = bbox2roi(
                        [res.pos_bboxes for res in sampling_results])
                    mask_feats = mask_roi_extractor(
                        x[:mask_roi_extractor.num_inputs], pos_rois)
                    if self.with_shared_head:
                        mask_feats = self.shared_head(mask_feats)
                else:
                    # reuse positive bbox feats
                    pos_inds = []
                    device = bbox_feats.device
                    for res in sampling_results:
                        pos_inds.append(
                            torch.ones(
                                res.pos_bboxes.shape[0],
                                device=device,
                                dtype=torch.uint8))
                        pos_inds.append(
                            torch.zeros(
                                res.neg_bboxes.shape[0],
                                device=device,
                                dtype=torch.uint8))
                    pos_inds = torch.cat(pos_inds)
                    mask_feats = bbox_feats[pos_inds]
                mask_head = self.mask_head[i]
                mask_pred = mask_head(mask_feats)
                mask_targets = mask_head.get_target(sampling_results, gt_masks,
                                                    rcnn_train_cfg)
                pos_labels = torch.cat(
                    [res.pos_gt_labels for res in sampling_results])
                loss_mask = mask_head.loss(mask_pred, mask_targets, pos_labels)
                for name, value in loss_mask.items():
                    losses['s{}.{}'.format(i, name)] = (
                        value * lw if 'loss' in name else value)

            # refine bboxes
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                roi_labels = bbox_targets[0]  # bbox_targets is a tuple
                # print('stage:{}'.format(i))
                # ic(proposal_list[1])
                with torch.no_grad():
                    proposal_list = bbox_head.refine_bboxes(
                        rois, roi_labels, bbox_pred, pos_is_gts, img_meta)
                # ic(proposal_list[1])
        # exit(0)

        return losses

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        if self.normal_train is False:
            return self.forward_train_single(img,
                                             img_meta,
                                             gt_bboxes,
                                             gt_labels,
                                             gt_bboxes_ignore=gt_bboxes_ignore,
                                             gt_masks=gt_masks,
                                             proposals=proposals)
        else:
            return self.forward_train_pair(img,
                                             img_meta,
                                             gt_bboxes,
                                             gt_labels,
                                             gt_bboxes_ignore=gt_bboxes_ignore,
                                             gt_masks=gt_masks,
                                             proposals=proposals)

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        x = self.extract_feat_pair(img)
        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        img_shape = img_meta[0]['img_shape']
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg.rcnn

        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_roi_extractor = self.bbox_roi_extractor[i]
            bbox_head = self.bbox_head[i]

            bbox_feats = bbox_roi_extractor(
                x[:len(bbox_roi_extractor.featmap_strides)], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)

            cls_score, bbox_pred = bbox_head(bbox_feats)
            ms_scores.append(cls_score)

            if self.test_cfg.keep_all_stages:
                det_bboxes, det_labels = bbox_head.get_det_bboxes(
                    rois,
                    cls_score,
                    bbox_pred,
                    img_shape,
                    scale_factor,
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
                bbox_result = bbox2result(det_bboxes, det_labels,
                                          bbox_head.num_classes)
                ms_bbox_result['stage{}'.format(i)] = bbox_result

                if self.with_mask:
                    mask_roi_extractor = self.mask_roi_extractor[i]
                    mask_head = self.mask_head[i]
                    if det_bboxes.shape[0] == 0:
                        mask_classes = mask_head.num_classes - 1
                        segm_result = [[] for _ in range(mask_classes)]
                    else:
                        _bboxes = (
                            det_bboxes[:, :4] *
                            scale_factor if rescale else det_bboxes)
                        mask_rois = bbox2roi([_bboxes])
                        mask_feats = mask_roi_extractor(
                            x[:len(mask_roi_extractor.featmap_strides)],
                            mask_rois)
                        if self.with_shared_head:
                            mask_feats = self.shared_head(mask_feats, i)
                        mask_pred = mask_head(mask_feats)
                        segm_result = mask_head.get_seg_masks(
                            mask_pred, _bboxes, det_labels, rcnn_test_cfg,
                            ori_shape, scale_factor, rescale)
                    ms_segm_result['stage{}'.format(i)] = segm_result

            if i < self.num_stages - 1:
                bbox_label = cls_score.argmax(dim=1)
                rois = bbox_head.regress_by_class(rois, bbox_label, bbox_pred,
                                                  img_meta[0])

        cls_score = sum(ms_scores) / self.num_stages
        det_bboxes, det_labels = self.bbox_head[-1].get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)
        ms_bbox_result['ensemble'] = bbox_result

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                mask_classes = self.mask_head[-1].num_classes - 1
                segm_result = [[] for _ in range(mask_classes)]
            else:
                if isinstance(scale_factor, float):  # aspect ratio fixed
                    _bboxes = (
                        det_bboxes[:, :4] *
                        scale_factor if rescale else det_bboxes)
                else:
                    _bboxes = (
                        det_bboxes[:, :4] *
                        torch.from_numpy(scale_factor).to(det_bboxes.device)
                        if rescale else det_bboxes)

                mask_rois = bbox2roi([_bboxes])
                aug_masks = []
                for i in range(self.num_stages):
                    mask_roi_extractor = self.mask_roi_extractor[i]
                    mask_feats = mask_roi_extractor(
                        x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                    if self.with_shared_head:
                        mask_feats = self.shared_head(mask_feats)
                    mask_pred = self.mask_head[i](mask_feats)
                    aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                merged_masks = merge_aug_masks(aug_masks,
                                               [img_meta] * self.num_stages,
                                               self.test_cfg.rcnn)
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks, _bboxes, det_labels, rcnn_test_cfg,
                    ori_shape, scale_factor, rescale)
            ms_segm_result['ensemble'] = segm_result

        if not self.test_cfg.keep_all_stages:
            if self.with_mask:
                results = (ms_bbox_result['ensemble'],
                           ms_segm_result['ensemble'])
            else:
                results = ms_bbox_result['ensemble']
        else:
            if self.with_mask:
                results = {
                    stage: (ms_bbox_result[stage], ms_segm_result[stage])
                    for stage in ms_bbox_result
                }
            else:
                results = ms_bbox_result

        return results

    # def aug_test(self, img, img_meta, proposals=None, rescale=False):
    #     raise NotImplementedError

    def aug_test_(self, imgs, img_metas, proposals=None, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory

        proposal_list = self.aug_test_rpn(
            self.extract_feats_pair(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.multi_bboxes_test(self.extract_feats_pair(imgs), img_metas, proposal_list,
                                                  self.test_cfg.rcnn, rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head[-1].num_classes)

        # proposal_list = self.aug_test_rpn(
        #     self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        # det_bboxes, det_labels = self.multi_bboxes_test(self.extract_feats(imgs), img_metas, proposal_list,
        #                                           self.test_cfg.rcnn, rescale)
        # bbox_results = bbox2result(det_bboxes, det_labels,
        #                            self.bbox_head[-1].num_classes)

        return bbox_results


    def aug_test__(self, imgs, img_metas, proposals=None, rescale=False, temp_feats=None):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # save feats to speed up
        if temp_feats is None:
            x, temp_feats = self.extract_feats_pair_test(imgs)
            proposal_list = self.aug_test_rpn(
                x, img_metas, self.test_cfg.rpn)

            det_bboxes, det_labels = self.multi_bboxes_test(x, img_metas, proposal_list,
                                                      self.test_cfg.rcnn, rescale)
            bbox_results = bbox2result(det_bboxes, det_labels,
                                       self.bbox_head[-1].num_classes)
            return bbox_results, temp_feats

        else:
            x = self.extract_feats_pair_test(imgs, temp_feats)
            proposal_list = self.aug_test_rpn(
                x, img_metas, self.test_cfg.rpn)

            det_bboxes, det_labels = self.multi_bboxes_test(x, img_metas, proposal_list,
                                                            self.test_cfg.rcnn, rescale)
            bbox_results = bbox2result(det_bboxes, det_labels,
                                       self.bbox_head[-1].num_classes)
            return bbox_results, temp_feats


    def aug_test(self, imgs, img_metas, is_new, proposals=None, rescale=False, temp_feats=None):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # save feats to speed up
        if is_new[0] == 1:
            x, temp_feats = self.extract_feats_pair_test(imgs)
            proposal_list = self.aug_test_rpn(
                x, img_metas, self.test_cfg.rpn)

            det_bboxes, det_labels = self.multi_bboxes_test(x, img_metas, proposal_list,
                                                      self.test_cfg.rcnn, rescale)
            bbox_results = bbox2result(det_bboxes, det_labels,
                                       self.bbox_head[-1].num_classes)
            return bbox_results, temp_feats

        else:
            x = self.extract_feats_pair_test(imgs, temp_feats)
            proposal_list = self.aug_test_rpn(
                x, img_metas, self.test_cfg.rpn)

            det_bboxes, det_labels = self.multi_bboxes_test(x, img_metas, proposal_list,
                                                            self.test_cfg.rcnn, rescale)
            bbox_results = bbox2result(det_bboxes, det_labels,
                                       self.bbox_head[-1].num_classes)
            return bbox_results, temp_feats

        # return bbox_results

    def show_result(self, data, result, **kwargs):
        if self.with_mask:
            ms_bbox_result, ms_segm_result = result
            if isinstance(ms_bbox_result, dict):
                result = (ms_bbox_result['ensemble'],
                          ms_segm_result['ensemble'])
        else:
            if isinstance(result, dict):
                result = result['ensemble']
        super(CascadeRCNN_pair, self).show_result(data, result, **kwargs)
