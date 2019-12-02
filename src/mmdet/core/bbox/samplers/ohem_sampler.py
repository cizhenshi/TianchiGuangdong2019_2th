import torch

from ..transforms import bbox2roi
from .base_sampler import BaseSampler

from icecream import ic

class OHEMSampler(BaseSampler):

    def __init__(self,
                 num,
                 pos_fraction,
                 context,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        super(OHEMSampler, self).__init__(num, pos_fraction, neg_pos_ub,
                                          add_gt_as_proposals)
        if not hasattr(context, 'num_stages'):
            self.bbox_roi_extractor = context.bbox_roi_extractor
            self.bbox_head = context.bbox_head
        else:
            self.bbox_roi_extractor = context.bbox_roi_extractor[
                context.current_stage]
            self.bbox_head = context.bbox_head[context.current_stage]

    def hard_mining(self, inds, num_expected, bboxes, labels, feats):
        with torch.no_grad():
            rois = bbox2roi([bboxes])
            bbox_feats = self.bbox_roi_extractor(
                feats[:self.bbox_roi_extractor.num_inputs], rois)
            cls_score, _ = self.bbox_head(bbox_feats)
            loss = self.bbox_head.loss(
                cls_score=cls_score,
                bbox_pred=None,
                labels=labels,
                label_weights=cls_score.new_ones(cls_score.size(0)),
                bbox_targets=None,
                bbox_weights=None,
                reduction_override='none')['loss_cls']
            _, topk_loss_inds = loss.topk(num_expected)
        return inds[topk_loss_inds]

    def _sample_pos(self,
                    assign_result,
                    num_expected,
                    bboxes=None,
                    feats=None,
                    **kwargs):
        # Sample some hard positive samples
        pos_inds = torch.nonzero(assign_result.gt_inds > 0)

        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.hard_mining(pos_inds, num_expected, bboxes[pos_inds],
                                    assign_result.labels[pos_inds], feats)

    def _sample_pos_pair(self,
                    assign_result,
                    num_expected,
                    bboxes=None,
                    feats_train=None,
                    feats_normal=None,
                    **kwargs):
        # Sample some hard positive samples
        pos_inds = torch.nonzero(assign_result.gt_inds > 0)

        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.hard_mining(pos_inds, num_expected, bboxes[pos_inds],
                                    assign_result.labels[pos_inds], feats_train)


    def _sample_neg(self,
                    assign_result,
                    num_expected,
                    bboxes=None,
                    feats=None,
                    **kwargs):
        # Sample some hard negative samples
        neg_inds = torch.nonzero(assign_result.gt_inds == 0)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.hard_mining(neg_inds, num_expected, bboxes[neg_inds],
                                    assign_result.labels[neg_inds], feats)


    def hard_mining_pair(self, inds_0, inds_1, num_expected, bboxes_0, bboxes_1, labels_0, labels_1, feats_0, feats_1):
        with torch.no_grad():
            rois_0 = bbox2roi([bboxes_0])
            bbox_feats_0 = self.bbox_roi_extractor(
                feats_0[:self.bbox_roi_extractor.num_inputs], rois_0)
            cls_score_0, _ = self.bbox_head(bbox_feats_0)
            loss_0 = self.bbox_head.loss(
                cls_score=cls_score_0,
                bbox_pred=None,
                labels=labels_0,
                label_weights=cls_score_0.new_ones(cls_score_0.size(0)),
                bbox_targets=None,
                bbox_weights=None,
                reduction_override='none')['loss_cls']

            rois_1 = bbox2roi([bboxes_1])
            bbox_feats_1 = self.bbox_roi_extractor(
                feats_1[:self.bbox_roi_extractor.num_inputs], rois_1)
            cls_score_1, _ = self.bbox_head(bbox_feats_1)
            loss_1 = self.bbox_head.loss(
                cls_score=cls_score_1,
                bbox_pred=None,
                labels=labels_1,
                label_weights=cls_score_1.new_ones(cls_score_1.size(0)),
                bbox_targets=None,
                bbox_weights=None,
                reduction_override='none')['loss_cls']

            train_num = num_expected // 2
            _, topk_loss_inds_0 = loss_0.topk(min(loss_0.size(0),train_num))
            _, topk_loss_inds_1 = loss_1.topk(min(loss_1.size(0),train_num))

            # loss_total = torch.cat((loss_0, loss_1), dim=0)
            # _, topk_loss_inds_total = loss_total.topk(num_expected)
            # ind_loc = loss_0.size(0)
            # topk_loss_inds_0 = torch.Tensor([x for x in topk_loss_inds_total if x < ind_loc]).long()
            # topk_loss_inds_1 = torch.Tensor([x for x in topk_loss_inds_total if x >= ind_loc]).long() - ind_loc
        return inds_0[topk_loss_inds_0], inds_1[topk_loss_inds_1]

    def _sample_neg_pair(self,
                         assign_result_train,
                         assign_result_normal,
                         num_expected,
                         bboxes_train=None,
                         bboxes_normal=None,
                         feats_train=None,
                         feats_normal=None,
                         **kwargs):
        neg_inds_train = torch.nonzero(assign_result_train.gt_inds == 0)
        neg_inds_normal = torch.nonzero(assign_result_normal.gt_inds == 0)

        if neg_inds_train.numel() != 0:
            neg_inds_train = neg_inds_train.squeeze(1)

        if neg_inds_normal.numel() != 0:
            neg_inds_normal = neg_inds_normal.squeeze(1)

        if len(neg_inds_train) + len(neg_inds_normal) <= num_expected:
            return neg_inds_train, neg_inds_normal
        else:
            return self.hard_mining_pair(neg_inds_train, neg_inds_normal, num_expected,
                                         bboxes_train[neg_inds_train], bboxes_normal[neg_inds_normal],
                                         assign_result_train.labels[neg_inds_train],
                                         assign_result_normal.labels[neg_inds_normal],
                                         feats_train, feats_normal)