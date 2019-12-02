from abc import ABCMeta, abstractmethod

import torch

from .sampling_result import SamplingResult
from icecream import ic

class BaseSampler(metaclass=ABCMeta):

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 pair_fraction=0.5,
                 **kwargs):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
        self.pair_fraction = pair_fraction
        self.pos_sampler = self
        self.neg_sampler = self

    @abstractmethod
    def _sample_pos(self, assign_result, num_expected, **kwargs):
        pass

    @abstractmethod
    def _sample_neg(self, assign_result, num_expected, **kwargs):
        pass



    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               **kwargs):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (Tensor): Boxes to be sampled from.
            gt_bboxes (Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.

        Returns:
            :obj:`SamplingResult`: Sampling result.
        """
        bboxes = bboxes[:, :4]

        gt_flags = bboxes.new_zeros((bboxes.shape[0], ), dtype=torch.uint8)
        if self.add_gt_as_proposals:
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(
            assign_result, num_expected_neg, bboxes=bboxes, **kwargs)
        neg_inds = neg_inds.unique()

        return SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                              assign_result, gt_flags)

    def pair_sample(self,
                    assign_result_train,
                    assign_result_normal,
                    bboxes_train,
                    bboxes_normal,
                    gt_bboxes,
                    gt_labels=None,
                    **kwargs):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (Tensor): Boxes to be sampled from.
            gt_bboxes (Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.

        Returns:
            :obj:`SamplingResult`: Sampling result.
        """
        #300, 5 -> throw the scores
        bboxes_train = bboxes_train[:, :4]
        bboxes_normal = bboxes_normal[:, :4]

        gt_flags_train = bboxes_train.new_zeros((bboxes_train.shape[0], ), dtype=torch.uint8)
        gt_flags_normal = bboxes_normal.new_zeros((bboxes_normal.shape[0], ), dtype=torch.uint8)

        if self.add_gt_as_proposals :
            bboxes_train = torch.cat([gt_bboxes, bboxes_train], dim=0)
            assign_result_train.add_gt_(gt_labels)
            gt_ones = bboxes_train.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags_train = torch.cat([gt_ones, gt_flags_train])


        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos_pair(
            assign_result_train, num_expected_pos, bboxes=bboxes_train, **kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()

        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds_train, neg_inds_normal = self.neg_sampler._sample_neg_pair(
            assign_result_train, assign_result_normal, num_expected_neg,
            bboxes_train=bboxes_train, bboxes_normal=bboxes_normal, **kwargs)

        # neg_inds = self.neg_sampler._sample_neg(
        #     assign_result_train, num_expected_neg, bboxes=bboxes_train, **kwargs)
        neg_inds_train = neg_inds_train.unique()
        neg_inds_normal = neg_inds_normal.unique()

        return SamplingResult(pos_inds, neg_inds_train, bboxes_train, gt_bboxes,assign_result_train, gt_flags_train),\
               SamplingResult(None, neg_inds_normal, bboxes_normal, gt_bboxes,assign_result_normal, gt_flags_normal)