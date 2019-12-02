import numpy as np
import torch

from .base_sampler import BaseSampler


class RandomSampler(BaseSampler):

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        super(RandomSampler, self).__init__(num, pos_fraction, neg_pos_ub,
                                            add_gt_as_proposals)

    @staticmethod
    def random_choice(gallery, num):
        """Random select some elements from the gallery.

        It seems that Pytorch's implementation is slower than numpy so we use
        numpy to randperm the indices.
        """
        assert len(gallery) >= num
        if isinstance(gallery, list):
            gallery = np.array(gallery)
        cands = np.arange(len(gallery))
        np.random.shuffle(cands)
        rand_inds = cands[:num]
        if not isinstance(gallery, np.ndarray):
            rand_inds = torch.from_numpy(rand_inds).long().to(gallery.device)
        return gallery[rand_inds]

    def random_choice_pair(self, gallery_1, gallery_2, num):
        """
        :param gallery_1: image ind with gt
        :param gallery_2: image ind without gt
        :param num: sample num
        :return:
        """
        assert len(gallery_1) + len(gallery_2) > num
        if isinstance(gallery_1, list):
            gallery_1 = np.array(gallery_1)
        if isinstance(gallery_2, list):
            gallery_2 = np.array(gallery_2)

        num_1 = int(num * self.pair_fraction)
        num_1 = min(len(gallery_1), num_1)
        cands = np.arange(len(gallery_1))
        np.random.shuffle(cands)
        rand_inds_1 = cands[:num_1]
        if not isinstance(gallery_1, np.ndarray):
            rand_inds_1 = torch.from_numpy(rand_inds_1).long().to(gallery_1.device)

        num_2 = num - num_1
        num_2 = min(len(gallery_2), num_2)
        cands = np.arange(len(gallery_2))
        np.random.shuffle(cands)
        rand_inds_2 = cands[:num_2]
        if not isinstance(gallery_2, np.ndarray):
            rand_inds_2 = torch.from_numpy(rand_inds_2).long().to(gallery_2.device)

        return gallery_1[rand_inds_1], gallery_2[rand_inds_2]

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples."""
        pos_inds = torch.nonzero(assign_result.gt_inds > 0)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Randomly sample some negative samples."""
        neg_inds = torch.nonzero(assign_result.gt_inds == 0)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)

    def _sample_pos_pair(self,
                    assign_result,
                    num_expected,
                    **kwargs):
        # Sample some hard positive samples
        pos_inds = torch.nonzero(assign_result.gt_inds > 0)

        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg_pair(self,
                         assign_result_train,
                         assign_result_normal,
                         num_expected,
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
            return self.random_choice_pair(neg_inds_train, neg_inds_normal, num_expected)
