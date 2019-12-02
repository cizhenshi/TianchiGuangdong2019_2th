import numpy as np
from pycocotools.coco import COCO

from .custom import CustomDataset
from .registry import DATASETS
from .pipelines import Compose
import torch
from mmcv.parallel import DataContainer as DC
import os
import random
from icecream import ic
import mmcv
@DATASETS.register_module
class CocopairDataset_r2(CustomDataset):

    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')

    def __init__(self,
                 ann_file,
                 pipeline,
                 normal_path=None,
                 normal_pipeline=None,
                 normal=False,
                 pair=False,
                 data_root=None,
                 img_prefix=None,
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False):
        super(CocopairDataset_r2, self).__init__(ann_file,
                                             pipeline,
                                             data_root,
                                             img_prefix,
                                             seg_prefix,
                                             proposal_file,
                                             test_mode)
        self.normal = normal
        self.pair = pair
        if self.normal:
            assert normal_path is not None, 'normal_path error'
            assert normal_pipeline is not None, 'normal_pipeline error'
        if self.pair:
            assert normal_pipeline is not None, 'normal_pipline error'

        self.normal_path = normal_path
        self.normal_pipeline = Compose(normal_pipeline)

        self.normal_dirs = []
        for lists in os.listdir(normal_path):
            self.normal_dirs.append(lists)

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def _filter_imgs(self, min_size=32, drop_ge100=False, drop_nogt=True):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if drop_nogt and self.img_ids[i] not in ids_with_ann:
                continue
            if drop_ge100 and len(self.coco.getAnnIds(self.img_ids[i])) > 100:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        results = self.pipeline(results)
        if self.normal is False and self.pair is False:
            return results
        # print(img_info['file_name'], img_info['filename'])\

        if self.pair is True:
            # path1 = img_info['file_name']
            # ic(path1)
            # dir_ = '/'.join(path1.split('/')[0:-1])
            # temp_ = '/template_' + path1.split('/')[-1].split('_')[0] + '.jpg'
            # path2 = dir_ + temp_
            path2 = img_info["template_name"]
            img_info_ = {}
            img_info_['file_name'] = path2
            img_info_['filename'] = path2
            img_info_['height'] = img_info['height']
            img_info_['width'] = img_info['width']
            results_pair = dict(img_info=img_info_)
            results_pair['scale'] = results['img_meta'].data['scale_factor']
            results_pair['flip'] = results['img_meta'].data['flip']
            results_pair['vflip'] = results['img_meta'].data['vflip']

            self.pre_pipeline(results_pair)
            results_pair['img_prefix'] = self.img_prefix
            results_pair = self.normal_pipeline(results_pair)
            results['img'] = DC(torch.cat((results['img'].data, results_pair['img'].data)), stack=True)
            return results
            # results['img'] = [results['img'], results_pair['img']]
        #
        # if self.normal is True:
        #     dir_name = random.choice(self.normal_dirs)
        #     names = os.listdir(os.path.join(self.normal_path, dir_name))
        #     if re.match('template_', names[0]) is not None:
        #         normal_pair_name = names[0]
        #         normal_name = names[1]
        #     elif re.match('template_', names[1]) is not None:
        #         normal_pair_name = names[1]
        #         normal_name = names[0]
        #     else:
        #         raise ValueError(
        #             'wron value in normal path')
        #     img_info = {}
        #     img_info['file_name']= normal_name
        #     img_info['filename']= normal_name
        #     # img_info['height']= 1000
        #     # img_info['width']= 2446
        #     results_normal = dict(img_info=img_info)
        #     results_normal['scale'] =  ['img_meta'].data['scale_factor']
        #     self.pre_pipeline(results_normal)
        #     results_normal['img_prefix'] = self.normal_path
        #     results_normal = self.normal_pipeline(results_normal)
        #
        #     if self.pair is True:
        #         img_info_ = {}
        #         img_info_['file_name'] = normal_pair_name
        #         img_info_['filename'] = normal_pair_name
        #         # img_info_['height'] = img_info['height']
        #         # img_info_['width'] = img_info['width']
        #         results_normal_pair = dict(img_info=img_info_)
        #         results_normal_pair['scale'] = results_normal['img_meta'].data['scale_factor']
        #         results_normal_pair['flip'] = results_normal['img_meta'].data['flip']
        #         results_normal_pair['vflip'] = results_normal['img_meta'].data['vflip']
        #
        #         self.pre_pipeline(results_normal_pair)
        #         results_normal_pair['img_prefix'] = self.img_prefix
        #         results_normal_pair = self.normal_pipeline(results_normal_pair)
        #
        # if self.normal is True and self.pair is True:
        #     results['img'] = [results['img'], results_pair['img'], results_normal['img'], results_normal_pair['img_prefix']]
        #     results['img_meta'] = [results['img_meta'], results_normal['img_meta']]
        # elif self.normal is True and self.pair is False:
        #     results['img'] = [results['img'], results_normal['img']]
        #     results['img_meta'] = [results['img_meta'], results_normal['img_meta']]
        # elif self.normal is False and self.pair is True:
        #     results['img'] = [results['img'], results_pair['img']]
        #
        # return results





    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        results = self.pipeline(results)
        if self.pair is False:
            return results
        # print(img_info['file_name'], img_info['filename'])

        path1 = img_info['file_name']
        dir_ = '/'.join(path1.split('/')[0:-1])
        temp_ = '/template_' + path1.split('/')[-1].split('_')[0] + '.jpg'
        path2 = dir_ + temp_
        img_info_ = {}
        img_info_['file_name'] = path2
        img_info_['filename'] = path2
        img_info_['height'] = img_info['height']
        img_info_['width'] = img_info['width']
        results_pair = dict(img_info=img_info_)
        self.pre_pipeline(results_pair)
        results_pair['img_prefix'] = self.img_prefix
        results_pair['flip'] = results['img_meta'][0].data['flip']
        results_pair['vflip'] = results['img_meta'][0].data['vflip']
        results_pair = self.pipeline(results_pair)
        for i in range(len(results['img'])):
            results['img'][i] = torch.cat((results['img'][i].data, results_pair['img'][i].data))
        # ic(results)
        return results