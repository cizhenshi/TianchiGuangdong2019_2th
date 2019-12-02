# -*- coding: utf-8 -*-
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import init_detector, inference_detector, LoadImage, inference_detector_, show_result
from mmdet.models import build_detector
import glob, os, json
import numpy as np
from tqdm import tqdm

import torch
from functools import partial
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
import argparse


class DataPrefetcher():
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in self.batch:
                self.batch[k] = scatter(self.batch[k], [self.device])[0]

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, path, model):
        img_dirs = os.listdir(path)
        img_dirs.sort()
        imgs = []
        template_name = ''
        for img_dir in img_dirs:
            name1 = img_dir + ".jpg"
            name2 = 'template_' + img_dir.split('_')[0] + '.jpg'
            path1 = os.path.join(path, img_dir, name1)
            if name2 != template_name:
                template_name = name2
                path2 = os.path.join(path, img_dir, name2)
                imgs.append((path1, path2))
            else:
                imgs.append((path1, None))
        self.imgs = imgs

        cfg = model.cfg
        self.device = next(model.parameters()).device  # model device
        test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
        self.test_pipeline = Compose(test_pipeline)

    def __getitem__(self, item):
        img, img_pair = self.imgs[item]
        data1 = dict(img=img)
        data1 = self.test_pipeline(data1)
        if img_pair is not None:
            data2 = dict(img=img_pair)
            data2 = self.test_pipeline(data2)
            data2_img = data2['img']
            for i in range(len(data1['img'])):
                data1['img'][i] = torch.cat((data1['img'][i].data, data2_img[i].data))
            data = data1
            data['is_new'] = 1
            # data = scatter(collate([data], samples_per_gpu=1), [self.device])[0]
            return data
        else:
            data = data1
            data['is_new'] = 0
            # data = scatter(collate([data], samples_per_gpu=1), [self.device])[0]
            return data

    def __len__(self):
        return len(self.imgs)

def inference_model_with_loader(config_file, checkpoint_file, path, out, thresh, thresh_box):
    # build the model from a config file and a checkpoint file
    print('loading model...')
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    print('loading complete!')
    # 测试多张图片
    dataset = EvalDataset(path=path, model=model)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         collate_fn=partial(collate, samples_per_gpu=1),
                                         num_workers=8,
                                         pin_memory=True)
    prefetcher = DataPrefetcher(loader, device=dataset.device)
    result = []
    pbar = mmcv.ProgressBar(len(dataset))
    temp_feats = None
    with torch.no_grad():
        batch = prefetcher.next()
        i = 0
        while batch is not None:
            pbar.update()
            res, temp_feats = model(return_loss=False, temp_feats=temp_feats, rescale=True, **batch)
            bboxes = np.vstack(res)
            labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(res)]
            labels = np.concatenate(labels)
            if len(bboxes) > 0:
                det_flag = False
                for j, bbox in enumerate(bboxes):
                    if float(bbox[4]) > thresh:
                        det_flag = True
                        break
                if det_flag is True:
                    for j, bbox in enumerate(bboxes):
                        if float(bbox[4]) > thresh_box:
                            name = dataset.imgs[i][0].split("/")[-1]
                            res_line = {'name': name, 'category': int(labels[j] + 1),
                                        'bbox': [round(float(x), 2) for x in bbox[
                                                                             :4]], 'score': float(bbox[4])}
                            result.append(res_line)
            i += 1
            batch = prefetcher.next()
        # for i, data in enumerate(loader):
        #     data = scatter(data, [dataset.device])[0]
        #     pbar.update()
        #     res, temp_feats = model(return_loss=False, temp_feats=temp_feats, rescale=True, **data)
        #     bboxes = np.vstack(res)
        #     labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(res)]
        #     labels = np.concatenate(labels)
        #     if len(bboxes) > 0:
        #         det_flag = False
        #         for j, bbox in enumerate(bboxes):
        #             if float(bbox[4]) > thresh:
        #                 det_flag = True
        #                 break
        #         if det_flag is True:
        #             for j, bbox in enumerate(bboxes):
        #                 if float(bbox[4]) > thresh_box:
        #                     name = dataset.imgs[i][0].split("/")[-1]
        #                     res_line = {'name': name, 'category': int(labels[j] + 1),
        #                                 'bbox': [round(float(x), 2) for x in bbox[
        #                                                                      :4]], 'score': float(bbox[4])}
        #                     result.append(res_line)
        #     # 写入结果
    with open(out, 'w') as fp:
        json.dump(result, fp, indent=4, separators=(',', ': '))
    print('over!')


def inference_model(config_file, checkpoint_file, path, out, thresh=0):
    # build the model from a config file and a checkpoint file
    print('loading model...')
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    print('loading complete!')
    # 测试多张图片
    img_dis = os.listdir(path)
    imgs = []
    for img_dir in img_dis:
        imgs.append(os.path.join(path, img_dir, img_dir+".jpg"))
    result = []
    pbar = mmcv.ProgressBar(len(imgs))
    for i, img in enumerate(imgs):
        pbar.update()
        res = inference_detector(model, img)
        bboxes = np.vstack(res)
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(res)]
        labels = np.concatenate(labels)
        if len(bboxes) > 0:
            for j, bbox in enumerate(bboxes):
                if float(bbox[4]) > thresh:
                    name = imgs[i].split("/")[-1]
                    res_line = {'name': name, 'category': int(labels[j] + 1), 'bbox':[round(float(x),2) for x in bbox[
                                                                                                              :4]], 'score':float(bbox[4])}
                    result.append(res_line)
    # 写入结果
    with open(out, 'w') as fp:
         json.dump(result, fp, indent=4, separators=(',', ': '))
    print('over!')
    
def inference_model_(config_file, checkpoint_file, path, out, thresh, thresh_box):
    # build the model from a config file and a checkpoint file
    print('loading model...')
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    print('loading complete!')
    # 测试多张图片
    img_dirs = os.listdir(path)
    img_dirs.sort()
    imgs = []
    template_name = ''
    for img_dir in img_dirs:
        name1 = img_dir+".jpg"
        name2 = 'template_' + img_dir.split('_')[0] + '.jpg'
        path1 = os.path.join(path, img_dir, name1)
        if name2 != template_name:
            template_name = name2
            path2 = os.path.join(path, img_dir, name2)
            imgs.append((path1, path2))
        else:
            imgs.append((path1, None))


    result = []
    pbar = mmcv.ProgressBar(len(imgs))
    temp_feats = None

    for i, (img, img_pair) in enumerate(imgs):
        pbar.update()
        if img_pair is not None:
            res, temp_feats = inference_detector_(model, img, img_pair)
        else:
            res, temp_feats = inference_detector_(model, img, img_pair, temp_feats=temp_feats)
        # res = inference_detector_(model, img, img_pair)
        bboxes = np.vstack(res)
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(res)]
        labels = np.concatenate(labels)
        if len(bboxes) > 0:
            det_flag = False
            for j, bbox in enumerate(bboxes):
                if float(bbox[4]) > thresh:
                    det_flag = True
                    break
            if det_flag is True:
                for j, bbox in enumerate(bboxes):
                    if float(bbox[4]) > thresh_box:
                        name = imgs[i][0].split("/")[-1]
                        res_line = {'name': name, 'category': int(labels[j] + 1), 'bbox':[round(float(x),2) for x in bbox[
                                                                                                                  :4]], 'score':float(bbox[4])}
                        result.append(res_line)
    # 写入结果
    with open(out, 'w') as fp:
         json.dump(result, fp, indent=4, separators=(',', ': '))
    print('over!')

def inference_model__(config_file, checkpoint_file, path, out, thresh, thresh_box):
    # build the model from a config file and a checkpoint file
    print('loading model...')
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    print('loading complete!')
    # 测试多张图片
    img_dis = os.listdir(path)
    imgs = []
    for img_dir in img_dis:
        name1 = img_dir+".jpg"
        name2 = 'template_' + img_dir.split('_')[0] + '.jpg'
        path1 = os.path.join(path, img_dir, name1)
        path2 = os.path.join(path, img_dir, name2)
        imgs.append((path1, path2))
    result = []
    pbar = mmcv.ProgressBar(len(imgs))
    for i, (img, img_pair) in enumerate(imgs):
        pbar.update()
        res = inference_detector_(model, img, img_pair)
        bboxes = np.vstack(res)
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(res)]
        labels = np.concatenate(labels)
        if len(bboxes) > 0:
            det_flag = False
            for j, bbox in enumerate(bboxes):
                if float(bbox[4]) > thresh:
                    det_flag = True
                    break
            if det_flag is True:
                for j, bbox in enumerate(bboxes):
                    if float(bbox[4]) > thresh:
                        name = imgs[i][0].split("/")[-1]
                        res_line = {'name': name, 'category': int(labels[j] + 1), 'bbox':[round(float(x),2) for x in bbox[
                                                                                                                  :4]], 'score':float(bbox[4])}
                        result.append(res_line)
    # 写入结果
    with open(out, 'w') as fp:
         json.dump(result, fp, indent=4, separators=(',', ': '))
    print('over!')

###pipeline
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval result")
    parser.add_argument("--config", help="config file")
    parser.add_argument("--checkpoint", help="checkpoint file")
    parser.add_argument("--out", help="config file")
    parser.add_argument("--path")
    args = parser.parse_args()
    config_file = args.config
    checkpoint_file = args.checkpoint
    path = args.path
    out = args.out
    thresh = 0.35
    thresh_box = 0
    inference_model_with_loader(config_file, checkpoint_file, path, out, thresh, thresh_box)
    # inference_model_(config_file, checkpoint_file, path, out, thresh, thresh_box)
