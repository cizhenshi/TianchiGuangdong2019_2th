# -*- coding: utf-8 -*-
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import init_detector, inference_detector, LoadImage, inference_detector_, show_result
from mmdet.models import build_detector
import glob, os, json
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
from prettytable import PrettyTable
from icecream import ic
import torch

from functools import partial
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector

from multiprocessing import Pool
import argparse

def generate_txt(ann, out_path, CLASSES):
    ann = mmcv.load(ann)
    ind2cls = mmcv.load("./source/ind2cls.pkl")
    detpath = "./result/{}".format(out_path)
    if not os.path.exists(detpath):
        os.makedirs(detpath)
    res = {CLASSES[cls]: [] for cls in range(len(CLASSES))}
    print(res)
    for i in tqdm(range(len(ann))):
        result = ann[i]
        name = result['name']
        curr_class = ind2cls[result['category']-1]
        bbox = [str(x) for x in result['bbox']]
        score = result['score']
        out = name + " "+str(score)+ " "+" ".join(bbox)
        res[curr_class].append(out)
    for key in res.keys():
        fp = open("./result/{}/".format(out_path)+key+".txt", 'w')
        for line in res[key]:
            fp.write(line+"\n")
        fp.close()


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
      Compute VOC AP given precision and recall.
      If use_07_metric is true, uses the
      VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
        ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        # first appicend sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def eval_warpper(classname, detpath, ovthresh, coco):
    ind2cls = mmcv.load("./source/ind2cls.pkl")
    imgIds = coco.getImgIds()
    recs = {}
    use_07_metric = False
    for imgid in imgIds:
        img = coco.loadImgs(imgid)[0]
        #### file name reflect
        file_name = img['file_name'].split('/')[-1]
        annIds = coco.getAnnIds(imgIds=[imgid], iscrowd=None)
        anns = coco.loadAnns(annIds)
        objects = []
        for ann in anns:
            obj = {}
            obj['name'] = ind2cls[ann['category_id']]
#             obj['name'] = submit_reflect[ind_to_class[ann['category_id']]]
            xmin, ymin, w, h = ann['bbox']
            obj['bbox'] = [xmin, ymin, xmin+w, ymin+h]
            objects.append(obj)
        recs[file_name] = objects
    class_recs = {}
    npos = 0
    for filename in list(recs.keys()):
        R = [obj for obj in recs[filename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        det = [False] * len(R)
        npos = npos + len(bbox)
        class_recs[filename] = {'bbox': bbox,
                                 'det': det}
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
    BB = np.reshape(BB, (-1, 2, 2))
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    if BB.shape[0] > 0:
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float).flatten()
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float).reshape(-1,4)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                BBGT_xmin =  np.min(BBGT[:, 0::2], axis=1)
                BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
                BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
                BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
                bb_xmin = np.min(bb[0::2])
                bb_ymin = np.min(bb[1::2])
                bb_xmax = np.max(bb[0::2])
                bb_ymax = np.max(bb[1::2])

                ixmin = np.maximum(BBGT_xmin, bb_xmin)
                iymin = np.maximum(BBGT_ymin, bb_ymin)
                ixmax = np.minimum(BBGT_xmax, bb_xmax)
                iymax = np.minimum(BBGT_ymax, bb_ymax)
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                       (BBGT_xmax - BBGT_xmin + 1.) *
                       (BBGT_ymax - BBGT_ymin + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    return classname, round(ap, 2)


def evaluate(coco, iou_thresh):
    CLASSES = mmcv.load("./source/ind2cls.pkl")
    detpath="./result/detection/{}.txt"
    eval_result = {}
    rets = []
    for class_name in CLASSES:
        rets.append(eval_warpper(class_name, detpath, iou_thresh, coco))
    aps = []
    for ret in rets:
        class_name, ap = ret
        eval_result[class_name] = ap
        aps.append(ap)
    return eval_result, np.round(np.mean(np.array(aps)), 2)

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


def inference_model(config_file, checkpoint_file, path, out, thresh):
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


def compute_acc(val_result):
    predicts = mmcv.load(val_result)
    gt_normal = set(mmcv.load("../data/round2_data/split/normal_0.pkl"))
    gt_defect = set(mmcv.load("../data/round2_data/split/val_0.pkl"))
    total = len(gt_normal) + len(gt_defect)
    correct = 0.0
    pd_defect = set()
    for predict in tqdm(predicts):
        pd_defect.add(predict["name"])
    print(len(pd_defect))
    for defect in gt_defect:
        if defect in pd_defect:
            correct += 1
    for normal in gt_normal:
        if normal+".jpg" not in pd_defect:
            correct += 1
    return round(correct/total, 2)
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
    thresh_box = 1e-3
    # inference_model_(config_file, checkpoint_file, path, out, thresh, thresh_box)
    inference_model_with_loader(config_file, checkpoint_file, path, out, thresh, thresh_box)

    CLASSES = mmcv.load("./source/ind2cls.pkl")
    generate_txt(out, "detection", CLASSES)
    gt_json = "../data/round2_data/val_0.json"
    # acc = compute_acc(out)
    coco = COCO(gt_json)
    result1, m1 = evaluate(coco, 0.1)
    result2, m3 = evaluate(coco, 0.3)
    result3, m5 = evaluate(coco, 0.5)
    o = PrettyTable(["class", "AP@0.1", "AP@0.3", "AP@0.5", "Avg"])
    for class_name in result1:
        o.add_row([class_name, result1[class_name], result2[class_name],
                   result3[class_name], round((result1[class_name]+result2[class_name]+result3[class_name])/3., 2)])
    o.add_row(["mAP", m1, m3, m5, round((m1+m3+m5)/3., 2)])
    print(o)
    # print("acc is {}".format(acc))
