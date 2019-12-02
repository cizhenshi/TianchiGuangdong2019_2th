import mmcv
import cv2
import os
import json
import mmcv
from multiprocessing  import Pool
from pycocotools.coco import COCO
import numpy as np
import time
from tqdm import tqdm_notebook as tqdm

def generate_submit(result_file, coco_file, out_file, thr=0.05):
    result = mmcv.load(result_file)
    coco = COCO(coco_file)
    imgIds = coco.getImgIds()
    submits = []
    for i in range(len(result)):
        imgId = imgIds[i]
        name = coco.loadImgs(imgId)[0]['file_name']
        predict = result[i]
        for j in range(len(predict)):
#             category = submit_reflect[ind_to_class[j]]
            category = j+1
            anns = predict[j]
            for ann in anns:
                score = float(ann[4])
                if score < thr:
                    continue
                bbox = [ round(float(x), 2) for x in ann[:4]]
                submits.append({'name': name,'category': category,'bbox':bbox,'score': score})
    print(len(submits))
    with open(out_file, 'w') as fp:
         json.dump(submits, fp, indent=4, separators=(',', ': '))

def merge(file1, file2):
    acc_root = file1
    map_root = file2
    with open(acc_root) as fp:
        ann = json.loads(fp.read())
        print(len(ann))

    filterdict = {}
    for ann_item in ann:
        name = ann_item['name']
        if name not in filterdict:
            filterdict[name] = [1, [ann_item['score']]]
        else:
            filterdict[name][0] += 1
            filterdict[name][1].append(ann_item['score'])

    tmp = {}
    with open(map_root) as fp:
        ann = json.loads(fp.read())
    ann_res = []

    alpha = 0.1
    counter = 0
    for ann_item in ann:
        name = ann_item['name']
        if name not in tmp:
            tmp[name] = 0
        else:
            tmp[name] += 1

        if name not in filterdict:
            #         if ann_item['score'] > alpha:
            #             ann_res.append(ann_item)
            #             counter += 1
            continue
        else:
            ann_res.append(ann_item)
    t = time.strftime('%Y%m%d%H%M')
    out_path = '../submit/summit_{}.json'.format(t)
    print(len(ann_res))
    with open(out_path, 'w') as fp:
        json.dump(ann_res, fp, indent=4, separators=(',', ': '))


if __name__ == "__main__":
    if not os.path.exists("../submit"):
        os.makedirs("../submit")
    result_file = "../data/faster.pkl"
    coco_file = "../data/annotations/testB.json"
    file1 = '../data/faster.json'
    generate_submit(result_file, coco_file, file1,thr=0.05)
    result_file = "../data/cas101.pkl"
    file2 = '../data/cas101.json'
    generate_submit(result_file, coco_file, file2, thr=0.001)
    merge(file1, file2)

