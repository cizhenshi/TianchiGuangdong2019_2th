# -*- coding: utf-8 -*-
import mmcv
import os
from tqdm import tqdm
import cv2
from pycocotools.coco import COCO
import shutil
import numpy as np
from icecream import ic
from collections import Counter


def construct_imginfo(root_dir, filename, template, h, w, ID):
    template_class = template.split(".")[0]
    filename = "{}/{}/{}".format(root_dir, filename.split('.')[0], filename)
    template = "{}/{}/template_{}".format(root_dir, filename.split("/")[1], template)
    image = {"license": 1,
             "file_name": filename,
             'template_name': template,
             "cls": template_class,
             "coco_url": "xxx",
             "height": h,
             "width": w,
             "date_captured": "2019-06-25",
             "flickr_url": "xxx",
             "id": ID
             }
    return image


def construct_ann(obj_id, ID, category_id, seg, area, bbox):
    ann = {"id": obj_id,
           "image_id": ID,
           "category_id": category_id,
           "segmentation": seg,
           "area": area,
           "bbox": bbox,
           "iscrowd": 0,
           }
    return ann


def add_normal(normal_dir, out_file):
    coco = COCO(out_file)
    ID = max(coco.getImgIds()) + 1
    annotations = mmcv.load(out_file)
    normal_list = os.listdir(normal_dir)
    for normal in tqdm(normal_list):
        source = "{}/{}".format(normal_dir, normal)
        img = cv2.imread(source + "/{}.jpg".format(normal))
        h, w, _ = img.shape
        filename = normal + ".jpg"
        template = normal.split("_")[0] + ".jpg"
        img_info = construct_imginfo("normal", filename, template, h, w, ID)
        ID += 1
        annotations["images"].append(img_info)
    print(len(annotations["images"]))
    a = open(out_file, 'w')
    a.close()
    mmcv.dump(annotations, out_file)


def generate_normal(normal_dir, out_file):
    cls2ind = mmcv.load("./source/cls2ind.pkl")
    ind2cls = mmcv.load("./source/ind2cls.pkl")
    info = {
        "description": "cloth",
        "url": "http://cocodataset.org",
        "version": "1.0",
        "year": 2014,
        "contributor": "COCO Consortium",
        "date_created": "2017/09/01"
    }
    license = [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"}]
    categories = []
    for ind in range(len(ind2cls)):
        category = {"id": ind, "name": ind2cls[ind], "supercategory": "object", }
        categories.append(category)
    annotations = {"info": info, "images": [], "annotations": [], "categories": categories, "license": license}
    ID = 0
    normal_list = os.listdir(normal_dir)
    for normal in tqdm(normal_list):
        source = "{}/{}".format(normal_dir, normal)
        img = cv2.imread(source + "/{}.jpg".format(normal))
        h, w, _ = img.shape
        filename = normal + ".jpg"
        template = normal.split("_")[0] + ".jpg"
        img_info = construct_imginfo("normal", filename, template, h, w, ID)
        ID += 1
        annotations["images"].append(img_info)
    print(len(annotations["images"]))
    a = open(out_file, 'w')
    a.close()
    mmcv.dump(annotations, out_file)

def generate_coco(annos, out_file):
    cls2ind = mmcv.load("./source/cls2ind.pkl")
    ind2cls = mmcv.load("./source/ind2cls.pkl")
    info = {
        "description": "cloth",
        "url": "http://cocodataset.org",
        "version": "1.0",
        "year": 2014,
        "contributor": "COCO Consortium",
        "date_created": "2017/09/01"
    }
    license = [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"}]
    categories = []
    for ind in range(len(ind2cls)):
        category = {"id": ind, "name": ind2cls[ind], "supercategory": "object", }
        categories.append(category)
    annotations = {"info": info, "images": [], "annotations": [], "categories": categories, "license": license}
    img_names = {}
    IMG_ID = 0
    OBJ_ID = 0
    for info in tqdm(annos):
        name = info['name']
        template = name.split('_')[0] + ".jpg"
        defect_name = info["defect_name"]
        bbox = info["bbox"]
        if name not in img_names:
            img_names[name] = IMG_ID
            img_path = "../data/round2_data/defect/{}/{}".format(name.split(".")[0], name)
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            img_info = construct_imginfo("defect", name, template, h, w, IMG_ID)
            annotations["images"].append(img_info)
            IMG_ID = IMG_ID + 1
        img_id = img_names[name]
        cat_ID = cls2ind[defect_name]
        xmin, ymin, xmax, ymax = bbox
        area = (ymax - ymin) * (xmax - xmin)
        seg = [[xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]]
        bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
        ann = construct_ann(OBJ_ID, img_id, cat_ID, seg, area, bbox)
        annotations["annotations"].append(ann)
        OBJ_ID += 1
    print(len(annotations["images"]))
    a = open(out_file, 'w')
    a.close()
    mmcv.dump(annotations, out_file)


def generate_train(coco, val):
    cls2ind = mmcv.load("./source/cls2ind.pkl")
    ind2cls = mmcv.load("./source/ind2cls.pkl")
    info = {
        "description": "cloth",
        "url": "http://cocodataset.org",
        "version": "1.0",
        "year": 2014,
        "contributor": "COCO Consortium",
        "date_created": "2017/09/01"
    }
    license = [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"}]
    categories = []
    for ind in range(len(ind2cls)):
        category = {"id": ind, "name": ind2cls[ind], "supercategory": "object", }
        categories.append(category)
    anno_train = {"info": info, "images": [], "annotations": [], "categories": categories, "license": license}
    anno_val = {"info": info, "images": [], "annotations": [], "categories": categories, "license": license}
    ids = coco.getImgIds()
    for imgId in ids:
        img_info = coco.loadImgs(imgId)[0]
        cls = img_info["cls"]
        ann_ids = coco.getAnnIds(img_info['id'])
        ann_info = coco.loadAnns(ann_ids)
        if cls in val:
            anno_val["images"].append(img_info)
            anno_val["annotations"] += ann_info
        else:
            anno_train["images"].append(img_info)
            anno_train["annotations"] += ann_info
    mmcv.dump(anno_train, "../data/round2_data/Annotations/anno_train.json")
    mmcv.dump(anno_val, "../data/round2_data/Annotations/anno_val.json")


def split(out_file):
    all_class = mmcv.load("./source/temp_cls.pkl")
    np.random.seed(1)
    val = np.random.choice(all_class, 28, replace=False)
    train = list(set(all_class) - set(val))
    coco = COCO(out_file)
    generate_train(coco, val)

if __name__ == "__main__":
    data_root = "../data/"
    annos1 = mmcv.load("../data/round2_data/Annotations/anno_train_0924.json")
    annos2 = mmcv.load("../data/round2_data/Annotations/anno_train_1004.json")
    annos = annos1 + annos2
    data_dir = "../data/round2_data"
    out_file = "{}/Annotations/train_1010.json".format(data_dir)
    print("convert to coco format...")
    generate_coco(annos, out_file)
    add_normal("../data/round2_data/normal", out_file)