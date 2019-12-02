# -*- coding: utf-8 -*-
import mmcv
import os
from tqdm import tqdm
import cv2
from pycocotools.coco import COCO
import shutil

def construct_imginfo(filename, h, w, ID):
    image = {"license": 1,
             "file_name": filename,
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


def generate_coco(ind_to_class, annos, out_dir, out_file):
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
    for ind in range(len(ind_to_class)):
        category = {"id": ind, "name": ind_to_class[ind], "supercategory": "object", }
        categories.append(category)
    annotations = {"info": info, "images": [], "annotations": [], "categories": categories, "license": license}
    img_names = {}
    IMG_ID = 0
    OBJ_ID = 0
    for info in tqdm(annos):
        name = info['name']
        defect_name = info["defect_name"]
        bbox = info["bbox"]
        if name not in img_names:
            img_names[name] = IMG_ID
            img = cv2.imread(info['dir'])
            cv2.imwrite(out_dir + name, img)
            h, w, _ = img.shape
            img_info = construct_imginfo(name, h, w, IMG_ID)
            annotations["images"].append(img_info)
            IMG_ID = IMG_ID + 1
        img_id = img_names[name]
        cat_ID = class_to_ind[defect_name]
        xmin, ymin, xmax, ymax = bbox
        area = (ymax - ymin) * (xmax - xmin)
        seg = None
        bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
        ann = construct_ann(OBJ_ID, img_id, cat_ID, seg, area, bbox)
        annotations["annotations"].append(ann)
        OBJ_ID += 1
    print(len(annotations["images"]))
    a = open(out_file, 'w')
    a.close()
    mmcv.dump(annotations, out_file)


def generate_data(ind2cls20, coco_file, out_file, choose_list):
    coco = COCO(coco_file)
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
    for ind in range(20):
        category = {"id": ind, "name": ind2cls20[ind], "supercategory": "object", }
        categories.append(category)
    annotations = {"info": info, "images": [], "annotations": [], "categories": categories, "license": license}
    img_names = {}
    imgIds = coco.getImgIds()
    for index in tqdm(imgIds):
        if index not in choose_list:
            continue
        else:
            img = coco.loadImgs(index)[0]
            annotations["images"].append(img)
            annIds = coco.getAnnIds(imgIds=[img['id']], iscrowd=None)
            anns = coco.loadAnns(annIds)
            for ann in anns:
                annotations["annotations"].append(ann)
    print(len(annotations["images"]))
    a = open(out_file, 'w')
    a.close()
    mmcv.dump(annotations, out_file)


def convert20(class_to_ind20, coco35_file, ind2cls20, out_file):
    categories = []
    for ind in range(20):
        category = {"id": ind, "name": ind2cls20[ind], "supercategory": "object", }
        categories.append(category)
    annotations = mmcv.load(coco35_file)
    annotations["categories"] = categories
    for i in range(len(annotations["annotations"])):
        cat_id = annotations["annotations"][i]["category_id"]
        annotations["annotations"][i]['category_id'] = class_to_ind20[ind_to_class[cat_id]]
    print(len(annotations["annotations"]))
    a = open(out_file, 'w')
    a.close()
    mmcv.dump(annotations, out_file)

def generate_test(ind_to_class, test_dir, out_file):
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
    for ind in range(len(ind_to_class)):
        category = {"id": ind, "name": ind_to_class[ind], "supercategory": "object", }
        categories.append(category)
    annotations = {"info": info, "images": [], "annotations": [], "categories": categories, "license": license}

    img_names = {}
    IMG_ID = 0
    annos = os.listdir(test_dir)
    for name in tqdm(annos):
        if name not in img_names:
            img_names[name] = IMG_ID
            img = cv2.imread(test_dir + name)
            h, w, _ = img.shape
            img_info = construct_imginfo(name, h, w, IMG_ID)
            annotations["images"].append(img_info)
            IMG_ID = IMG_ID + 1
        else:
            print("error")
    print(len(annotations["images"]))
    a = open(out_file, 'w')
    a.close()
    mmcv.dump(annotations, out_file)

ind2cls20 = {0: '破洞',
     1: '油渍/水渍/污渍',
     2: '三丝',
     3: '结头',
     4: '花板跳',
     5: '百脚',
     6: '毛粒',
     7: '粗经',
     8: '松经',
     9: '断经',
     10: '吊经',
     11: '粗维',
     12: '纬缩',
     13: '浆斑',
     14: '整经结',
     15: '星跳/跳花',
     16: '断氨纶',
     17: '浪纹档/稀密档/色差档',
     18: '轧痕/磨痕/修痕/烧毛痕',
     19: '死皱/云织/双经/双纬/筘路/跳纱/纬纱不良'}

class_to_ind20 = {'破洞': 0,
                  '油渍': 1, '水渍': 1, '污渍': 1,
                  '三丝': 2,
                  '结头': 3,
                  '花板跳': 4,
                  '百脚': 5,
                  '毛粒': 6,
                  '粗经': 7,
                  '松经': 8,
                  '断经': 9,
                  '吊经': 10,
                  '粗维': 11,
                  '纬缩': 12,
                  '浆斑': 13,
                  '整经结': 14,
                  '星跳': 15, '跳花': 15,
                  '断氨纶': 16,
                  '浪纹档': 17, '稀密档': 17, '色差档': 17,
                  '轧痕': 18, '磨痕': 18, '修痕': 18, '烧毛痕': 18,
                  '死皱': 19, '云织': 19, '双经': 19, '双纬': 19, '筘路': 19, '跳纱': 19, '纬纱不良': 19}
# mmcv.dump(class_to_ind, "cls2ind.pkl")

def moveFiles(dst, src):
    if not os.path.isdir(dst):
        os.makedirs(dst)

    if isinstance(src, str):
        for name in os.listdir(src):
            src_f = os.path.join(src, name)
            dst_f = os.path.join(dst, name)
            shutil.copyfile(src_f, dst_f)
    elif isinstance(src, list) and isinstance(src[0], str):
        for src_ in src:
            for name in os.listdir(src_):
                src_f = os.path.join(src_, name)
                dst_f = os.path.join(dst, name)
                shutil.copyfile(src_f, dst_f)




if __name__ == "__main__":


    data_root = "../data/"
    print("generate normal image dir")
    moveFiles(data_root+'normal_image', [data_root+'guangdong1_round1_train1_20190818/normal_Images',
                                         data_root+'guangdong1_round1_train2_20190828/normal_Images'])
    print('done')
    class_to_ind = mmcv.load("./source/cls2ind.pkl")
    ind_to_class = mmcv.load("./source/ind2cls.pkl")
    a = mmcv.load(data_root + "guangdong1_round1_train1_20190818/"+"Annotations/anno_train.json")
    for i in range(len(a)):
        a[i]["dir"] = data_root + "guangdong1_round1_train1_20190818/" + "defect_Images/" + a[i]['name']
    b = mmcv.load(data_root + "guangdong1_round1_train2_20190828/"+"Annotations/anno_train.json")
    for i in range(len(b)):
        b[i]["dir"] = data_root + "guangdong1_round1_train2_20190828/" + "defect_Images/" + b[i]['name']
    annos = a + b
    data_dir = "../data"
    out_dir = "{}/train/".format(data_dir)
    out_annos_dir = "{}/annotations/".format(data_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(out_annos_dir):
        os.makedirs(out_annos_dir)
    out_file = "{}/annotations/train_coco.json".format(data_dir)
    print("convert to coco fotmat")
    generate_coco(ind_to_class, annos, out_dir, out_file)
    convert20(class_to_ind20, out_file, ind2cls20, out_file)
    train_list = mmcv.load("./source/train_list.pkl")
    val_list = mmcv.load("./source/val_list.pkl")
    print("generate train and val...")
    generate_data(ind2cls20, out_file, out_annos_dir + "train20.json", train_list)
    generate_data(ind2cls20, out_file, out_annos_dir + "val20.json", val_list)
    print("done")
    print("generate test...")
    test_dir = "../data/guangdong1_round1_testB_20190919/"
    out_file = out_annos_dir + "testB.json"
    generate_test(ind_to_class, test_dir, out_file)
    print("done")


