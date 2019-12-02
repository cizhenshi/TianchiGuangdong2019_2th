import mmcv
import os
from tqdm import tqdm
import cv2
from pycocotools.coco import COCO
import shutil
import numpy as np
from icecream import ic
from PrepareData import construct_imginfo


def move_val(val_json, index):
    val_list = []
    print("copy val image to ../data/round2_data/val_{}".format(index))
    coco = COCO(val_json)
    imgIds = coco.getImgIds()
    for imgId in tqdm(imgIds):
        img = coco.loadImgs(imgId)[0]
        img_dir = img['file_name'].split('/')[0]
        source = "../data/round2_data/defect/{}".format(img_dir)
        dest = "../data/round2_data/val_{}/{}".format(index, img_dir)
        shutil.copytree(source, dest)
        val_list.append(img["file_name"].split("/")[-1])
    mmcv.dump(val_list, "../data/round2_data/split/val_{}.pkl".format(index))


def move_normal(normal_list, index, val_json_file):
    coco = COCO(val_json_file)
    ID = max(coco.getImgIds())+1
    val_json = mmcv.load(val_json_file)
    images = val_json["images"]
    print("copy normal image to ../data/round2_data/val_{}".format(index))
    for normal in tqdm(normal_list):
        source = "../data/round2_data/normal/{}".format(normal)
        dest = "../data/round2_data/val_{}/{}".format(index, normal)
        shutil.copytree(source, dest)
        img = cv2.imread(source + "/{}.jpg".format(normal))
        h, w, _ = img.shape
        filename = normal + ".jpg"
        template = normal.split("_")[0] = ".jpg"
        img_info = construct_imginfo(filename, template, h, w, ID)
        ID += 1
        images.append(img_info)
    val_json["images"] = images
    mmcv.dump(val_json, val_json_file)




def split_trainval(coco_file, index):
    coco = COCO(coco_file)
    all = coco.getImgIds()
    val_list = []
    catIds = coco.getCatIds()
    for cat in catIds:
        imgIds = coco.getImgIds(catIds=[cat])
        np.random.seed(index)
        vals = list(np.random.choice(imgIds, int(len(imgIds) * 0.15), replace=False))
        val_list += vals
    val_list = set(val_list)
    train_list = list(set(all) - val_list)
    mmcv.dump(train_list, "../data/round2_data/split/train_id_{}.pkl".format(index))
    mmcv.dump(val_list, "../data/round2_data/split/val_id_{}.pkl".format(index))
    out_file = "../data/round2_data/Annotations/train_{}.json".format(index)
    generate_data(coco, out_file, train_list)
    out_file = "../data/round2_data/Annotations/val_{}.json".format(index)
    generate_data(coco, out_file, val_list)
    # move image to dst
    val_dir = "../data/round2_data/val_{}".format(index)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    else:
        shutil.rmtree(val_dir)
        os.makedirs(val_dir)
    move_val(out_file, index)
    normal_list = os.listdir("../data/round2_data/normal")
    np.random.seed(index)
    normal = np.random.choice(normal_list, 200, replace=False)
    mmcv.dump(normal, "../data/round2_data/split/normal_{}.pkl".format(index))
    move_normal(normal, index, out_file)


def generate_data(coco, out_file, choose_list):
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



if __name__ == "__main__":
    """
    1.generate train and val list
    2.choose normal image
    3.move to commit dir
    """
    data_dir = "../data/round2_data"
    out_file = "{}/Annotations/train_coco.json".format(data_dir)
    for i in range(1):
        split_trainval(out_file, i)
    print("done")

