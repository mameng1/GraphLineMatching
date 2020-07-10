import torch
import time
from datetime import datetime
from pathlib import Path

from utils.hungarian import hungarian
from data.data_loader import GMDataset, get_dataloader
from utils.evaluation_metric import matching_accuracy
from parallel import DataParallel
from utils.model_sl import load_model
from torchvision import transforms
from utils.config import cfg
from PIL import Image
import numpy as np
import cv2
def readLines(lines_path,params):
    ratio = params[0]
    delta_w = params[1]
    delta_h = params[2]
    keypoint_list = []
    with open(lines_path, "r") as anno:
        all_keypts = anno.readlines()
        all_keypts = [i.strip().split(" ") for i in all_keypts]
        for keypoint in all_keypts:
            ckeypoint=keypoint[1:]
            ckeypoint[0] = float(ckeypoint[0]) * ratio + delta_w // 2
            ckeypoint[2] = float(ckeypoint[2]) * ratio + delta_w // 2
            ckeypoint[1] = float(ckeypoint[1]) * ratio + delta_h // 2
            ckeypoint[3] = float(ckeypoint[3]) * ratio + delta_h // 2
            keypoint_list.append(ckeypoint)
    return keypoint_list,all_keypts
def mergeImg(limg,rimg,edge_w=50):
    width,height = limg.size
    result = Image.new(limg.mode,(width*2+edge_w,height))
    result.paste(limg, box=(0, 0))
    result.paste(rimg, box=(width+edge_w, 0))
    return result
import os
import random
using_path="/home/mameng/dataset/scannet/vis/scannet/using.txt"
use_f=open(using_path,"r")
use_indeces=use_f.readline().strip().split(" ")
use_indeces=[int(idx) for idx in use_indeces]
use_f.close()

val_im_path = "/home/mameng/dataset/scannet/datadir_line09nms94/uval_img.txt"
val_label_path = "/home/mameng/dataset/scannet/datadir_line09nms94/uval_label.txt"
save_dir = "/home/mameng/dataset/scannet/vis/scannet/end/our/img"
using_path = "/home/mameng/dataset/scannet/vis/scannet/using.txt"
use_f = open(using_path, "r")
use_indeces = use_f.readline().strip().split(" ")
use_indeces = [int(idx) for idx in use_indeces]
use_f.close()

if (not os.path.exists(save_dir)):
    os.makedirs(save_dir)
val_imf = open(val_im_path, "r")
val_labelf = open(val_label_path, "r")


def rndColor():
    return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))

val_ims = val_imf.readlines()
val_labels = val_labelf.readlines()
val_ims = [item.strip() for item in val_ims]
val_labels = [item.strip() for item in val_labels]
for name_idx in use_indeces:
    pair_im=val_ims[name_idx]
    pair_label = val_labels[name_idx]
    save_name = os.path.join(save_dir, "{}.png".format(name_idx))

    lim_p, r_imp = pair_im.split(" ")
    llines_p, rlines_p = pair_label.split(" ")

    left_lines, orig_left_lines = readLines(llines_p, (1,0,0))
    right_lines, orig_right_lines = readLines(rlines_p, (1,0,0))

    left_image = Image.open(lim_p)
    left_image = left_image.convert("RGB")
    right_image = Image.open(r_imp)
    right_image = right_image.convert("RGB")

    res = mergeImg(left_image, right_image, 50)
    width, height = left_image.size
    res_draw = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2BGR)
    using_name=0
    for left_idx in range(len(left_lines)):
        left_line=left_lines[left_idx]
        left_org_line=orig_left_lines[left_idx]
        if(left_org_line[0]!="-1"):
            continue
        using_name = using_name + 1
        left_line = [int(item) for item in left_line]
        left_coord = ((left_line[0] + left_line[2]) // 2,
                      (left_line[1] + left_line[3]) // 2)

        # if(gt_num[left_id,right_id]==1):
        #    cv2.line(res_draw,left_coord,right_coord,(255,0,0),thickness=8)
        # else:
        #    cv2.line(res_draw,left_coord,right_coord,(0,0,255),thickness=8)
        color=rndColor()
        cv2.line(res_draw, (left_line[0], left_line[1]),
                 (left_line[2], left_line[3]), color, thickness=8)

        cv2.putText(res_draw, "{}".format(left_idx), left_coord, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(res_draw, "{}".format(left_idx), (left_line[0], left_line[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(res_draw, "{}".format(left_idx), (left_line[2], left_line[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


    cv2.putText(res_draw, "{}".format(using_name), (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    right_use_name=0
    for right_idx in range(len(right_lines)):
        right_line=right_lines[right_idx]
        right_org_line = orig_right_lines[right_idx]

        if (right_org_line[0] != "-1"):
            continue

        right_use_name=right_use_name+1
        right_line[0]=right_line[0]+50+width
        right_line[2] = right_line[2] + 50 + width

        right_line=[int(item) for item in right_line]
        color=rndColor()

        right_coord = ((right_line[0] + right_line[2]) // 2,
                       (right_line[1] + right_line[3]) // 2)

        cv2.line(res_draw, (right_line[0], right_line[1]),
                 (right_line[2], right_line[3]), color, thickness=8)

        cv2.putText(res_draw, "{}".format(right_idx), right_coord, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(res_draw, "{}".format(right_idx), (right_line[0], right_line[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(res_draw, "{}".format(right_idx), (right_line[2], right_line[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.putText(res_draw, "{}".format(right_use_name), (width+500, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imwrite(save_name, res_draw)