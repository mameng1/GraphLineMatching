import torch
import time
from datetime import datetime
from pathlib import Path
import argparse
from utils.parse_args import test_parse_args
import math
import random
from utils.hungarian import hungarian
from utils.evaluation_metric import matching_accuracy
from parallel import DataParallel
from utils.model_sl import load_model
from torchvision import transforms
from utils.config import cfg
from PIL import Image,ImageDraw, ImageOps
from data.scannet_load import ScannetDataset,get_dataloader
import numpy as np
import importlib
def mergeImg(limg,rimg,edge_w=50):
    width,height = limg.size
    result = Image.new(limg.mode,(width*2+edge_w,height))
    result.paste(limg, box=(0, 0))
    result.paste(rimg, box=(width+edge_w, 0))
    return result
def pred_vis(limg_path,rimg_path,lline_path,rline_path,left_idx,right_idx,output_path):
    left_image=Image.open(limg_path)
    right_image=Image.open(rimg_path)

    left_lines=readLines(lline_path,(1,0,0))
    right_lines=readLines(rline_path,(1,0,0))
    """
    draw_left = ImageDraw.Draw(left_image)
    draw_right = ImageDraw.Draw(right_image)
    for line in left_lines:
        draw_left.line((line[0], line[1], line[2], line[3]), fill=(255, 255, 255), width=5)
    for line in right_lines:
        draw_right.line((line[0], line[1], line[2], line[3]), fill=(255, 255, 255), width=5)
    """
    res=mergeImg(left_image,right_image,50)
    width,height=left_image.size
    res_draw=ImageDraw.Draw(res)

    def rndColor():
        return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))

    for left_id,right_id in zip(left_idx,right_idx):
        left_line=left_lines[left_id]
        right_line=right_lines[right_id]
        right_line[0]=right_line[0]+50+width
        right_line[2] = right_line[2] + 50 + width
        color=rndColor()
        res_draw.line((left_line[0], left_line[1], left_line[2], left_line[3]), fill=color, width=8)
        res_draw.line((right_line[0], right_line[1], right_line[2], right_line[3]), fill=color, width=8)
        # cv2.putText(limg,"{}".format(lidx),(lcx,lcy),cv2.FONT_HERSHEY_SIMPLEX,0.9,color,2)
        # cv2.putText(rimg, "{}".format(ridx), (rcx, rcy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    #res.show()
    res.save(output_path+"res.png")


def eval_model(l_img,r_img,l_boxes,r_boxes,l_pts,r_pts,model,model_path):

    load_model(model, model_path)
    model.eval()

    lap_solver = hungarian
    l_img=l_img.cuda()
    r_img=r_img.cuda()
    l_boxes=l_boxes.cuda()
    r_boxes=r_boxes.cuda()
    l_pts=l_pts.cuda()
    r_pts=r_pts.cuda()
    with torch.set_grad_enabled(False):
        score_thresh=0.6
        s_pred, indeces1, indeces2, newn1_gt, newn2_gt = \
            model(l_img, r_img, l_boxes, r_boxes, l_pts, r_pts, train_stage=False, perm_mat=None,
                  score_thresh=score_thresh)

        s_pred_perm = lap_solver(s_pred, newn1_gt, newn2_gt, indeces1, indeces2, l_pts, r_pts)
        #s_pred_perm = lap_solver(s_pred, None, l_pts, r_pts)
    row, col = np.where(s_pred_perm.cpu().numpy()[0] == 1)
    return row,col
def image2tensor(img_dir,obj_resize):
    with Image.open(img_dir) as img:
        height = img.height
        width = img.width
        old_size = (width, height)
        ratio = float(obj_resize[0]) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        obj = img.resize(new_size, resample=Image.BICUBIC)
        delta_w = obj_resize[0] - new_size[0]
        delta_h = obj_resize[1] - new_size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        obj = ImageOps.expand(obj, padding)
        return obj,(ratio,delta_w,delta_h)
def readLines(lines_path,params):
    ratio = params[0]
    delta_w = params[1]
    delta_h = params[2]
    keypoint_list = []
    with open(lines_path, "r") as anno:
        all_keypts = anno.readlines()
        all_keypts = [i.split(" ") for i in all_keypts]
        for keypoint in all_keypts:
            keypoint[0] = float(keypoint[0]) * ratio + delta_w // 2
            keypoint[2] = float(keypoint[2]) * ratio + delta_w // 2
            keypoint[1] = float(keypoint[1]) * ratio + delta_h // 2
            keypoint[3] = float(keypoint[3]) * ratio + delta_h // 2
            keypoint_list.append(keypoint)
    return keypoint_list
def pts_to_boxes(pt_gt,num):
    boxes = []
    for idx in range(num):
        pts = pt_gt[idx]
        pt1 = (pts[0], pts[1])
        pt2 = (pts[2], pts[3])
        # print pt2[0], pt3[0]
        angle = 0
        if (pt1[0] - pt2[0]) != 0:
            angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
        else:
            angle = 90.0

        if angle < -45.0:
            angle = angle + 180

        x_ctr = float(pt1[0] + pt2[0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
        y_ctr = float(pt1[1] + pt2[1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2
        width = math.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
        boxes.append([x_ctr, y_ctr,cfg.EXPAND_REGION,width,angle])
    boxes=np.array(boxes)
    return boxes
if __name__ == '__main__':

    args = test_parse_args('test')

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.NORM_MEANS, cfg.NORM_STD)
    ])

    mod = importlib.import_module('LM.model')
    Net = mod.Net


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net(cfg.OUTPUT_SIZE, cfg.SCALES)
    model = model.to(device)

    obj_resize = (800, 800)
    left_img, left_param = image2tensor(args.left_img, obj_resize)
    right_img, right_param = image2tensor(args.right_img, obj_resize)

    left_lines = readLines(args.left_lines, left_param)
    right_lines = readLines(args.right_lines, right_param)
    n1_pts = len(left_lines)
    n2_pts = len(right_lines)

    left_boxes = pts_to_boxes(left_lines, n1_pts)
    right_boxes = pts_to_boxes(right_lines, n2_pts)

    left_boxes = torch.Tensor(left_boxes).unsqueeze(0)
    right_boxes = torch.Tensor(right_boxes).unsqueeze(0)
    left_normimg = trans(left_img).unsqueeze(0)
    right_normimg = trans(right_img).unsqueeze(0)

    n1_pts = torch.tensor(n1_pts).unsqueeze(0)
    n2_pts = torch.tensor(n2_pts).unsqueeze(0)

    left_id,right_id=eval_model(left_normimg,right_normimg,left_boxes,right_boxes,n1_pts,n2_pts,model,args.model_path)
    pred_vis(args.left_img,args.right_img,args.left_lines,
             args.right_lines,left_id,right_id,args.output_path)