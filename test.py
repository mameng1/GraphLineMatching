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
from PIL import Image,ImageDraw, ImageOps,ImageFont
from data.scannet_load import ScannetDataset,get_dataloader
import numpy as np
import importlib
import cv2
import os


def rndColor():
    color_list = ["47 79 79",
                  "105 105 105",
                  "119 136 153",
                  "190 190 190",
                  "25 25 112",
                  "100 149 237",
                  "72 61 139",
                  "106 90 205",
                  "0 0 205",
                  "65 105 225",
                  "30 144 255",
                  "0 191 255",
                  "135 206 235",
                  "72 209 204",
                  "70 130 180",
                  "135 206 250",
                  "0 206 209",
                  "224 255 255",
                  "46 139 87",
                  "0 100 0",
                  "85 107 47",
                  "143 188 143",
                  "152 251 152",
                  "0 250 154",
                  "50 205 50",
                  "255 255 0",
                  "139 69 19",
                  "144 238 144",
                  "187 255 255",
                  "150 205 205",
                  "102 139 139",
                  "0 245 255",
                  "0 205 205",
                  "127 255 212",
                  "102 205 170",
                  "69 139 116",
                  "84 255 159",
                  "67 205 128",
                  "0 238 118",
                  "192 255 62",
                  "154 205 50",
                  "255 246 143",
                  "205 198 115",
                  "205 198 115",
                  "139 139 122",
                  "255 255 0",
                  "205 205 0",
                  "139 139 0",
                  "255 215 0",
                  "205 173 0",
                  "139 90 0",
                  "224 102 255",
                  "122 55 139",
                  "104 34 139",
                  "144 238 144",
                  "0 139 139"]

    select_color = random.sample(color_list, 1)
    select_color = select_color[0].split(" ")
    select_color = [int(item) for item in select_color]
    return (select_color[0], select_color[1], select_color[2])

def mergeImg(limg,rimg,edge_w=50):
    width,height = limg.size
    result = Image.new(limg.mode,(width*2+edge_w,height),color=(255,255,255))
    #result = Image.new(limg.mode, (width, height* 2 + edge_w),color=(255,255,255))
    result.paste(limg, box=(0, 0))
    result.paste(rimg, box=(width+edge_w, 0))
    #result.paste(rimg, box=(0, height+edge_w))
    return result
def pred_vis(limg_path,rimg_path,lline_path,rline_path,left_idx,right_idx,output_path,match,gt_num,pred,orig_num,save_name,res_anme,scale):
    match_sum=np.sum(match)
    gt_sum=np.sum(gt_num)
    pred_sum=np.sum(pred)

    left_image=Image.open(limg_path)
    left_image=left_image.convert("RGB")
    right_image=Image.open(rimg_path)
    right_image=right_image.convert("RGB")

    old_size = (left_image.width, left_image.height)
    new_size = tuple([int(x * scale) for x in old_size])
    left_image = left_image.resize(new_size, resample=Image.BICUBIC)
    right_image = right_image.resize(new_size, resample=Image.BICUBIC)

    left_lines,left_label_lines=readLines(lline_path,(scale,0,0))
    right_lines,right_label_lines=readLines(rline_path,(scale,0,0))

    #textsize = 20
    #ft = ImageFont.truetype(size=24)

    """ 
    draw_left = cv2.imread(limg_path)
    draw_right = cv2.imread(rimg_path)
    for left_id in left_idx.tolist():
        left_line = left_lines[left_id]
        left_line = [int(item) for item in left_line]

        left_coord = ((left_line[0] + left_line[2]) // 2,
                      (left_line[1] + left_line[3]) // 2)
        color=rndColor()
        cv2.line(draw_left, (left_line[0], left_line[1]),
                 (left_line[2], left_line[3]), color, thickness=3)
        cv2.putText(draw_left, "{}".format(left_id), left_coord, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    for right_id in right_idx.tolist():
        right_line = right_lines[right_id]
        right_line = [int(item) for item in right_line]

        color = rndColor()
        right_coord = ((right_line[0] + right_line[2]) // 2,
                       (right_line[1] + right_line[3]) // 2)

        cv2.line(draw_right, (right_line[0], right_line[1]),
                 (right_line[2], right_line[3]), color, thickness=3)

        cv2.putText(draw_right, "{}".format(right_id), right_coord, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    left_image = Image.fromarray(cv2.cvtColor(draw_left, cv2.COLOR_BGR2RGB))
    right_image = Image.fromarray(cv2.cvtColor(draw_right, cv2.COLOR_BGR2RGB))

    res=mergeImg(left_image,right_image,50)
    res_draw=cv2.cvtColor(np.asarray(res),cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_name, res_draw)
    return 0
    """
    inter_height=20
    res = mergeImg(left_image, right_image, inter_height)
    width,height=left_image.size
    res_draw=ImageDraw.Draw(res)#cv2.cvtColor(np.asarray(res),cv2.COLOR_RGB2BGR)
    font=ImageFont.truetype("Times_New_Roman.ttf", 25)
    unlabel_num=0
    for left_id,right_id in zip(left_idx,right_idx):
        left_line=left_lines[left_id]
        right_line=right_lines[right_id]

        left_label=left_label_lines[left_id][0]
        right_label=right_label_lines[right_id][0]

        #if(left_label=="-1" or right_label=="-1"):
        unlabel_num=unlabel_num+1

        right_line[0]=right_line[0]+inter_height+width
        right_line[2] = right_line[2] + inter_height + width
        #right_line[1]=right_line[1]+inter_height+height
        #right_line[3] = right_line[3] + inter_height + height
        if(unlabel_num==35):
            debug=0
        if(gt_num[left_id,right_id]==1):
            label_tex=str(unlabel_num)#+" "+str(right_id)
            color=rndColor()
        else:
            label_tex=str(unlabel_num)#+" "+str(right_id)
            print("{} {} {}".format(unlabel_num,left_id,right_id))
            color=(255,0,0)
        left_line=[int(item) for item in left_line]
        right_line=[int(item) for item in right_line]
        #color=rndColor()
        left_coord = ((left_line[0] + left_line[2]) // 2,
                      (left_line[1] + left_line[3]) // 2)
        right_coord = ((right_line[0] + right_line[2]) // 2,
                       (right_line[1] + right_line[3]) // 2)
        #if(gt_num[left_id,right_id]==1):
        #    cv2.line(res_draw,left_coord,right_coord,(255,0,0),thickness=8)
        #else:
        #    cv2.line(res_draw,left_coord,right_coord,(0,0,255),thickness=8)
        res_draw.line((left_line[0], left_line[1],
                 left_line[2], left_line[3]), fill=color,width=6)
        #cv2.line(res_draw, (left_line[0], left_line[1]),
        #         (left_line[2], left_line[3]), color, thickness=8)
        res_draw.line((right_line[0], right_line[1],
                 right_line[2], right_line[3]), fill=color,width=6)

        #cv2.line(res_draw, (right_line[0], right_line[1]),
        #         (right_line[2], right_line[3]), color, thickness=8)
        res_draw.text(left_coord,label_tex,color,font)
        res_draw.text(right_coord, label_tex, color,font)

        #cv2.putText(res_draw, label_tex, left_coord, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        #cv2.putText(res_draw, label_tex, right_coord, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


        #cv2.putText(res_draw, "m:{}/gt:{}".format(match_sum,gt_sum), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        #cv2.putText(res_draw, "m:{}/pred:{}".format(match_sum,pred_sum), (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        #cv2.putText(res_draw, "opred{}".format(orig_num), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    font = ImageFont.truetype("Times_New_Roman.ttf", 30)
    fontscale=1.5
    interval=30
    precision = (match_sum / pred_sum)
    recall = match_sum / gt_sum
    F1 = 2 * precision * recall / (precision + recall)
    color = (255, 0, 0)
    """
    res_draw.text((5, 5), "Our", color,font)
    res_draw.text((5, 5+0*interval), "Detected lines: ({},{})".format(len(left_lines), len(right_lines)), color,font)
    res_draw.text((5, 5+1*interval), "Total matches: {}".format(pred_sum), color,font)
    res_draw.text((5, 5+2*interval), "Correct matches: {}".format(match_sum), color,font)
    res_draw.text((5, 5+3*interval), "Ground truth matches: {}".format(gt_sum), color,font)
    res_draw.text((5, 5+5*interval), "Precision: {}%".format(round(100*match_sum/pred_sum,2)),color,font)
    res_draw.text((5, 5+6*interval), "Recall: {}%".format(round(100*match_sum / gt_sum,2)), color,font)
    res_draw.text((5, 5 + 7 * interval), "F-Measure: {}%".format(round(100 *F1, 2)), color, font)
    """
    """
    cv2.putText(res_draw, "Our", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, fontscale,(255, 255, 255), 2)
    cv2.putText(res_draw, "Detected lines: ({},{})".format(len(left_lines), len(right_lines)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, fontscale,(255, 255, 255), 2)
    cv2.putText(res_draw, "Total matches: {}".format(pred_sum), (10, 20+interval), cv2.FONT_HERSHEY_COMPLEX, fontscale,(255, 255, 255), 2)
    cv2.putText(res_draw, "Correct matches: {}".format(match_sum), (10, 20+2*interval), cv2.FONT_HERSHEY_COMPLEX, fontscale,
                (255, 255, 255), 2)
    cv2.putText(res_draw, "Ground truth matches: {}".format(gt_sum), (10, 20+3*interval), cv2.FONT_HERSHEY_COMPLEX, fontscale,
                (255, 255, 255), 2)
    cv2.putText(res_draw, "Accuracy: {}".format(match_sum/pred_sum), (10, 20+4*interval), cv2.FONT_HERSHEY_COMPLEX, fontscale,
                (255, 255, 255), 2)
    cv2.putText(res_draw, "Recall: {}".format(match_sum / gt_sum), (10, 20+5*interval), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontscale,
                (255, 255, 255), 2)
    """
    res_f=open(res_anme,"w")
    res_f.write("{} {} {} {}\n".format(match_sum,gt_sum,pred_sum,orig_num))
    res_f.close()
    #cv2.putText(res_draw, "{}".format(unlabel_num), (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    res.save(save_name)
    #cv2.imwrite(save_name,res_draw)
    #cv2.imshow("fr",res_draw)
    #cv2.waitKey(0)
    #res.save(output_path+"res.png")


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
        score_thresh=0.3
        s_pred, _,_,_,_,_,indeces1, indeces2, newn1_gt, newn2_gt = \
            model(l_img, r_img, l_boxes, r_boxes, l_pts, r_pts,perm_mat=None,train_stage=False,
                  score_thresh=score_thresh)
        s_pred_perm = lap_solver(s_pred, newn1_gt, newn2_gt, indeces1, indeces2, l_pts, r_pts)
        #s_pred_perm = lap_solver(s_pred, None, l_pts, r_pts)
    row, col = np.where(s_pred_perm.cpu().numpy()[0] == 1)
    return row,col
def image2tensor(img_dir,obj_resize):
    with Image.open(img_dir) as img:
        img=img.convert("RGB")
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
        all_keypts = [i.strip().split(" ") for i in all_keypts]
        for keypoint in all_keypts:
            ckeypoint=keypoint[1:]
            valid_key = []
            for citem in ckeypoint:
                if (len(citem) != 0):
                    valid_key.append(citem)
            ckeypoint = valid_key
            ckeypoint[0] = float(ckeypoint[0]) * ratio + delta_w // 2
            ckeypoint[2] = float(ckeypoint[2]) * ratio + delta_w // 2
            ckeypoint[1] = float(ckeypoint[1]) * ratio + delta_h // 2
            ckeypoint[3] = float(ckeypoint[3]) * ratio + delta_h // 2
            keypoint_list.append(ckeypoint)
    return keypoint_list,all_keypts
def computeGTweight(left_lines,right_lines):
    row_list=[]
    col_list=[]
    perm_mat=np.zeros((len(left_lines),len(right_lines)),dtype=np.int)
    for i, keypoint in enumerate(left_lines):
        for j, _keypoint in enumerate(right_lines):
            if keypoint[0] == _keypoint[0] and keypoint[0] != "-1":
                perm_mat[i, j] = 1
                row_list.append(i)
                col_list.append(j)

    valid_rowsid, valid_colsid = [], []
    for i, keypoint in enumerate(left_lines):
        if (keypoint[0] != "-1"):
            valid_rowsid.append(i)
    for j, _keypoint in enumerate(right_lines):
        if (_keypoint[0] != "-1"):
            valid_colsid.append(j)
    weight_mat = np.zeros_like(perm_mat)
    for r_idx in valid_rowsid:
        weight_mat[r_idx, :] = 1
    for c_idx in valid_colsid:
        weight_mat[:, c_idx] = 1

    return perm_mat, weight_mat
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

    mod = importlib.import_module('LSM.model')
    Net = mod.Net


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net(cfg.OUTPUT_SIZE, cfg.SCALES)
    model = model.to(device)

    obj_resize = (800, 800)
    #val_im_path="/media/mameng/18765DA00374155D/KITTI_RURoC_label/uval_img.txt"#"/home/mameng/dataset/scannet/datadir_line09nms94/uval_img.txt"
    #val_label_path="/media/mameng/18765DA00374155D/KITTI_RURoC_label/uval_label.txt"#"/home/mameng/dataset/scannet/datadir_line09nms94/uval_label.txt"
    #save_dir="/home/mameng/dataset/KITTI/vis/our/img"#"/home/mameng/dataset/scannet/vis/scannet/end/our/img"
    #ressave_dir ="/home/mameng/dataset/KITTI/vis/our/img/res" #"/home/mameng/dataset/scannet/vis/scannet/end/our/res"
    #val_im_path = "/home/mameng/dataset/scannet/datadir_line09nms94/uval_img.txt"
    #val_label_path = "/home/mameng/dataset/scannet/datadir_line09nms94/uval_label.txt"
    #save_dir = "/home/mameng/dataset/scannet/vis/scannet/end/our/img"
    #ressave_dir = "/home/mameng/dataset/scannet/vis/scannet/end/our/res"
    # using_path="/home/mameng/dataset/scannet/vis/scannet/using.txt"
    #using_path="/home/mameng/dataset/scannet/vis/scannet/using.txt"
    #use_f=open(using_path,"r")
    #use_indeces=use_f.readline().strip().split(" ")
    use_indeces=[5]#[int(idx) for idx in use_indeces]
    #use_f.close()
    val_im_path="/media/mameng/18765DA00374155D/KITTI_RURoC_label/eurouuval_imgb.txt"#"/home/mameng/dataset/scannet/datadir_line09nms94/uval_img.txt"
    val_label_path="/media/mameng/18765DA00374155D/KITTI_RURoC_label/eurouuval_labelb.txt"#"/home/mameng/dataset/scannet/datadir_line09nms94/uval_label.txt"
    save_dir="/media/mameng/18765DA00374155D/vis/our/img"#"/home/mameng/dataset/scannet/vis/scannet/end/our/img"
    ressave_dir ="/media/mameng/18765DA00374155D/vis/our/img/res" #"/home/mameng/dataset/scannet/vis/scannet/end/our/res"
    if(not os.path.exists(save_dir)):
        os.makedirs(save_dir)
    if(not os.path.exists(ressave_dir)):
        os.makedirs(ressave_dir)
    val_imf=open(val_im_path,"r")
    val_labelf=open(val_label_path,"r")

    scale=1

    val_ims=val_imf.readlines()
    val_labels=val_labelf.readlines()
    val_ims=[item.strip() for item in val_ims]
    val_labels = [item.strip() for item in val_labels]
    name_idx=-1
    for pair_im,pair_label in zip(val_ims,val_labels):
        name_idx = name_idx + 1
        if(name_idx not in use_indeces):
            continue
        save_name= os.path.join(save_dir,"{}.png".format(name_idx))
        res_anme= os.path.join(ressave_dir,"{}.txt".format(name_idx))

        l_imp,r_imp=pair_im.split(" ")
        llines_p,rlines_p=pair_label.split(" ")
        l_imp = '/media/mameng/18765DA00374155D/euroc_vis/001305.png'
        r_imp = '/media/mameng/18765DA00374155D/euroc_vis/001310.png'
        llines_p = '/media/mameng/18765DA00374155D/euroc_vis/001305.txt'
        rlines_p = '/media/mameng/18765DA00374155D/euroc_vis/001310.txt'

        #l_imp = "/home/mameng/dataset/scannet/vis/scene0628_02/000000.jpg"
        #r_imp = "/home/mameng/dataset/scannet/vis/scene0628_02/001000.jpg"
        #llines_p = '/home/mameng/dataset/scannet/vis/scene0628_02/000000_001000_000000.txt'
        #rlines_p = '/home/mameng/dataset/scannet/vis/scene0628_02/000000_001000_001000.txt'

        #l_imp = "/home/mameng/dataset/scannet/vis/scene0610_00/000100.jpg"
        #r_imp = "/home/mameng/dataset/scannet/vis/scene0610_00/001200.jpg"
        #llines_p = '/home/mameng/dataset/scannet/vis/scene0610_00/000100_001200_000100.txt'
        #rlines_p = '/home/mameng/dataset/scannet/vis/scene0610_00/000100_001200_001200.txt'

        #l_imp = "/home/mameng/dataset/scannet/vis/scene0653_00/000700.jpg"
        #r_imp = "/home/mameng/dataset/scannet/vis/scene0653_00/000800.jpg"
        #llines_p = '/home/mameng/dataset/scannet/vis/scene0653_00/000700_000800_000700.txt'
        #rlines_p = '/home/mameng/dataset/scannet/vis/scene0653_00/000700_000800_000800.txt'

        #l_imp = "/home/mameng/dataset/scannet/vis/scene0592_00/000400.jpg"
        #r_imp = "/home/mameng/dataset/scannet/vis/scene0592_00/001900.jpg"
        #llines_p = '/home/mameng/dataset/scannet/vis/scene0592_00/000400_001900_000400.txt'
        #rlines_p = '/home/mameng/dataset/scannet/vis/scene0592_00/000400_001900_001900.txt'

        #l_imp = "/home/mameng/dataset/scannet/vis/scene0591_01/000000.jpg"
        #r_imp = "/home/mameng/dataset/scannet/vis/scene0591_01/001200.jpg"
        #llines_p = '/home/mameng/dataset/scannet/vis/scene0591_01/000000_001200_000000.txt'
        #rlines_p = '/home/mameng/dataset/scannet/vis/scene0591_01/000000_001200_001200.txt'

        #l_imp = "/home/mameng/dataset/scannet/vis/scene0670_00/001800.jpg"
        #r_imp = "/home/mameng/dataset/scannet/vis/scene0670_00/rot_001900.jpg"
        #llines_p = '/home/mameng/dataset/scannet/vis/scene0670_00/001800_001900_001800.txt'
        #rlines_p = '/home/mameng/dataset/scannet/vis/scene0670_00/rot_001800_001900_001900.txt'

        #l_imp = "/home/mameng/dataset/scannet/vis/scene0596_00/000100.jpg"
        #r_imp = "/home/mameng/dataset/scannet/vis/scene0596_00/001000.jpg"
        #llines_p = '/home/mameng/dataset/scannet/vis/scene0596_00/000100_001000_000100.txt'
        #rlines_p = '/home/mameng/dataset/scannet/vis/scene0596_00/000100_001000_001000.txt'

        #l_imp = "/home/mameng/dataset/scannet/vis/scene0643_00/000000.jpg"
        #r_imp = "/home/mameng/dataset/scannet/vis/scene0643_00/blur_001200.jpg"
        #llines_p = '/home/mameng/dataset/scannet/vis/scene0643_00/000000_001200_000000.txt'
        #rlines_p = '/home/mameng/dataset/scannet/vis/scene0643_00/000000_001200_001200.txt'

        #l_imp = "/home/mameng/dataset/scannet/vis/scene0653_00/000700.jpg"
        #r_imp = "/home/mameng/dataset/scannet/vis/scene0653_00/000800.jpg"
        #llines_p = "/home/mameng/dataset/scannet/vis/scene0653_00/000700_000800_000700.txt"
        #rlines_p = "/home/mameng/dataset/scannet/vis/scene0653_00/000700_000800_000800.txt"

        args.left_img=l_imp
        args.right_img=r_imp
        args.left_lines=llines_p
        args.right_lines=rlines_p
        left_img, left_param = image2tensor(args.left_img, obj_resize)
        right_img, right_param = image2tensor(args.right_img, obj_resize)

        left_lines,orig_left_lines = readLines(args.left_lines, left_param)
        right_lines,orig_right_lines = readLines(args.right_lines, right_param)
        gt_mat,weights=computeGTweight(orig_left_lines,orig_right_lines)

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
        pred_mat=np.zeros_like(gt_mat)
        for row_idx,col_idx in zip(left_id,right_id):
            pred_mat[row_idx,col_idx]=1

        orig_num=len(left_id)
        match = pred_mat * gt_mat #* weights
        gt_num = gt_mat #* weights
        pred= pred_mat #* weights

        #de=np.sum(gt_mat, axis=0)
        gt_zero_c = np.where(np.sum(gt_mat, axis=0) == 0)
        gt_zero_r = np.where(np.sum(gt_mat, axis=1) == 0)
        new_row, new_col = np.where(pred_mat == 1)
        left_id=new_row#gt_zero_r[0]#new_row
        right_id=new_col#gt_zero_c[0]#new_col
        #r_imp="/home/mameng/dataset/scannet/vis/scene0653_00/light_000800.jpg"
        #args.right_img=r_imp
        pred_vis(args.left_img,args.right_img,args.left_lines,
                 args.right_lines,left_id,right_id,args.output_path,match,gt_num,pred,orig_num,save_name,res_anme,scale)
