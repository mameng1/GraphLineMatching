import sys
import os
import pickle
#import Augmentor.Operations as op
#import imgaug as ia
import imgaug.augmenters as iaa
import imageio
import shutil
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from multiprocessing import Pool
import math
import numpy as np
# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.


seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
       # iaa.Fliplr(0.6), # horizontally flip 50% of all images
       # iaa.Flipud(0.2), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        #sometimes(iaa.CropAndPad(
        #    percent=(-0.05, 0.1))),

        iaa.Affine(
            scale=(0.6, 2), # scale images to 80-120% of their size, individually per axis
            #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-20, 20), # rotate by -45 to +45 degrees
            #shear=(-16, 16), # shear by -16 to +16 degrees
            #order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
           # cval=(0, 255), # if mode is constant, use a cval between 0 and 255
           # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        ),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((1, 2),
            [
                #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 5)), # blur image using local means with kernel sizes between 2 and 7
                    #iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                #iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                #iaa.SimplexNoiseAlpha(iaa.OneOf([
                #    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                #    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
               # ])),
                #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
               # iaa.OneOf([
               #     iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
               #     iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
               # ]),
                #iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-40, 40), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                #iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                #iaa.OneOf([
                #    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                #    iaa.FrequencyNoiseAlpha(
                #        exponent=(-4, 0),
                #        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                #        second=iaa.LinearContrast((0.5, 2.0))
                #    )
                #]),
                #iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                #iaa.Grayscale(alpha=(0.0, 1.0)),
                #sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                #sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                #iaa.PerspectiveTransform(scale=(0.1, 1))
            ],
            random_order=True
        )
    ],
    random_order=True
)
def load(filename):
    with open(filename, "rb") as file:
        output = pickle.load(file, encoding='latin1')
        return output['_line_segs']
def saveLineString(save_name,lsoi):
    save_file=open(save_name,"w")
    linesstr = lsoi.line_strings
    for line_l in linesstr:
        label = line_l.label
        coords = line_l.coords
        co_shape=coords.shape
        if(co_shape[0] !=2 or co_shape[1] !=2):
            continue
        coord1=coords[0]
        coord2=coords[1]
        save_str="{} {} {} {} {}".format(label,coord1[0],coord1[1],coord2[0],coord2[1])
        save_file.write(save_str+"\n")
    save_file.close()
def load_line(line_path):
    lines_list=[]
    with open(line_path,"r") as f:
        lines_list=f.readlines()
        lines_list=[item.strip() for item in lines_list]
        lines_list=[item.split(" ") for item in lines_list]
        lines_list=[[float(ii) for ii in item]for item in lines_list]
    return lines_list

train_file="/home/mameng/dataset/scannet/datadir_line09nms94/train_img.txt"
label_file="/home/mameng/dataset/scannet/datadir_line09nms94/train_label.txt"
img_savedir="/home/mameng/dataset/scannet/scannet_aug/train/image"
label_savedir="/home/mameng/dataset/scannet/scannet_aug/train/label"
"""
def dealEachFile(line_path):
    ppath=os.path.dirname(line_path)
    pppath=os.path.dirname(ppath)
    basename=os.path.basename(line_path).split(".")[0]
    img_path=os.path.join(pppath,"color")
    img_path=os.path.join(img_path,basename+".jpg")
    scene_name=os.path.basename(pppath)
    img_savesubdir = os.path.join(img_savedir, scene_name,basename)
    label_savesubdir = os.path.join(label_savedir, scene_name,basename)
    if (not os.path.exists(img_savesubdir)):
        os.makedirs(img_savesubdir)
    if (not os.path.exists(label_savesubdir)):
        os.makedirs(label_savesubdir)

    save_imgname = basename+".jpg"
    save_labelname = basename+".txt"

    image = imageio.imread(img_path)
    lines_vec = []

    lines_tem = load_line(line_path)
    lines_svec = []
    for idx, line in enumerate(lines_tem):
        line_pts = []
        p1 = (line[0], line[1])
        p2 = (line[2], line[3])

        line_pts.append(p1)
        line_pts.append(p2)
        line_s = LineString(line_pts, label="{}".format(idx))
        lines_svec.append(line_s)

    lsoi = LineStringsOnImage(lines_svec, shape=image.shape)

    imageio.imwrite(os.path.join(img_savesubdir, save_imgname), image)
    saveLineString(os.path.join(label_savesubdir, save_labelname), lsoi)
    for t in range(10):
        save_aug_imgname = save_imgname.split(".")[0] + "_" + str(t) + ".jpg"
        save_aug_labelname = save_labelname.split(".")[0] + "_" + str(t) + ".txt"
        image_aug, lsoi_aug = seq(image=image, line_strings=lsoi)
        lsoi_aug = lsoi_aug.remove_out_of_image()
        lsoi_aug = lsoi_aug.clip_out_of_image()
        #ia.imshow(lsoi_aug.draw_on_image(image_aug, size=3))
        imageio.imwrite(os.path.join(img_savesubdir, save_aug_imgname), image_aug)
        saveLineString(os.path.join(label_savesubdir, save_aug_labelname), lsoi_aug)


lines_file=open("train.txt","r")
lines_list=lines_file.readlines()
lines_list=[line.strip() for line in lines_list]


pool = Pool(22)
requests = pool.map(dealEachFile,lines_list)
pool.close()
pool.join()
#for line_path in lines_list:
#    dealEachFile(line_path)
"""
def dealFile(img_path,img_savedir,label_path,label_savedir):
    basename = os.path.basename(img_path).split(".")[0]
    img_savesubdir = img_savedir
    label_savesubdir = label_savedir
    if (not os.path.exists(img_savesubdir)):
        os.makedirs(img_savesubdir)
    if (not os.path.exists(label_savesubdir)):
        os.makedirs(label_savesubdir)

    save_imgname = basename + ".jpg"
    save_labelname = basename + ".txt"

    image = imageio.imread(img_path)
    lines_vec = []

    lines_tem = load_line(label_path)
    lines_svec = []
    for idx, line in enumerate(lines_tem):
        line_pts = []
        p1 = (line[1], line[2])
        p2 = (line[3], line[4])

        line_pts.append(p1)
        line_pts.append(p2)
        line_s = LineString(line_pts, label="{}".format(int(line[0])))
        lines_svec.append(line_s)
    if (len(lines_svec) < 25):
        return
    lsoi = LineStringsOnImage(lines_svec, shape=image.shape)

    imageio.imwrite(os.path.join(img_savesubdir, save_imgname), image)
    saveLineString(os.path.join(label_savesubdir, save_labelname), lsoi)

    for t in range(10):
        save_aug_imgname = save_imgname.split(".")[0] + "_" + str(t) + ".jpg"
        save_aug_labelname = save_labelname.split(".")[0] + "_" + str(t) + ".txt"
        image_aug, lsoi_aug = seq(image=image, line_strings=lsoi)
        lsoi_aug = lsoi_aug.remove_out_of_image()
        lsoi_aug = lsoi_aug.clip_out_of_image()
        line_len = len(lsoi_aug.line_strings)
        if (line_len < 25):
            continue
        # ia.imshow(lsoi_aug.draw_on_image(image_aug, size=3))
        imageio.imwrite(os.path.join(img_savesubdir, save_aug_imgname), image_aug)
        saveLineString(os.path.join(label_savesubdir, save_aug_labelname), lsoi_aug)
        img_len = len(os.listdir(img_savesubdir))
        if(img_len<2):
            shutil.rmtree(img_savesubdir)
            shutil.rmtree(label_savesubdir)

def dealEachFile(merge_path):
    imgs,labels=merge_path
    left_img,right_img=imgs
    left_label,right_label=labels
    left_imgbasename=os.path.basename(left_img)
    right_imgbasename=os.path.basename(right_img)
    save_basename=left_imgbasename.split(".")[0]+"_"+right_imgbasename.split(".")[0]
    secene_name=os.path.dirname(left_img)
    secene_name=os.path.dirname(secene_name)
    secene_name=os.path.basename(secene_name)
    img_savesubdir=os.path.join(img_savedir,secene_name,save_basename)
    leftimg_savename=os.path.join(img_savesubdir,left_imgbasename.split(".")[0])
    label_savesubdir=os.path.join(label_savedir,secene_name,save_basename)
    leftlabel_savename=os.path.join(label_savesubdir,left_imgbasename.split(".")[0])
    dealFile(left_img,leftimg_savename,left_label,leftlabel_savename)

    rightimg_savename=os.path.join(img_savesubdir,right_imgbasename.split(".")[0])
    rightlabel_savename = os.path.join(label_savesubdir, right_imgbasename.split(".")[0])
    dealFile(right_img, rightimg_savename, right_label, rightlabel_savename)


img_file=open(train_file,"r")
img_list=img_file.readlines()
line_file=open(label_file,"r")
lines_list=line_file.readlines()
img_list=[item.strip() for item in img_list]
lines_list=[item.strip() for item in lines_list]
img_list=[item.split(" ") for item in img_list]
lines_list=[item.split(" ") for item in lines_list]
merge_list=[]
for idx in range(len(img_list)):
    merge_list.append((img_list[idx],lines_list[idx]))

pool = Pool(22)
requests = pool.map(dealEachFile,merge_list)
pool.close()
pool.join()
#for merge in merge_list:
#    dealEachFile(merge)