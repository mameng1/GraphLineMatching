import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PosFeatureLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features):
        super(PosFeatureLayer, self).__init__()

        self.W = nn.Linear(in_features, out_features, bias=False)
        #self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)

    def forward(self, emb,num,pts,indeces,image_shape):
        _,_,height,width=image_shape
        using_pts=self.normalize_keypoints(pts[:,:,:2],(width,height))
        using_angle=self.normalize_angle(pts[:,:,4])
        using_len=self.normalize_len(pts[:,:,3],(width,height))

        using_pts=torch.cat((using_pts,using_angle,using_len),dim=-1)
        pose_feature=self.W(using_pts)
        batch_size = len(indeces)
        for b in range(batch_size):
            bindex = indeces[b]
            bpts = pose_feature[b]
            bnum=num[b]
            emb[b,:bnum,:]=torch.add(emb[b,:bnum,:],bpts[bindex,:])
        return emb

    def normalize_keypoints(self,kpts, image_shape):
        width, height = image_shape
        one = kpts.new_tensor(1)
        size = torch.stack([one * width, one * height])[None]
        center = size / 2
        scaling = size.max(1, keepdim=True).values * 0.7
        return (kpts - center[:, None, :]) / scaling[:, None, :]
    def normalize_angle(self,angle):
        angle=angle.unsqueeze(2)
        center = angle.new_tensor(45).unsqueeze(0)
        scaling = angle.new_tensor(180).unsqueeze(0)*0.7
        return (angle - center[None, None, :]) / scaling[None, None, :]
    def normalize_len(self,length,image_shape):
        length=length.unsqueeze(2)
        width, height = image_shape
        max_length=np.sqrt(width*width+height*height)
        scaling = length.new_tensor(1).unsqueeze(0)*max_length*0.7
        center = scaling / 2
        return (length - center[None, None, :]) / scaling[None, None, :]