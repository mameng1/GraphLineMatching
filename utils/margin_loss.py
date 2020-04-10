import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MarginLoss(nn.Module):
    """
    Point description matching loss
    """
    def __init__(self,s=10.0, m=0.5, easy_margin=False):
        super(MarginLoss, self).__init__()
        self.s = s
        self.m = m

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
    def forward(self, emd1,emd2,label,nrows,ncols):
        batch_num = emd1.shape[0]
        cosine = torch.bmm(F.normalize(emd1,dim=-1), F.normalize(emd2,dim=-1).transpose(1,2))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        loss = torch.tensor(0.).to(emd1.device)
        for b in range(batch_num):
            b_rows = slice(0, nrows[b])
            b_cols = slice(0, ncols[b])
            valid_label=label[b,b_rows,b_cols]
            valid_phi=phi[b,b_rows,b_cols]
            valid_cosine=cosine[b,b_rows,b_cols]
            scale_value = valid_phi*valid_label+(1-valid_label)*valid_cosine
            scale_value = scale_value*self.s

            row_idx,col_idx=torch.nonzero(valid_label,as_tuple=True)
            if(len(row_idx)==0):
                continue
            row_idx,_=torch.sort(row_idx)
            col_idx,_=torch.sort(col_idx)
            valid_label=valid_label[row_idx,:]
            valid_label=valid_label[:,col_idx]
            scale_value=scale_value[row_idx,:]
            scale_value=scale_value[:,col_idx]
            class_idx=torch.nonzero(valid_label,as_tuple=True)[1]
            loss+=F.cross_entropy(scale_value,class_idx)
        return loss/batch_num