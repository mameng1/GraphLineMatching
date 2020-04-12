import torch
import scipy.optimize as opt
import ot
import numpy as np
from pymprog import *
def optimTans(s: torch.Tensor,gt,n1=None, n2=None):
    """
          Solve optimal LAP permutation by hungarian algorithm.
          :param s: input 3d tensor (first dimension represents batch)
          :param n1: [num of objs in dim1] (against padding)
          :param n2: [num of objs in dim2] (against padding)
          :return: optimal permutation matrix
          """
    device = s.device
    batch_num = s.shape[0]

    perm_mat = s.cpu().detach().numpy() * -1
    gt_s = gt.cpu().numpy()
    batch_size, height, width = perm_mat.shape
    small_perm_mat = np.zeros((batch_size, height - 1, width - 1), dtype=np.float32)
    score_mat = -1 * perm_mat
    for b in range(batch_num):
        n1b = n1[b]
        n2b = n2[b]
        # row, col = opt.linear_sum_assignment(perm_mat[b, :n1b, :n2b])
        rows_max = np.ones(n1b + 1, dtype=np.float32)
        rows_max[-1] = n2b
        cols_max = np.ones(n2b + 1, dtype=np.float32)
        cols_max[-1] = n1b
        invscore_region = perm_mat[b, :n1b + 1, :n2b + 1]


def assignment(s: torch.Tensor,gt,n1=None, n2=None):
    """
      Solve optimal LAP permutation by hungarian algorithm.
      :param s: input 3d tensor (first dimension represents batch)
      :param n1: [num of objs in dim1] (against padding)
      :param n2: [num of objs in dim2] (against padding)
      :return: optimal permutation matrix
      """
    device = s.device
    batch_num = s.shape[0]

    perm_mat = s.cpu().detach().numpy() * -1
   # gt_s = gt.cpu().numpy()
    batch_size, height, width = perm_mat.shape
    small_perm_mat = np.zeros((batch_size, height - 1, width - 1), dtype=np.float32)
    score_mat = -1 * perm_mat
    for b in range(batch_num):
        n1b = n1[b]
        n2b = n2[b]
        # row, col = opt.linear_sum_assignment(perm_mat[b, :n1b, :n2b])
        rows_max = np.ones(n1b + 1, dtype=np.float32)
        rows_max[-1] = n2b
        cols_max = np.ones(n2b + 1, dtype=np.float32)
        cols_max[-1] = n1b
        invscore_region = perm_mat[b, :n1b + 1, :n2b + 1]

        score_region = score_mat[b, :n1b + 1, :n2b + 1]
        expand_scorem = np.zeros((n1b + n2b, n1b + n2b), dtype=np.float32)
        expand_scorem[:n1b, :n2b] = score_region[:n1b, :n2b]
        diagup = np.diag(score_region[:, n2b])
        diaglow = np.diag(score_region[n1b, :])
        expand_scorem[:n1b, n2b:] = diagup[:-1, :-1]
        expand_scorem[n1b:, :n2b] = diaglow[:-1, :-1]
        # if(n1b>n2b):
        #rowi, coli = opt.linear_sum_assignment(expand_scorem[:n1b,:],maximize=True)
        # else:
        rowi, coli = opt.linear_sum_assignment(expand_scorem,maximize=True)
        perm_m=np.zeros_like(expand_scorem)
        perm_m[rowi,coli]=1
        perm_ms=perm_m[:n1b,:n2b]
        # cur_s=np.max(score_mat[b,:n1b,:n2b],axis=-1)
        #gt_sf = gt_s[b, :n1b, :n2b]

        small_perm_mat[b] = np.zeros_like(small_perm_mat[b])
        small_perm_mat[b, :n1b,:n2b] = perm_ms
        # if(not (np.all(np.sum(small_perm_mat,axis=-1) <= 1) and np.all(np.sum(small_perm_mat, axis=-2) <= 1))):
        #    row_sum=torch.sum(s[b, :n1b, :n2b],dim=0)
        #    col_sum=torch.sum(s[b, :n1b, :n2b],dim=1)
        #    debug=0
    small_perm_mat = torch.from_numpy(small_perm_mat).to(device)

    return small_perm_mat
def hungarian(s: torch.Tensor,gt,n1=None, n2=None):
    """
    Solve optimal LAP permutation by hungarian algorithm.
    :param s: input 3d tensor (first dimension represents batch)
    :param n1: [num of objs in dim1] (against padding)
    :param n2: [num of objs in dim2] (against padding)
    :return: optimal permutation matrix
    """
    #return assignment(s,gt,n1,n2)
    device = s.device
    batch_num = s.shape[0]

    perm_mat = s.cpu().detach().numpy() * -1
    #gt_s=gt.cpu().numpy()
    batch_size,height,width=perm_mat.shape
    small_perm_mat=np.zeros((batch_size,height-1,width-1),dtype=np.float32)
    score_mat=-1*perm_mat
    for b in range(batch_num):
        n1b = n1[b]
        n2b = n2[b]
        #row, col = opt.linear_sum_assignment(perm_mat[b, :n1b, :n2b])
        score_region = score_mat[b, :n1b + 1, :n2b + 1]
        #print(score_region[-1, -1])
        match_num=np.round(score_region[-1,-1])
        avg_score=score_region[-1,-1]/match_num
        rows_max=np.ones(n1b+1,dtype=np.float32)
        rows_max[-1]=n2b
        cols_max=np.ones(n2b+1,dtype=np.float32)
        cols_max[-1]=n1b
        invscore_region=perm_mat[b, :n1b+1, :n2b+1]


        expand_scorem=np.zeros((n1b+n2b,n1b+n2b),dtype=np.float32)
        expand_scorem[:n1b,:n2b]=score_region[:n1b,:n2b]
        diagup=np.diag(score_region[:,n2b])
        diaglow=np.diag(score_region[n1b,:])
        expand_scorem[:n1b,n2b:]=diagup[:-1,:-1]
        expand_scorem[n1b:,:n2b]=diaglow[:-1,:-1]
        #if(n1b>n2b):
        #rowi, coli = opt.linear_sum_assignment(expand_scorem[:,:n2b],maximize=True)
        #else:
        #rowi, coli = opt.linear_sum_assignment(expand_scorem,maximize=True)
        #perm_m=np.zeros_like(expand_scorem)
        #perm_m[rowi,coli]=1
        #perm_ms=perm_m[:n1b,:n2b]
        #cur_s=np.max(score_mat[b,:n1b,:n2b],axis=-1)
        #gt_sf=gt_s[b,:n1b,:n2b]
        #print(np.sum(gt_sf))
        #gt_sa=np.max(gt_s,axis=-1)
        debug=0
        #value=score_region[-1][-1]
        #fff=score_region[:n1b, :n2b]
        #debug1=np.sum(score_region[:n1b, :n2b],axis=-1)
        lambd = 1
        #ffff=ot.sinkhorn(rows_max, cols_max, invscore_region,lambd)
        #invscore_region[-1,-1]=0
        G0 = ot.emd(rows_max, cols_max, invscore_region)
        #debug=np.sum(G0[:n1b, :n2b],axis=-1)
        row,col=np.where(G0[:n1b,:n2b]==1)
        #filter_row=[]
        #filter_col=[]
        #for row_idx,col_idx in zip(row,col):
        #    cur_score=score_mat[b,row_idx,col_idx]
        #    if(cur_score>0):
        #        filter_row.append(row_idx)
        #        filter_col.append(col_idx)
        #tem_row,tem_col=np.where(score_mat[b, :n1b, :n2b]>0.5)
        result=np.zeros_like(small_perm_mat[b])
        small_perm_mat[b] = np.zeros_like(small_perm_mat[b])
        #small_perm_mat[b, :n1b,:n2b] = perm_ms
        result[row,col]=1

        for row_idx,col_idx in zip(row,col):
            cur_score=score_region[row_idx,col_idx]
            last_cols=score_region[row_idx,-1]
            last_rows=score_region[-1,col_idx]
            if((0.85*cur_score<last_cols) or
               (0.85*cur_score<last_rows)):
                result[row_idx,col_idx]=0

        small_perm_mat[b] = result
        ffff=small_perm_mat[b, :n1b,:n2b]
        score=expand_scorem[:n1b,:n2b]
        #if(not (np.all(np.sum(small_perm_mat,axis=-1) <= 1) and np.all(np.sum(small_perm_mat, axis=-2) <= 1))):
        #    row_sum=torch.sum(s[b, :n1b, :n2b],dim=0)
        #    col_sum=torch.sum(s[b, :n1b, :n2b],dim=1)
        #    debug=0
    small_perm_mat = torch.from_numpy(small_perm_mat).to(device)

    return small_perm_mat
