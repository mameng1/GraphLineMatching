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

def hungarian(s: torch.Tensor,gt,n1=None, n2=None):
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

    batch_size,height,width=perm_mat.shape
    small_perm_mat=np.zeros((batch_size,height-1,width-1),dtype=np.float32)
    score_mat=-1*perm_mat
    for b in range(batch_num):
        n1b = n1[b]
        n2b = n2[b]

        score_region = score_mat[b, :n1b + 1, :n2b + 1]

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
       
        G0 = ot.emd(rows_max, cols_max, invscore_region)

        row,col=np.where(G0[:n1b,:n2b]==1)
      
        result=np.zeros_like(small_perm_mat[b])
        small_perm_mat[b] = np.zeros_like(small_perm_mat[b])

        result[row,col]=1

        for row_idx,col_idx in zip(row,col):
            cur_score=score_region[row_idx,col_idx]
            last_cols=score_region[row_idx,-1]
            last_rows=score_region[-1,col_idx]
            if((0.8*cur_score<last_cols) or
               (0.8*cur_score<last_rows)):
                result[row_idx,col_idx]=0

        small_perm_mat[b] = result

        score=expand_scorem[:n1b,:n2b]
     
    small_perm_mat = torch.from_numpy(small_perm_mat).to(device)

    return small_perm_mat
