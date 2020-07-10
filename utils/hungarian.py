import torch
import scipy.optimize as opt
import ot
import numpy as np
from pymprog import *

def compute_perm(perm_pred,perm_score):
    row_sum=np.sum(perm_pred, axis=0)
    col_indces = np.where(row_sum > 1)[0].tolist()
    for col_id in col_indces:
        row_indces = np.where(perm_pred[:, col_id] == 1)[0].tolist()
        max_row_idx = np.argmax(perm_score[:, col_id])
        for rowid in row_indces:
            if (rowid != max_row_idx):
                perm_pred[rowid, col_id] = 0
                perm_score[rowid, col_id] = -1
    return perm_pred,perm_score
def hungarian(s: torch.Tensor,n1, n2,indeces1,indeces2,on1,on2,score_th=0.45):
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
    batch_size,height,width=perm_mat.shape
    max_rows=torch.max(on1)
    max_cols=torch.max(on2)
    small_perm_mat=np.zeros((batch_size,max_rows,max_cols),dtype=np.float32)
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

        batch_score=-invscore_region[:n1b,:n2b]
        match_lst = []

        # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        # matches = bf.match(d1.cpu().numpy(), d2.cpu().numpy())
        # matches = bf.knnMatch(d1.cpu().numpy(), d2.cpu().numpy(), k=1)
        for i in range(0, n1b):
            min_score = 0
            best_match = -1
            for j in range(0, n2b):
                score = batch_score[i, j]  # 2 - np.linalg.norm(np ld1.dot(ld2)
                if score > min_score:
                    min_score = score
                    best_match = j

            match_result = [i, best_match, min_score]
            match_lst.append(match_result)

        result = np.zeros_like(invscore_region[:n1b, :n2b])

        perm_score = np.ones_like(result) * -1
        perm_score = perm_score.astype(np.float)
        for item in match_lst:
            row_idx, col_idx, cur_dist = item
            if (cur_dist > score_th):
                result[row_idx, col_idx] = 1
                perm_score[row_idx, col_idx] = cur_dist

        result, perm_score = compute_perm(result.copy(), perm_score.copy())
        result, perm_score = compute_perm(result.copy().transpose(1, 0), perm_score.copy().transpose(1, 0))
        result=result.transpose(1,0)
        """
        #perm_mat[-1,-1]=0
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
        result=np.zeros_like(G0[:n1b,:n2b])
        assign_mask=np.ones_like(G0[:n1b,:n2b])
        small_perm_mat[b] = np.zeros_like(small_perm_mat[b])
        #small_perm_mat[b, :n1b,:n2b] = perm_ms
        result[row,col]=1

        for row_idx,col_idx in zip(row,col):
            cur_score=score_region[row_idx,col_idx]
            last_cols=score_region[row_idx,-1]
            last_rows=score_region[-1,col_idx]
            if((0.9*cur_score<last_cols) or
               (0.9*cur_score<last_rows)):
                result[row_idx,col_idx]=0
        """
        new_row,new_col=np.where(result==1)
        new_result=np.zeros_like(small_perm_mat[b])
        bindeces1=indeces1[b].cpu().numpy()
        bindeces2 = indeces2[b].cpu().numpy()
        for row_idx, col_idx in zip(new_row, new_col):
            new_result[bindeces1[row_idx],bindeces2[col_idx]]=1

        small_perm_mat[b] = new_result
        #if(not (np.all(np.sum(small_perm_mat,axis=-1) <= 1) and np.all(np.sum(small_perm_mat, axis=-2) <= 1))):
        #    row_sum=torch.sum(s[b, :n1b, :n2b],dim=0)
        #    col_sum=torch.sum(s[b, :n1b, :n2b],dim=1)
        #    debug=0
    small_perm_mat = torch.from_numpy(small_perm_mat).to(device)

    return small_perm_mat
