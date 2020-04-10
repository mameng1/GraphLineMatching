import torch
import torch.nn as nn

class Sinkhorn(nn.Module):
    """
    BiStochastic Layer turns the input matrix into a bi-stochastic matrix.
    Parameter: maximum iterations max_iter
               a small number for numerical stability epsilon
    Input: input matrix s
    Output: bi-stochastic matrix s
    """
    def __init__(self, max_iter=100, epsilon=1e-4):
        super(Sinkhorn, self).__init__()
        self.max_iter = max_iter
        self.epsilon = epsilon
    def sink(self,M, reg, numItermax=100, stopThr=1e-3, cuda=True):

        # we assume that no distances are null except those of the diagonal of
        # distances

        if cuda:
            a = torch.ones((M.size()[0],)).cuda()
            b = torch.ones((M.size()[1],)).cuda()
            a[-1]=M.size()[1]-1
            b[-1]=M.size()[0]-1
        else:
            a = torch.ones((M.size()[0],))
            b = torch.ones((M.size()[1],))
            a[-1] = M.size()[1]-1
            b[-1] = M.size()[0]-1
        # init data
        na = len(a)
        nb = len(b)

        if cuda:
            u, v = (torch.ones(na)).cuda(), (torch.ones(nb)).cuda()
        else:
            u, v = (torch.ones(na)), (torch.ones(nb))

        # print(reg)

        #K = torch.exp(M / reg)
        # print(np.min(K))
        K=M
        Kp = (1 / a).view(-1, 1) * K
        cpt = 0
        err = 1
        err_res=1

        while (cpt < numItermax and err_res>stopThr):
            uprev = u
            vprev = v
            # print(T(K).size(), u.view(u.size()[0],1).size())
            KtransposeU = K.t().matmul(u)
            v = torch.div(b, KtransposeU)
            u = 1. / Kp.matmul(v)
            u_res=(uprev-u).norm(1).pow(2).item()
            v_res=(vprev-v).norm(1).pow(2).item()
            err_res=u_res+v_res
            #print("{},{}".format(u_res,v_res))
            if cpt % 10 == 0:
                transp = u.view(-1, 1) * (K * v)
                err = (torch.sum(transp) - b).norm(1).pow(2).item()

            cpt += 1
        return u.view((-1, 1)) * K * v.view((1, -1))


    def forward(self, s, nrows=None, ncols=None, exp=False, exp_alpha=20, dummy_row=False, dtype=torch.float32,last_layer=False,train_stage=True):
        batch_size = s.shape[0]

        if dummy_row:
            dummy_shape = list(s.shape)
            dummy_shape[1] = s.shape[2] - s.shape[1]
            s = torch.cat((s, torch.full(dummy_shape, 0.).to(s.device)), dim=1)
            new_nrows = ncols
            for b in range(batch_size):
                s[b, nrows[b]:new_nrows[b], :ncols[b]] = self.epsilon
            nrows = new_nrows

        s += self.epsilon
        h, w = s.shape[1] - 1, s.shape[2] - 1
        if (last_layer):
            tmp_mat = torch.zeros(batch_size, h + 1, w + 1, device=s.device)
        else:
            tmp_mat = torch.zeros(batch_size, h, w, device=s.device)
        for b in range(batch_size):
            b_rows = slice(0, nrows[b] if nrows is not None else s.shape[2])
            b_cols = slice(0, ncols[b] if ncols is not None else s.shape[1])
            eb_rows = slice(0, nrows[b] + 1)
            eb_cols = slice(0, ncols[b] + 1)
            expand_mat= s[b, eb_rows, eb_cols]
            s_tem = self.sink(expand_mat, 0.1)
            if (train_stage):
                s_tem[-1][-1] = 0
            if (last_layer):
                tmp_mat[b, eb_rows, eb_cols] = s_tem
            else:
                tmp_mat[b, b_rows, b_cols] = s_tem[b_rows, b_cols]

        return tmp_mat