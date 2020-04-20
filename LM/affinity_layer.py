import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor
import math


class Affinity(nn.Module):
    """
    Affinity Layer to compute the affinity matrix from feature space.
    M = X * A * Y^T
    Parameter: scale of weight d
    Input: feature X, Y
    Output: affinity matrix M
    """
    def __init__(self, d):
        super(Affinity, self).__init__()
        self.d = d
        self.A = Parameter(Tensor(self.d, self.d))

        self.learning_feature_x = Parameter(torch.Tensor(self.d))
        #self.learning_feature_y = Parameter(torch.Tensor(self.d))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.d)
        self.A.data.uniform_(-stdv, stdv)
        self.A.data += torch.eye(self.d)

        self.learning_feature_x.data.uniform_(-stdv, stdv)
        #self.learning_feature_y.data.uniform_(-stdv, stdv)

    def forward(self, X, Y,ns_src, ns_tgt):

        assert X.shape[2] == Y.shape[2] == self.d

        batch_size=X.shape[0]
        x_h,y_h=X.shape[1],Y.shape[1]
        expand_X=torch.zeros(batch_size,x_h+1,X.shape[2],device=X.device)
        expand_Y=torch.zeros(batch_size,y_h+1,Y.shape[2],device=X.device)
        for b in range(batch_size):
            b_srcs = slice(0, ns_src[b])
            b_tgts = slice(0, ns_tgt[b])
            expand_X[b,b_srcs,:]=X[b,b_srcs,:]

            expand_Y[b, b_tgts, :] = Y[b, b_tgts, :]
            expand_Y[b, ns_tgt[b], :] = self.learning_feature_x
            expand_X[b, ns_src[b], :] = self.learning_feature_x

        M = torch.matmul(expand_X, (self.A + self.A.transpose(0, 1)) / 2)
        M = torch.matmul(M, expand_Y.transpose(1, 2))
        return M


