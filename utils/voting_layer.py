import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
class Voting(nn.Module):
    """
    Voting Layer computes a new row-stotatic matrix with softmax. A large number (alpha) is multiplied to the input
    stochastic matrix to scale up the difference.
    Parameter: value multiplied before softmax alpha
               threshold that will ignore such points while calculating displacement in pixels pixel_thresh
    Input: permutation or doubly stochastic matrix s
           ///point set on source image P_src
           ///point set on target image P_tgt
           ground truth number of effective points in source image ns_gt
    Output: softmax matrix s
    """
    def __init__(self, alpha=200, pixel_thresh=None):
        super(Voting, self).__init__()
        self.alpha = alpha
        self.softmax = nn.Softmax(dim=-1)  # Voting among columns
        self.pixel_thresh = pixel_thresh
        self.learning_z = Parameter(torch.Tensor(1))
        self.reset_param()

    def reset_param(self):
        self.learning_z.data.fill_(0)
    def forward(self, s, nrow_gt, ncol_gt=None):

        # filter dummy nodes
        batch_size,h,w = s.shape
        ret_s = torch.zeros(batch_size, h, w, device=s.device)

        for b, n in enumerate(nrow_gt):
            if ncol_gt is None:
                ret_s[b, 0:n, :] = \
                    self.softmax(self.alpha * s[b, 0:(n+1), :])
            else:
                tem_mat=s[b, 0:(n+1), 0:(ncol_gt[b]+1)]
                ret_s[b, 0:(n+1), 0:(ncol_gt[b]+1)] =\
                    self.softmax(self.alpha *tem_mat)
        return ret_s