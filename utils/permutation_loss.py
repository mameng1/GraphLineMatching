import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    Cross entropy loss between two permutations.
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, pred_perm, gt_perm, pred_ns, gt_ns,weights):
        batch_num = pred_perm.shape[0]
        # for b in range(batch_num):
        #    debug=pred_perm[b, (pred_ns[b]+1), (gt_ns[b]+1)]
        #    ff=0
        pred_perm = pred_perm.to(dtype=torch.float32)
        # print("bef min:{},max:{}".format(torch.min(pred_perm).item(), torch.max(pred_perm).item()))
        if (not torch.all((pred_perm >= 0) * (pred_perm <= 1))):
            if (torch.sum(torch.isnan(pred_perm)).item() == 0):
                # print(torch.min(pred_perm).item())
                # print(torch.max(pred_perm).item())
                # pred_perm=pred_perm-torch.min(pred_perm)
                pred_perm = pred_perm / torch.max(pred_perm)
        # print("aft min:{},max:{}".format(torch.min(pred_perm).item(),torch.max(pred_perm).item()))

        assert torch.all((pred_perm >= 0) * (pred_perm <= 1))
        assert torch.all((gt_perm >= 0) * (gt_perm <= 1))

        loss = torch.tensor(0.).to(pred_perm.device)
        n_sum = torch.zeros_like(loss)

        for b in range(batch_num):
            # mask=gt_perm[b, :pred_ns[b], :gt_ns[b]].detach()
            # num_positive = torch.sum(mask).float()
            # num_negative = mask.numel() - num_positive
            # print (num_positive, num_negative)
            # weights=torch.zeros_like(mask)
            # weights[mask != 0] = num_negative / (num_positive + num_negative)
            # weights[mask == 0] = num_positive / (num_positive + num_negative)
            # weights=torch.ones(pred_ns[b]+1,gt_ns[b]+1,device=pred_perm.device)
            # weights[-1][-1]=0
            # loss += F.binary_cross_entropy(
            #    pred_perm[b, :(pred_ns[b]+1), :(gt_ns[b]+1)],
            #    gt_perm[b, :(pred_ns[b]+1), :(gt_ns[b]+1)],
            #    reduction='sum',weight=weights[b,:(pred_ns[b]+1), :(gt_ns[b]+1)])
            loss += F.binary_cross_entropy(
                pred_perm[b, :(pred_ns[b] + 1), :(gt_ns[b] + 1)],
                gt_perm[b, :(pred_ns[b] + 1), :(gt_ns[b] + 1)],
                reduction='sum',weight=weights[b,:(pred_ns[b] + 1), :(gt_ns[b] + 1)])
            # p = torch.exp(-logp)
            # focal_loss = (1 - p) ** 2 * logp*weights[b, :(pred_ns[b] + 1), :(gt_ns[b] + 1)]
            # loss += focal_loss.sum()
            n_sum += (pred_ns[b] + 1).to(n_sum.dtype).to(pred_perm.device)
        return loss / n_sum