import torch
import torch.nn as nn

from utils.sinkhorn import Sinkhorn
from utils.voting_layer import Voting
from utils.feature_align import feature_align
from LM.gconv import Siamese_Gconv
from LM.affinity_layer import Affinity
from extension.line_alignpooling.linepool_align import LinePoolAlign
from utils.config import cfg
from utils.pos_feature import PosFeatureLayer
import cv2
import numpy as np
import math
import utils.backbone
from utils.GATlayer import GraphAttentionLayer
import torch.nn.functional as F
CNN = eval('utils.backbone.{}'.format(cfg.BACKBONE))


class Net(CNN):
    def __init__(self,output_size,scales):
        super(Net, self).__init__()
        self.bi_stochastic = Sinkhorn(max_iter=cfg.LM.BS_ITER_NUM, epsilon=cfg.LM.BS_EPSILON)
        self.voting_layer = Voting(alpha=cfg.LM.VOTING_ALPHA)
        self.l2norm = nn.LocalResponseNorm(cfg.LM.FEATURE_CHANNEL * 2, alpha=cfg.LM.FEATURE_CHANNEL * 2, beta=0.5, k=0)
        self.gnn_layer = cfg.LM.GNN_LAYER
        self.feature_channel=cfg.LM.FEATURE_CHANNEL
        gauss_k = cv2.getGaussianKernel(output_size[0], 0.5, cv2.CV_32F)
        gauss_k=gauss_k/np.linalg.norm(gauss_k)
        self.gauss_k=torch.from_numpy(gauss_k).cuda()
        self.gauss_k = self.gauss_k.unsqueeze(dim=0)
        self.gauss_k = self.gauss_k.unsqueeze(dim=1)
        self.gauss_k = None#self.gauss_k.contiguous()
        self.pos_layer = PosFeatureLayer(4, cfg.PCA.FEATURE_CHANNEL * 2)
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_Gconv(cfg.LM.FEATURE_CHANNEL * 2, cfg.LM.GNN_FEAT)
                adjacA = GraphAttentionLayer(cfg.LM.FEATURE_CHANNEL * 2, 16, 0.8, 0.2,3,i)
                adjacB = GraphAttentionLayer(cfg.LM.FEATURE_CHANNEL * 2, 16, 0.8, 0.2,3,i)
            else:
                gnn_layer = Siamese_Gconv(cfg.LM.GNN_FEAT, cfg.LM.GNN_FEAT)
                adjacA = GraphAttentionLayer(cfg.LM.GNN_FEAT, 16, 0.8, 0.2,3,i)
                adjacB = GraphAttentionLayer(cfg.LM.GNN_FEAT, 16, 0.8, 0.2,3,i)
            self.add_module('adjacA_{}'.format(i), adjacA)
            self.add_module('adjacB_{}'.format(i), adjacB)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            self.add_module('affinity_{}'.format(i), Affinity(cfg.LM.GNN_FEAT))
            if i == self.gnn_layer - 2:  # only second last layer will have cross-graph module
                self.add_module('cross_graph_{}'.format(i), nn.Linear(cfg.LM.GNN_FEAT * 2, cfg.LM.GNN_FEAT))

        self.pyramid_linepooling = []
        for scale in scales:
            self.pyramid_linepooling.append(
                LinePoolAlign(output_size, scale),
            )
            self.pyramid_linepooling.append(
                LinePoolAlign(output_size, scale),
            )
        self.poolers = nn.ModuleList(self.pyramid_linepooling)

    def forward(self, src, tgt, P_src, P_tgt, ns_src, ns_tgt, train_stage=True,perm_mat=None,score_thresh=None):
        # extract feature
        src_node = self.node_layers(src)
        src_edge = self.edge_layers(src_node)
        tgt_node = self.node_layers(tgt)
        tgt_edge = self.edge_layers(tgt_node)

        # feature normalization
        src_node = self.l2norm(src_node)
        src_edge = self.l2norm(src_edge)
        tgt_node = self.l2norm(tgt_node)
        tgt_edge = self.l2norm(tgt_edge)

        U_src = self.line_feature_align(src_node, P_src, ns_src, 0)
        F_src = self.line_feature_align(src_edge, P_src, ns_src, 2)
        U_tgt = self.line_feature_align(tgt_node, P_tgt, ns_tgt, 1)
        F_tgt = self.line_feature_align(tgt_edge, P_tgt, ns_tgt, 3)

        ns_src_bk = ns_src
        ns_tgt_bk = ns_tgt
        U_src, U_tgt, F_src, F_tgt, ns_src, ns_tgt, indeces1, indeces2 \
            = self.filterOutlier(U_src, U_tgt, F_src, F_tgt, ns_src, ns_tgt, score_thresh)
        if (train_stage):
            perm_mat = self.update_perm(indeces1, indeces2, ns_src_bk, ns_tgt_bk, perm_mat)

        # adjacency matrices
        emb1, emb2 = torch.cat((U_src, F_src), dim=1).transpose(1, 2), torch.cat((U_tgt, F_tgt), dim=1).transpose(1, 2)
        emb1 = self.pos_layer(emb1, ns_src, P_src, indeces1, src.shape)
        emb2 = self.pos_layer(emb2, ns_tgt, P_tgt, indeces2, tgt.shape)

        match_emb1=U_src.transpose(1,2)
        match_emb2=U_tgt.transpose(1,2)
        match_edgeemb1=F_src.transpose(1,2)
        match_edgeemb2=F_tgt.transpose(1,2)
        for i in range(self.gnn_layer):
            adjacA=getattr(self, 'adjacA_{}'.format(i))
            A_src = adjacA(emb1, ns_src, ns_src)
            adjacB = getattr(self, 'adjacB_{}'.format(i))
            A_tgt = adjacB(emb2, ns_tgt, ns_tgt)

            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])
            affinity = getattr(self, 'affinity_{}'.format(i))
            s = affinity(emb1, emb2,ns_src, ns_tgt)
            s = self.voting_layer(s, ns_src, ns_tgt)

            if(i==self.gnn_layer-1):
                s = self.bi_stochastic(s, ns_src, ns_tgt,last_layer=True,train_stage=train_stage)
            else:
                s = self.bi_stochastic(s, ns_src, ns_tgt,train_stage=train_stage)
            if i == self.gnn_layer - 2:
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                emb1_new = cross_graph(torch.cat((emb1, torch.bmm(s, emb2)), dim=-1))
                emb2_new = cross_graph(torch.cat((emb2, torch.bmm(s.transpose(1,2), emb1)), dim=-1))
                emb1 = emb1_new
                emb2 = emb2_new
        if (train_stage):
            return s, match_emb1, match_emb2, match_edgeemb1, match_edgeemb2, perm_mat, ns_src, ns_tgt
        else:
            return s, indeces1, indeces2, ns_src, ns_tgt
    def line_feature_align(self,raw_feature, P, ns_t, layer_idx,device=None):
        if device is None:
            device = raw_feature.device

        batch_num = raw_feature.shape[0]
        channel_num = raw_feature.shape[1]
        n_max = P.shape[1]

        all_rois_num=torch.sum(ns_t)
        all_rois=torch.zeros(all_rois_num,6,dtype=torch.float32,device=device)
        base_idx=0
        for idx,rois in enumerate(P):
            n=ns_t[idx]
            _rois=rois[0:n]
            all_rois[base_idx:base_idx + n, 0]=idx
            all_rois[base_idx:base_idx+n,1:]=_rois
            base_idx=base_idx+n
        all_rois=all_rois.contiguous()
        rois_feature=self.poolers[layer_idx](raw_feature,all_rois)
        if(self.gauss_k is not None):
            gauss_k=self.gauss_k.expand_as(rois_feature)
            gauss_k=gauss_k.contiguous()
            rois_feature=rois_feature.mul(gauss_k)
            rois_feature=torch.mean(rois_feature,dim=2)
        else:
            rois_feature=torch.max(rois_feature,dim=2)[0]
        mean_features=torch.mean(rois_feature,dim=-1)
        mean_features=mean_features.squeeze()
        mean_features=mean_features.transpose(1,0)
        F = torch.zeros(batch_num, channel_num, n_max, dtype=torch.float32, device=device)
        base_idx=0
        for idx,n in enumerate(ns_t):
            F[idx,:,0:n]=mean_features[:,base_idx:base_idx + n]
            base_idx=base_idx+n
        return F
    def filterOutlier(self, low_emb1,low_emb2,high_emb1,high_emb2,ns_src, ns_tgt,score_thresh):
        #low_emb1 = F.normalize(low_emb1,dim=1)
        #low_emb2 = F.normalize(low_emb2, dim=1)
        #high_emb1 = F.normalize(high_emb1,dim=1)
        #high_emb2 = F.normalize(high_emb2,dim=1)

        score_low = torch.matmul(low_emb1.transpose(1,2),low_emb2)
        score_high = torch.matmul(high_emb1.transpose(1,2),high_emb2)

        batch_size = low_emb1.shape[0]

        lemb1_new, lemb2_new = [], []
        hemb1_new, hemb2_new = [], []

        ns_src_new = torch.zeros_like(ns_src)
        ns_tgt_new = torch.zeros_like(ns_tgt)
        indeces1,indeces2=[],[]
        for b in range(batch_size):
            lrow_max_v,_=torch.max(score_low[b,:ns_src[b],:ns_tgt[b]],dim=1)
            lcol_max_v,_=torch.max(score_low[b,:ns_tgt[b],:ns_tgt[b]],dim=0)

            hrow_max_v, _ = torch.max(score_high[b, :ns_src[b], :ns_tgt[b]], dim=1)
            hcol_max_v, _ = torch.max(score_high[b, :ns_tgt[b], :ns_tgt[b]], dim=0)

            row_max_i=torch.where((lrow_max_v>score_thresh)&(hrow_max_v>(2*score_thresh/3)))[0]
            col_max_i=torch.where((lcol_max_v>score_thresh)&(hcol_max_v>(2*score_thresh/3)))[0]

            indeces1.append(row_max_i)
            indeces2.append(col_max_i)

            lrows_elem=torch.index_select(low_emb1[b],1,row_max_i)
            lcols_elem=torch.index_select(low_emb2[b],1,col_max_i)

            hrows_elem = torch.index_select(high_emb1[b], 1, row_max_i)
            hcols_elem = torch.index_select(high_emb2[b], 1, col_max_i)

            lemb1_new.append(lrows_elem)
            lemb2_new.append(lcols_elem)

            hemb1_new.append(hrows_elem)
            hemb2_new.append(hcols_elem)

            ns_src_new[b]=len(row_max_i)
            ns_tgt_new[b]=len(col_max_i)

        lemb1_new = self.pad_tensor(lemb1_new)
        lemb2_new = self.pad_tensor(lemb2_new)

        lemb1_new = torch.stack(lemb1_new, dim=0)
        lemb2_new = torch.stack(lemb2_new, dim=0)

        hemb1_new = self.pad_tensor(hemb1_new)
        hemb2_new = self.pad_tensor(hemb2_new)

        hemb1_new = torch.stack(hemb1_new, dim=0)
        hemb2_new = torch.stack(hemb2_new, dim=0)

        return lemb1_new,lemb2_new,hemb1_new,hemb2_new,ns_src_new,ns_tgt_new,indeces1,indeces2
    def pad_tensor(self, inp):
        assert type(inp[0]) == torch.Tensor
        it = iter(inp)
        t = next(it)
        max_shape = list(t.shape)
        while True:
            try:
                t = next(it)
                for i in range(len(max_shape)):
                    max_shape[i] = int(max(max_shape[i], t.shape[i]))
            except StopIteration:
                break
        max_shape = np.array(max_shape)

        padded_ts = []
        for t in inp:
            pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
            pad_pattern[::-2] = max_shape - np.array(t.shape)
            pad_pattern = tuple(pad_pattern.tolist())
            padded_ts.append(F.pad(t, pad_pattern, 'constant', 0))

        return padded_ts
    def update_perm(self,indeces1,indeces2,ns_src,ns_tgt,perm_gt):
        batch_size = len(indeces1)
        update_perm_list=[]
        for b in range(batch_size):
            batch_indeces1 = indeces1[b]
            batch_indeces2 = indeces2[b]

            len1 = len(batch_indeces1)
            len2 = len(batch_indeces2)
            b_indecesn1 = torch.zeros((len1 + 1), dtype=torch.long, device=ns_src.device)
            b_indecesn1[:len1] = batch_indeces1
            b_indecesn1[len1] = ns_src[b]

            b_indecesn2 = torch.zeros((len2 + 1), dtype=torch.long, device=ns_src.device)
            b_indecesn2[:len2] = batch_indeces2
            b_indecesn2[len2] = ns_tgt[b]

            batch_perm=perm_gt[b]
            batch_perm=batch_perm[b_indecesn1,:]
            batch_perm=batch_perm[:,b_indecesn2]
            update_perm_list.append(batch_perm)

        perm = self.pad_tensor(update_perm_list)

        perm = torch.stack(perm, dim=0)
        return perm