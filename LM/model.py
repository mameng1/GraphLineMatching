import torch
import torch.nn as nn

from utils.sinkhorn import Sinkhorn
from utils.voting_layer import Voting
from utils.feature_align import feature_align
from LM.gconv import Siamese_Gconv
from LM.affinity_layer import Affinity
from extension.line_alignpooling.linepool_align import LinePoolAlign
from utils.config import cfg
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
        gauss_k = cv2.getGaussianKernel(output_size[0], 1.5, cv2.CV_32F)
        gauss_k=gauss_k/np.linalg.norm(gauss_k)
        self.gauss_k=torch.from_numpy(gauss_k).cuda()
        self.gauss_k = self.gauss_k.unsqueeze(dim=0)
        self.gauss_k = self.gauss_k.unsqueeze(dim=1)
        self.gauss_k = None#self.gauss_k.contiguous()

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

    def forward(self, src, tgt, P_src, P_tgt, ns_src, ns_tgt, train_stage=True):
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

        # adjacency matrices
        emb1, emb2 = torch.cat((U_src, F_src), dim=1).transpose(1, 2), torch.cat((U_tgt, F_tgt), dim=1).transpose(1, 2)

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

        return s, None,match_emb1,match_emb2,match_edgeemb1,match_edgeemb2
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