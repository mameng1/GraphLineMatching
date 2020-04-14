import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
import math
from data.scannet import Scannet
#from data.scannet_aug import Scannet

from utils.config import cfg


class ScannetDataset(Dataset):
    def __init__(self, name, length, expand_region,cls=None, **args):
        self.name = name
        self.ds = eval(self.name)(**args)
        if (self.ds.sets == "train"):
            self.length = length
        else:
            self.length = self.ds.length
        self.obj_size = self.ds.obj_resize
        self.expand_region=expand_region
        self.classes=self.ds.classes
    def __len__(self):
        return self.length

    def pts_to_boxes(self,pt_gt,num):
        boxes = []
        for idx in range(num):
            pts = pt_gt[idx]
            pt1 = (pts[0], pts[1])
            pt2 = (pts[2], pts[3])
            # print pt2[0], pt3[0]
            angle = 0
            if (pt1[0] - pt2[0]) != 0:
                angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
            else:
                angle = 90.0

            if angle < -45.0:
                angle = angle + 180

            x_ctr = float(pt1[0] + pt2[0]) / 2  
            y_ctr = float(pt1[1] + pt2[1]) / 2  
            width = math.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
            boxes.append([x_ctr, y_ctr,self.expand_region,width,angle])
        boxes=np.array(boxes)
        return boxes
    def __getitem__(self, idx):
        anno_pair, perm_mat,weights = self.ds.get_pair(idx)
        if perm_mat.size <= 2 * 2:
            return self.__getitem__(idx)

        P1_gt = [(kp[1], kp[2],kp[3],kp[4]) for kp in anno_pair[0]['keypoints']]
        P2_gt = [(kp[1], kp[2],kp[3],kp[4]) for kp in anno_pair[1]['keypoints']]

        n1_gt, n2_gt = len(P1_gt), len(P2_gt)

        P1_gt = np.array(P1_gt)
        P2_gt = np.array(P2_gt)

        pt1_boxes=self.pts_to_boxes(P1_gt,n1_gt)
        pt2_boxes=self.pts_to_boxes(P2_gt,n2_gt)

        idx_tens=np.array([idx])
        ret_dict = {'Ps': [torch.Tensor(x) for x in [pt1_boxes, pt2_boxes]],
                    'ns': [torch.tensor(x) for x in [n1_gt, n2_gt]],
                    'gt_perm_mat': perm_mat}

        imgs = [anno['image'] for anno in anno_pair]
        if imgs[0] is not None:
            trans = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(cfg.NORM_MEANS, cfg.NORM_STD)
                    ])
            imgs = [trans(img) for img in imgs]
            ret_dict['images'] = imgs
        elif 'feat' in anno_pair[0]['keypoints'][0]:
            feat1 = np.stack([kp['feat'] for kp in anno_pair[0]['keypoints']], axis=-1)
            feat2 = np.stack([kp['feat'] for kp in anno_pair[1]['keypoints']], axis=-1)
            ret_dict['features'] = [torch.Tensor(x) for x in [feat1, feat2]]

        return ret_dict


def collate_fn(data: list):
    """
    Create mini-batch data for training.
    :param data: data dict
    :return: mini-batch
    """
    def pad_tensor(inp):
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

    def stack(inp):
        if type(inp[0]) == list:
            ret = []
            for vs in zip(*inp):
                ret.append(stack(vs))
        elif type(inp[0]) == dict:
            ret = {}
            for kvs in zip(*[x.items() for x in inp]):
                ks, vs = zip(*kvs)
                for k in ks:
                    assert k == ks[0], "Key value mismatch."
                ret[k] = stack(vs)
        elif type(inp[0]) == torch.Tensor:
            new_t = pad_tensor(inp)
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == np.ndarray:
            new_t = pad_tensor([torch.from_numpy(x) for x in inp])
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == str:
            ret = inp
        else:
            raise ValueError('Cannot handle type {}'.format(type(inp[0])))
        return ret

    ret = stack(data)

    return ret


def worker_init_fix(worker_id):
    """
    Init dataloader workers with fixed seed.
    """
    random.seed(cfg.RANDOM_SEED + worker_id)
    np.random.seed(cfg.RANDOM_SEED + worker_id)


def worker_init_rand(worker_id):
    """
    Init dataloader workers with torch.initial_seed().
    torch.initial_seed() returns different seeds when called from different dataloader threads.
    """
    random.seed(torch.initial_seed())
    np.random.seed(torch.initial_seed() % 2 ** 32)


def get_dataloader(dataset, fix_seed=True, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=shuffle, num_workers=cfg.DATALOADER_NUM, collate_fn=collate_fn,
        pin_memory=False, worker_init_fn=worker_init_fix if fix_seed else worker_init_rand
    )
