import torch
import time
from datetime import datetime
from pathlib import Path
import argparse

from utils.hungarian import hungarian
from utils.evaluation_metric import matching_accuracy
from parallel import DataParallel
from utils.model_sl import load_model
from torchvision import transforms
from utils.config import cfg
from PIL import Image,ImageDraw, ImageOps
from data.scannet_load import ScannetDataset,get_dataloader
def pred_vis(data1,data2,pts1,pts2,pred_link,cfg):
    mean=torch.tensor([cfg.NORM_MEANS])
    std=torch.tensor([cfg.NORM_STD])

    mean=mean.transpose(1,0)
    std=std.transpose(1,0)

    mean = torch.unsqueeze(mean,2)
    std = torch.unsqueeze(std, 2)

    data1=data1.cpu()
    data2 = data2.cpu()
    pts1 = pts1.cpu().numpy()
    pts2 = pts2.cpu().numpy()
    pred_link = pred_link.cpu().numpy()
    _,lrow,lcol=pred_link.shape

    batch_size=data1.size(0)
    for i in range(batch_size):
        data1_slice=data1[i]
        data2_slice=data2[i]
        pts1_slice=pts1[i]
        pts2_slice=pts2[i]
        predlink_slice=pred_link[i]

        data1_slice=data1_slice*std+mean
        data2_slice=data2_slice*std+mean
        height,width=data1_slice.size(1),data1_slice.size(2)
        data1_slice=transforms.ToPILImage()(data1_slice)
        data2_slice=transforms.ToPILImage()(data2_slice)
        vis_img=Image.new(data1_slice.mode,(width*2,height))
        vis_img.paste(data1_slice,(0,0))
        vis_img.paste(data2_slice,(width,0))
        canvas=ImageDraw.Draw(vis_img)

        for ridx in range(lrow):
            for cidx in range(lcol):
                if(predlink_slice[ridx,cidx]>0.95):
                    pt1=pts1_slice[ridx]
                    pt2=pts2_slice[cidx]
                    canvas.ellipse((pt1[0]-3,pt1[1]-3,pt1[0]+3,pt1[1]+3),fill=(0,0,255))
                    canvas.ellipse((width+pt2[0] - 3,pt2[1] - 3, width+pt2[0] + 3, pt2[1] + 3), fill=(0, 0, 255))
                    canvas.line((pt1[0],pt1[1],width+pt2[0],pt2[1]),fill=(0,255,0))
        img_name="result_{}.png".format(i)
        vis_img.save("./output/"+img_name)
        debug=0

def eval_model(model, dataloader, eval_epoch=None, verbose=False):
    print('Start evaluation...')
    since = time.time()

    device = next(model.parameters()).device

    if eval_epoch is not None:
        model_path = str(Path(cfg.OUTPUT_PATH) / 'params' / 'params_{:04}.pt'.format(eval_epoch))
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path)

    was_training = model.training
    model.eval()

    ds = dataloader.dataset
    classes = ds.classes
    #cls_cache = ds.cls

    lap_solver = hungarian

    accs = torch.zeros(len(classes), device=device)
    accs_prec = torch.zeros(len(classes), device=device)

    for i, cls in enumerate(classes):
        if verbose:
            print('Evaluating class {}: {}/{}'.format(cls, i, len(classes)))

        running_since = time.time()
        iter_num = 0

        ds.cls = cls
        acc_match_num = torch.zeros(1, device=device)
        acc_total_num = torch.zeros(1, device=device)
        acc_total_pred_num= torch.zeros(1, device=device)
        for inputs in dataloader:
            if 'images' in inputs:
                data1, data2 = [_.cuda() for _ in inputs['images']]
                inp_type = 'img'
            elif 'features' in inputs:
                data1, data2 = [_.cuda() for _ in inputs['features']]
                inp_type = 'feat'
            else:
                raise ValueError('no valid data key (\'images\' or \'features\') found from dataloader!')
            P1_gt, P2_gt = [_.cuda() for _ in inputs['Ps']]
            n1_gt, n2_gt = [_.cuda() for _ in inputs['ns']]
            e1_gt, e2_gt = [_.cuda() for _ in inputs['es']]
            G1_gt, G2_gt = [_.cuda() for _ in inputs['Gs']]
            H1_gt, H2_gt = [_.cuda() for _ in inputs['Hs']]
            KG, KH = [_.cuda() for _ in inputs['Ks']]
            perm_mat = inputs['gt_perm_mat'].cuda()
            #src_inliers,tgt_inliers = [_.cuda() for _ in inputs['inliers']]
            batch_num = data1.size(0)

            iter_num = iter_num + 1

            with torch.set_grad_enabled(False):
                s_pred, pred,match_emb1,match_emb2,match_edgeemb1,match_edgeemb2= \
                    model(data1, data2, P1_gt, P2_gt, G1_gt, G2_gt, H1_gt, H2_gt, n1_gt, n2_gt, KG, KH, inp_type,train_stage=False)

            #print("min:{},max:{}".format(torch.min(s_pred).item(), torch.max(s_pred).item()))
            s_pred_perm = lap_solver(s_pred, perm_mat, n1_gt, n2_gt)
            #if (iter_num == 1):
            #    pred_vis(data1, data2, P1_gt, P2_gt, s_pred_perm,cfg)
            _, _acc_match_num, _acc_total_num,_acc_totalpred_num = matching_accuracy(s_pred_perm, perm_mat, n1_gt,n2_gt)
            acc_match_num += _acc_match_num
            acc_total_num += _acc_total_num
            acc_total_pred_num += _acc_totalpred_num

            if iter_num % cfg.STATISTIC_STEP == 0 and verbose:
                running_speed = cfg.STATISTIC_STEP * batch_num / (time.time() - running_since)
                print('Class {:<8} Iteration {:<4} {:>4.2f}sample/s'.format(cls, iter_num, running_speed))
                running_since = time.time()

        accs[i] = acc_match_num / acc_total_num
        accs_prec[i] = acc_match_num / acc_total_pred_num
        if verbose:
            print('Class {} acc = {:.4f}'.format(cls, accs[i]))

    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    model.train(mode=was_training)
    #ds.cls = cls_cache

    print('Matching accuracy')
    for cls, single_acc in zip(classes, accs):
        print('{} = {:.4f}'.format(cls, single_acc))
    print('average = {:.4f}'.format(torch.mean(accs)))
    print('average precision = {:.4f}'.format(torch.mean(accs_prec)))
    return accs
def image2tensor(img_dir):
    with Image.open(img_dir) as img:
        height = img.height
        width = img.width
        old_size = (width, height)
        ratio = float(cfg.obj_resize[0]) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        obj = img.resize(new_size, resample=Image.BICUBIC)
        delta_w = cfg.obj_resize[0] - new_size[0]
        delta_h = cfg.obj_resize[1] - new_size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        obj = ImageOps.expand(obj, padding)
        return obj
def readLines(img_dir,lines_path):
    img=Image.open(img_dir)
    height = img.height
    width = img.width
    old_size = (width, height)
    ratio = float(cfg.obj_resize[0]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    delta_w = cfg.obj_resize[0] - new_size[0]
    delta_h = cfg.obj_resize[1] - new_size[1]
    keypoint_list = []
    with open(lines_path, "r") as anno:
        all_keypts = anno.readlines()
        all_keypts = [i.split(" ") for i in all_keypts]
        for keypoint in all_keypts:
            keypoint[1] = float(keypoint[1]) * ratio + delta_w // 2
            keypoint[3] = float(keypoint[3]) * ratio + delta_w // 2
            keypoint[2] = float(keypoint[2]) * ratio + delta_h // 2
            keypoint[4] = float(keypoint[4]) * ratio + delta_h // 2
            keypoint_list.append(keypoint)
    return keypoint_list
if __name__ == '__main__':
    from utils.dup_stdout_manager import DupStdoutFileManager
    from utils.parse_args import parse_args
    from utils.print_easydict import print_easydict

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cfg', dest='cfg_file', action='append',
                        help='an optional config file', default=[
            "/home/mameng/deeplearning/graph-matching/PCA/PCA-GM1/experiments/vgg16_pca_scannet.yaml"], type=str)
    parser.add_argument('--model', dest='model',
                        help='model name', default=None, type=str)
    parser.add_argument('--left_img', dest='dataset',
                        help='left image name', default=None, type=str)
    parser.add_argument('--right_img', dest='dataset',
                        help='right image name', default=None, type=str)
    parser.add_argument('--left_lines', dest='dataset',
                        help='left lines name', default=None, type=str)
    parser.add_argument('--right_lines', dest='dataset',
                        help='right lines name', default=None, type=str)
    args = parser.parse_args()

    left_img = Image.open(args.left_img)
    right_img = Image.open(args.right_img)

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.NORM_MEANS, cfg.NORM_STD)
    ])


    left_normimg=image2tensor(left_img)
    right_normimg=image2tensor(left_img)



    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    torch.manual_seed(cfg.RANDOM_SEED)

    image_dataset = ScannetDataset(cfg.DATASET_FULL_NAME,
                              sets='test',
                              length=cfg.EVAL.SAMPLES,
                              obj_resize=cfg.PAIR.RESCALE,
                              expand_region=cfg.EXPAND_REGION)
    dataloader = get_dataloader(image_dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net(cfg.OUTPUT_SIZE,cfg.SCALES)
    model = model.to(device)
    model = DataParallel(model, device_ids=cfg.GPUS)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('eval_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        classes = dataloader.dataset.classes
        pcks = eval_model(model, dataloader,
                          eval_epoch=cfg.EVAL.EPOCH if cfg.EVAL.EPOCH != 0 else None,
                          verbose=True)
