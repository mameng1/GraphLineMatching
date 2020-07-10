import torch
import time
from datetime import datetime
from pathlib import Path
from utils.hungarian import hungarian

from utils.evaluation_metric import matching_accuracy

from utils.model_sl import load_model
from torchvision import transforms
from utils.config import cfg
from PIL import Image,ImageDraw

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

def eval_model(model, dataloader, eval_epoch=None, verbose=False,train_epoch=None):
    print('Start evaluation...')

    device = next(model.parameters()).device

    if eval_epoch is not None:
        model_path = str(Path(cfg.OUTPUT_PATH) / 'params' / 'params_{:04}.pt'.format(eval_epoch))
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path)
        score_thresh = 0.2
        print("score_thresh{}".format(score_thresh))
    if train_epoch is not None:
        score_thresh = min(train_epoch * 0.1, 0.5)
        print("score_thresh{}".format(score_thresh))

    model.eval()

    ds = dataloader.dataset

    lap_solver = hungarian

    running_since = time.time()
    iter_num = 0

    score_th_list1= list(range(9, 0, -1))
    score_th_list1=[i/10 for i in score_th_list1]
    score_th_list2=list(range(10, 0, -1))
    score_th_list2 = [i / 1000 for i in score_th_list2]
    score_th_list =score_th_list1+score_th_list2#score_th_list1

    acc_match_num = torch.zeros(len(score_th_list), device=device)#torch.zeros(1, device=device)
    acc_total_num = torch.zeros(len(score_th_list), device=device)#torch.zeros(1, device=device)
    acc_total_pred_num= torch.zeros(len(score_th_list), device=device)#torch.zeros(1, device=device)

    for inputs in dataloader:
        data1, data2 = [_.cuda() for _ in inputs['images']]

        P1_gt, P2_gt = [_.cuda() for _ in inputs['Ps']]
        n1_gt, n2_gt = [_.cuda() for _ in inputs['ns']]

        perm_mat = inputs['gt_perm_mat'].cuda()
        weights=inputs['ws'].cuda()
        batch_num = data1.size(0)

        iter_num = iter_num + 1

        with torch.set_grad_enabled(False):
            s_pred, pred,match_emb1,match_emb2,match_edgeemb1,match_edgeemb2,indeces1,indeces2,newn1_gt,newn2_gt= \
                model(data1, data2, P1_gt, P2_gt, n1_gt, n2_gt,train_stage=False,perm_mat=perm_mat,score_thresh=score_thresh)

        for idx,score_th in enumerate(score_th_list):
            s_pred_perm = lap_solver(s_pred, newn1_gt, newn2_gt,indeces1,indeces2,n1_gt, n2_gt,score_th=score_th)
            _, _acc_match_num, _acc_total_num,_acc_totalpred_num = matching_accuracy(s_pred_perm, perm_mat, n1_gt,n2_gt,weights)

            acc_match_num[idx] += _acc_match_num
            acc_total_num[idx] += _acc_total_num
            acc_total_pred_num[idx] += _acc_totalpred_num

        if iter_num % cfg.STATISTIC_STEP == 0 and verbose:
            running_speed = cfg.STATISTIC_STEP * batch_num / (time.time() - running_since)
            print('Iteration {:<4} {:>4.2f}sample/s'.format(iter_num, running_speed))
            running_since = time.time()
    recalls = acc_match_num / acc_total_num
    accs_prec = acc_match_num / acc_total_pred_num
    F1=2*recalls*accs_prec/(accs_prec+recalls)
    print("score")
    print(score_th_list)
    print("recall")
    print(recalls.cpu().numpy().tolist())
    print("accu")
    print(accs_prec.cpu().numpy().tolist())
    print("F1")
    print(F1.cpu().numpy().tolist())
    return None

if __name__ == '__main__':
    from utils.dup_stdout_manager import DupStdoutFileManager
    from utils.parse_args import parse_args
    from utils.print_easydict import print_easydict

    args = parse_args('Deep learning of graph matching evaluation code.')

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    torch.manual_seed(cfg.RANDOM_SEED)

    image_dataset = ScannetDataset(cfg.DATASET_FULL_NAME,
                              sets='test',
                              length=cfg.EVAL.SAMPLES,
                              obj_resize=cfg.PAIR.RESCALE,
                              expand_region=cfg.EXPAND_REGION)
    dataloader = get_dataloader(image_dataset,shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net(cfg.OUTPUT_SIZE,cfg.SCALES)
    model = model.to(device)
    #model = DataParallel(model, device_ids=cfg.GPUS)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('eval_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        classes = dataloader.dataset.classes
        pcks = eval_model(model, dataloader,
                        eval_epoch=cfg.EVAL.EPOCH if cfg.EVAL.EPOCH != 0 else None,
                        verbose=True)

