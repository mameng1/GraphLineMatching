import torch
import time
from datetime import datetime
from pathlib import Path

from utils.hungarian import hungarian
from utils.evaluation_metric import matching_accuracy
from parallel import DataParallel
from utils.model_sl import load_model
from utils.config import cfg
from data.scannet_load import ScannetDataset,get_dataloader
import numpy as np
def eval_model(model, dataloader, eval_epoch=None, verbose=False,train_epoch=None):
    print('Start evaluation...')
    since = time.time()

    device = next(model.parameters()).device

    if eval_epoch is not None:
        model_path = str(Path(cfg.OUTPUT_PATH) / 'params' / 'params_{:04}.pt'.format(eval_epoch))
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path)
        score_thresh = 0.5
        print("score_thresh{}".format(score_thresh))
    if train_epoch is not None:
        score_thresh = min(train_epoch * 0.1, 0.5)
        print("score_thresh{}".format(score_thresh))
    was_training = model.training
    model.eval()

    lap_solver = hungarian


    running_since = time.time()
    iter_num = 0

    acc_match_num = torch.zeros(1, device=device)
    acc_total_num = torch.zeros(1, device=device)
    acc_total_pred_num= torch.zeros(1, device=device)
    for inputs in dataloader:

        data1, data2 = [_.cuda() for _ in inputs['images']]
        P1_gt, P2_gt = [_.cuda() for _ in inputs['Ps']]
        n1_gt, n2_gt = [_.cuda() for _ in inputs['ns']]
        perm_mat = inputs['gt_perm_mat'].cuda()
        batch_num = data1.size(0)

        iter_num = iter_num + 1

        with torch.set_grad_enabled(False):
            s_pred,indeces1,indeces2,newn1_gt,newn2_gt= \
                model(data1, data2, P1_gt, P2_gt, n1_gt, n2_gt,train_stage=False,perm_mat=perm_mat,score_thresh=score_thresh)

        s_pred_perm = lap_solver(s_pred, newn1_gt, newn2_gt, indeces1, indeces2, n1_gt, n2_gt)
        _acc_match_num, _acc_total_num,_acc_totalpred_num = matching_accuracy(s_pred_perm, perm_mat, n1_gt,n2_gt)
        acc_match_num += _acc_match_num
        acc_total_num += _acc_total_num
        acc_total_pred_num += _acc_totalpred_num
        if iter_num % cfg.STATISTIC_STEP == 0 and verbose:
            running_speed = cfg.STATISTIC_STEP * batch_num / (time.time() - running_since)
            print('Iteration {:<4} {:>4.2f}sample/s'.format(iter_num, running_speed))
            running_since = time.time()

    recalls = acc_match_num / acc_total_num
    accs_prec = acc_match_num / acc_total_pred_num

    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    model.train(mode=was_training)

    print('Matching accuracy')

    print('recall = {:.4f}'.format(recalls.item()))
    print('precision = {:.4f}'.format(accs_prec.item()))
    return [recalls.item()]

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
    dataloader = get_dataloader(image_dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net(cfg.OUTPUT_SIZE,cfg.SCALES)
    model = model.to(device)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('eval_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        classes = dataloader.dataset.classes
        pcks = eval_model(model, dataloader,
                          eval_epoch=cfg.EVAL.EPOCH if cfg.EVAL.EPOCH != 0 else None,
                          verbose=True)
