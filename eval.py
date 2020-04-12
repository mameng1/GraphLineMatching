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
            perm_mat = inputs['gt_perm_mat'].cuda()
            #src_inliers,tgt_inliers = [_.cuda() for _ in inputs['inliers']]
            batch_num = data1.size(0)

            iter_num = iter_num + 1

            with torch.set_grad_enabled(False):
                s_pred, pred,match_emb1,match_emb2,match_edgeemb1,match_edgeemb2= \
                    model(data1, data2, P1_gt, P2_gt, n1_gt, n2_gt,train_stage=False)

            s_pred_perm = lap_solver(s_pred, perm_mat, n1_gt, n2_gt)
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
