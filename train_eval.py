import torch
import torch.optim as optim
import time
from datetime import datetime
from pathlib import Path
from tensorboardX import SummaryWriter

from data.scannet_load import ScannetDataset,get_dataloader
from utils.permutation_loss import CrossEntropyLoss
from utils.evaluation_metric import matching_accuracy
from utils.model_sl import load_model, save_model
from eval import eval_model
from utils.hungarian import hungarian
from utils.config import cfg
from utils.margin_loss import MarginLoss
import cv2
import numpy as np
def train_eval_model(model,
                     criterion,
                     optimizer,
                     dataloader,
                     tfboard_writer,
                     num_epochs=25,
                     resume=False,
                     start_epoch=0):
    print('Start training...')

    since = time.time()
    dataset_size = len(dataloader['train'].dataset)

    device = next(model.parameters()).device
    print('model on device: {}'.format(device))

    checkpoint_path = Path(cfg.OUTPUT_PATH) / 'params'
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    #model_path = str(checkpoint_path / 'params_{:04}.pt'.format(2))
    #print('Loading model parameters from {}'.format(model_path))
    #load_model(model, model_path)
    if resume:
        assert start_epoch != 0
        model_path = str(checkpoint_path / 'params_{:04}.pt'.format(start_epoch))
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path)

        optim_path = str(checkpoint_path / 'optim_{:04}.pt'.format(start_epoch))
        print('Loading optimizer state from {}'.format(optim_path))
        optimizer.load_state_dict(torch.load(optim_path))

    margin_loss=MarginLoss(30)
    marginedge_loss=MarginLoss(1,0.3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                 gamma=cfg.TRAIN.LR_DECAY,
                                                 last_epoch=cfg.TRAIN.START_EPOCH - 1)
    #scheduler.step()
    for epoch in range(start_epoch, num_epochs):
        score_thresh=min(epoch*0.1,0.5)
        print('Epoch {}/{},score_thresh {}'.format(epoch, num_epochs - 1,score_thresh))
        print('-' * 10)

        model.train()  # Set model to training mode

        print('lr = ' + ', '.join(['{:.2e}'.format(x['lr']) for x in optimizer.param_groups]))

        epoch_loss = 0.0
        running_loss = 0.0
        running_since = time.time()
        iter_num = 0

        # Iterate over data.
        for inputs in dataloader['train']:
            data1, data2 = [_.cuda() for _ in inputs['images']]

            P1_gt, P2_gt = [_.cuda() for _ in inputs['Ps']]
            n1_gt, n2_gt = [_.cuda() for _ in inputs['ns']]

            weights=inputs['ws'].cuda()
            perm_mat = inputs['gt_perm_mat'].cuda()
            iter_num = iter_num + 1

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # forward
                s_pred, d_pred,match_emb1,match_emb2,match_edgeemb1,match_edgeemb2,perm_mat,n1_gt,n2_gt = \
                    model(data1, data2, P1_gt, P2_gt, n1_gt, n2_gt,perm_mat=perm_mat,score_thresh=score_thresh)

                multi_loss = []
                loss_lsm = criterion(s_pred, perm_mat, n1_gt, n2_gt,weights)

                loss_marg=margin_loss(match_emb1,match_emb2,perm_mat,n1_gt, n2_gt)
                loss_edgemarg=marginedge_loss(match_edgeemb1,match_edgeemb2,perm_mat,n1_gt, n2_gt)
                loss=(loss_marg+loss_edgemarg)*0.25+loss_lsm#(loss_marg)*0.5+loss_pca
                # backward + optimize
                loss.backward()
                optimizer.step()

                # tfboard writer
                loss_dict = {'loss_{}'.format(i): l.item() for i, l in enumerate(multi_loss)}
                loss_dict['loss'] = loss.item()
                tfboard_writer.add_scalars('loss', loss_dict, epoch * cfg.TRAIN.EPOCH_ITERS + iter_num)
                # statistics
                running_loss += loss.item() * perm_mat.size(0)
                epoch_loss += loss.item() * perm_mat.size(0)

                if iter_num % cfg.STATISTIC_STEP == 0:
                    running_speed = cfg.STATISTIC_STEP * perm_mat.size(0) / (time.time() - running_since)
                    print('Epoch {:<4} Iteration {:<4} {:>4.2f}sample/s Loss={:<8.4f}'
                          .format(epoch, iter_num, running_speed, running_loss / cfg.STATISTIC_STEP / perm_mat.size(0)))
                    tfboard_writer.add_scalars(
                        'speed',
                        {'speed': running_speed},
                        epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                    )
                    running_loss = 0.0
                    running_since = time.time()

        epoch_loss = epoch_loss / dataset_size

        save_model(model, str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1)))
        torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}.pt'.format(epoch + 1)))

        print('Epoch {:<4} Loss: {:.4f}'.format(epoch, epoch_loss))
        print()

        # Eval in each epoch
        accs = eval_model(model, dataloader['test'],train_epoch=epoch)
        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'
          .format(time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))

    return model


if __name__ == '__main__':
    from utils.dup_stdout_manager import DupStdoutFileManager
    from utils.parse_args import parse_args
    from utils.print_easydict import print_easydict

    args = parse_args('Deep learning of graph matching training & evaluation code.')

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    torch.manual_seed(cfg.RANDOM_SEED)

    dataset_len = {'train': cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, 'test': cfg.EVAL.SAMPLES}
    image_dataset = {
        x: ScannetDataset(cfg.DATASET_FULL_NAME,
                     sets=x,
                     length=dataset_len[x],
                     cls=cfg.TRAIN.CLASS if x == 'train' else None,
                     obj_resize=cfg.PAIR.RESCALE,
                     expand_region=cfg.EXPAND_REGION)
        for x in ('train', 'test')}
    dataloader = {x: get_dataloader(image_dataset[x], fix_seed=(x == 'test'))
        for x in ('train', 'test')}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net(cfg.OUTPUT_SIZE,cfg.SCALES,)
    model = model.cuda()


    criterion = CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, nesterov=True)
    #model = DataParallel(model, device_ids=cfg.GPUS)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)

    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    tfboardwriter = SummaryWriter(logdir=str(Path(cfg.OUTPUT_PATH) / 'tensorboard' / 'training_{}'.format(now_time)))

    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('train_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        model = train_eval_model(model, criterion, optimizer, dataloader, tfboardwriter,
                                 num_epochs=cfg.TRAIN.NUM_EPOCHS,
                                 resume=cfg.TRAIN.START_EPOCH != 0,
                                 start_epoch=cfg.TRAIN.START_EPOCH)
