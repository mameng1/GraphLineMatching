import argparse
from utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from pathlib import Path


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cfg', dest='cfg_file', type=str,
                        help='an optional config file', default="experiments/vgg16_scannet.yaml")
    parser.add_argument('--batch', dest='batch_size',
                        help='batch size', default=None, type=int)
    parser.add_argument('--epoch', dest='epoch',
                        help='epoch number', default=None, type=int)
    parser.add_argument('--model', dest='model',
                        help='model name', default=None, type=str)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset name', default=None, type=str)
    args = parser.parse_args()

    # load cfg from file
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    # load cfg from arguments
    if args.batch_size is not None:
        cfg_from_list(['BATCH_SIZE', args.batch_size])
    if args.epoch is not None:
        cfg_from_list(['TRAIN.START_EPOCH', args.epoch, 'EVAL.EPOCH', args.epoch, 'VISUAL.EPOCH', args.epoch])
    if args.model is not None:
        cfg_from_list(['MODEL_NAME', args.model])
    if args.dataset is not None:
        cfg_from_list(['DATASET_NAME', args.dataset])

    if len(cfg.MODEL_NAME) != 0 and len(cfg.DATASET_NAME) != 0:
        outp_path = get_output_dir(cfg.MODEL_NAME, cfg.DATASET_NAME)
        cfg_from_list(['OUTPUT_PATH', outp_path])
    assert len(cfg.OUTPUT_PATH) != 0, 'Invalid OUTPUT_PATH! Make sure model name and dataset name are specified.'
    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)

    return args

def test_parse_args(description):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', type=str,
                        help='an optional config file',
                        default="experiments/vgg16_scannet.yaml")
    parser.add_argument('--model_path', dest='model_path',
                        help='model name',
                        default='output/vgg16_linematching_wire/params/params_0004.pt', type=str)
    parser.add_argument('--left_img', dest='left_img',
                        help='left image name',
                        default='test_data/000800.jpg',
                        type=str)
    parser.add_argument('--right_img', dest='right_img',
                        help='right image name',
                        default='test_data/000900.jpg',
                        type=str)
    parser.add_argument('--left_lines', dest='left_lines',
                        help='left lines name',
                        default='test_data/000800.txt',
                        type=str)
    parser.add_argument('--right_lines', dest='right_lines',
                        help='right lines name',
                        default='test_data/000900.txt',
                        type=str)
    parser.add_argument('--output_path', dest='output_path',
                        help='output path name',
                        default='./test_data/',
                        type=str)
    args = parser.parse_args()

    # load cfg from file
    if args.cfg_file is not None:
         cfg_from_file(args.cfg_file)

    if len(cfg.MODEL_NAME) != 0 and len(cfg.DATASET_NAME) != 0:
        outp_path = get_output_dir(cfg.MODEL_NAME, cfg.DATASET_NAME)
        cfg_from_list(['OUTPUT_PATH', outp_path])
    assert len(cfg.OUTPUT_PATH) != 0, 'Invalid OUTPUT_PATH! Make sure model name and dataset name are specified.'
    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)

    return args
