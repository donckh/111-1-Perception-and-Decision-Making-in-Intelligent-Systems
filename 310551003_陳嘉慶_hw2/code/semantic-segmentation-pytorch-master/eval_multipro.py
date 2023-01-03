# System libs
import os
import argparse
from distutils.version import LooseVersion
from multiprocessing import Queue, Process
# Numerical libs
import numpy as np
import math
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from mit_semseg.config import cfg
from mit_semseg.dataset import ValDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, parse_devices, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
import pandas as pd

colors = loadmat('data/color101.mat')['colors']


def visualize_result(data, pred, dir_result):
    (img, seg, info) = data

    # segmentation
    seg_color = colorEncode(seg, colors)

    # prediction
    pred_color = colorEncode(pred, colors)

    # aggregate images and save
    im_vis = np.concatenate((img, seg_color, pred_color), axis=1).astype(np.uint8)
    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(os.path.join(dir_result, img_name.replace('.jpg', '.png')))
    im_vis = pred_color  # seg_color, pred_color
    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(os.path.join("%s_pred" % dir_result, img_name.replace('.jpg', '.png')))


def evaluate(segmentation_module, loader, cfg, gpu_id, result_queue, epoch):
    segmentation_module.eval()

    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']

        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu_id)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu_id)

                # forward pass
                scores_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        # calculate accuracy and SEND THEM TO MASTER
        acc, pix = accuracy(pred, seg_label)
        intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class, mode='origi')
        # IOU: cfg.DATASET.num_class, 49, mode='origin'
        result_queue.put_nowait((acc, pix, intersection, union))

        # visualization
        if not os.path.isdir(os.path.join(cfg.DIR, 'result{}'.format(epoch))):
            os.makedirs(os.path.join(cfg.DIR, 'result{}'.format(epoch)))
        if not os.path.isdir(os.path.join(cfg.DIR, 'result_pred{}'.format(epoch))):
            os.makedirs(os.path.join(cfg.DIR, 'result_pred{}'.format(epoch)))

        if cfg.VAL.visualize:
            visualize_result(
                (batch_data['img_ori'], seg_label, batch_data['info']),
                pred,
                os.path.join(cfg.DIR, 'result{}'.format(epoch))
            )


def worker(cfg, gpu_id, start_idx, end_idx, result_queue, epoch):
    torch.cuda.set_device(gpu_id)
    print("set_device")
    # Dataset and Loader
    dataset_val = ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET,
        start_idx=start_idx, end_idx=end_idx)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=cfg.TRAIN.workers)  # 2

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)
    print("build network")
    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    segmentation_module.cuda()
    print("finish segmentation")
    # Main loop
    evaluate(segmentation_module, loader_val, cfg, gpu_id, result_queue, epoch)
    print("evaluate done!")


def main_eval(cfg, gpus, epoch="", mode='eval'):
    with open(cfg.DATASET.list_val, 'r') as f:
        lines = f.readlines()
        num_files = len(lines)

    num_files_per_gpu = math.ceil(num_files / len(gpus))

    pbar = tqdm(total=num_files)

    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    result_queue = Queue(500)
    procs = []

    for idx, gpu_id in enumerate(gpus):
        start_idx = idx * num_files_per_gpu
        end_idx = min(start_idx + num_files_per_gpu, num_files)  # cal num_files run per gpu
        proc = Process(target=worker, args=(cfg, gpu_id, start_idx, end_idx, result_queue, epoch))  # check worker & eval & visualize
        print('gpu:{}, start_idx:{}, end_idx:{}'.format(gpu_id, start_idx, end_idx))
        proc.start()  # start all gpus
        procs.append(proc)

    # master fetches results
    processed_counter = 0
    while processed_counter < num_files:
        if result_queue.empty():
            continue
        (acc, pix, intersection, union) = result_queue.get()
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
        processed_counter += 1
        # pbar.set_description("{}, {}, {}, {}".format(acc, pix, intersection, union))
        pbar.update(1)

    for p in procs:
        p.join()

    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    read_data = pd.read_csv("./data/apartment0_classes.csv")  # new
    numClass = read_data['label'].values  # label, Idx

    if mode == 'eval':
        for i, _iou in enumerate(iou):
            print('class [{}], IoU: {:.4f}'.format(i, _iou))
            with open('./{}/eval_record.txt'.format(cfg.DIR), 'a') as eval_record:
                eval_record.write(('class [{}], IoU: {:.4f}\n'.format(numClass[i], _iou)))  # numClass[i], i
        print('[Eval Summary]:')

    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%'.format(iou.mean(), acc_meter.average()*100))
    with open('./{}/eval_record.txt'.format(cfg.DIR), 'a') as eval_record:
        # eval_record.write('[Eval Summary]:')
        eval_record.write('Mean IoU: {:.4f}, Accuracy: {:.2f}%'.format(iou.mean(), acc_meter.average()*100))

    if mode != 'eval':
        return acc_meter.average()*100, iou.mean()
    # print('Evaluation Done!')


def test():
    print("Evaluation Done!!")


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Validation"
    )
    parser.add_argument(
        "--cfg",
        default="config/habitat-resnet50.yaml",  # ade20k-resnet50dilated-ppm_deepsup, habitat-resnet50
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0",  # 0-3, 0
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()
    cfg.DATASET.num_class = 101
    if os.path.exists('./{}/eval_record.txt'.format(cfg.DIR)):
        os.remove('./{}/eval_record.txt'.format(cfg.DIR))
    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))
    with open('./{}/eval_record.txt'.format(cfg.DIR), 'a') as eval_record:
        eval_record.write("Running with config:\n{}\n\n".format(cfg))

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.VAL.checkpoint)  # load directly from save checkpoint
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # if not os.path.isdir(os.path.join(cfg.DIR, "result")):
    #     os.makedirs(os.path.join(cfg.DIR, "result"))

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]

    main_eval(cfg, gpus)
    test()
