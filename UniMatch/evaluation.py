import argparse
import logging
import os
import pprint

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.endovis import EndovisDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.backbone.swin import Transformer
from supervised import evaluate
from util.classes import CLASSES
from util.utils import count_params, init_log

parser = argparse.ArgumentParser(description='Proyecto AML')
parser.add_argument('--config', type=str, default = "/home/eugenie/These/ProyectoAML/UniMatch/configs/endovis2018.yaml")
parser.add_argument('--split', type=str, default = "1_2")
parser.add_argument('--device', default='cuda:3')
parser.add_argument('--supervision', type=str, default='semi', choices=['semi', 'fully'])

def main():
    args = parser.parse_args()
    device = args.device
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = 0, 1

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)

    model = DeepLabV3Plus(cfg)
    
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                    {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                    'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    model.to(device)

    valset = EndovisDataset(cfg['dataset'], cfg['data_root'], 'val')

    valsampler = torch.utils.data.RandomSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    epoch = -1
    if args.supervision == "semi":
        checkpoint_path = os.path.join("/home/eugenie/These/UniMatch/exp/endovis2018/unimatch/base/r101_OHEM", args.split, "seed")
    else:
        checkpoint_path = os.path.join("/home/eugenie/These/UniMatch/exp/endovis2018/supervised/base/r101", args.split)
    if os.path.exists(os.path.join(checkpoint_path, 'best.pth')):
        checkpoint = torch.load(os.path.join(checkpoint_path, 'best.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mcIoU, iou_class, IoU, mIoU = evaluate(model, valloader, eval_mode, cfg, device, func='eval')

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
              logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> IoU: {:.2f}'.format(eval_mode, IoU))
            logger.info('***** Evaluation {} ***** >>>> mIoU: {:.2f}'.format(eval_mode, mIoU))
            logger.info('***** Evaluation {} ***** >>>> mcIoU: {:.2f}\n'.format(eval_mode, mcIoU))
            
            writer.add_scalar('eval/IoU', IoU)
            writer.add_scalar('eval/mIoU', mIoU)
            writer.add_scalar('eval/mIoU_inst', mcIoU)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou)

if __name__ == '__main__':
    main()