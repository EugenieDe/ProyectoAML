import argparse
import logging
import os
import pprint

import torch
import numpy as np
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from UniMatch.model.semseg.deeplabv3plus import DeepLabV3Plus
from UniMatch.util.classes import CLASSES
from UniMatch.util.ohem import ProbOhemCrossEntropy2d
from UniMatch.util.deep_lab_ce import DeepLabCE
from UniMatch.util.utils import count_params, AverageMeter, intersectionAndUnion, init_log

from UniMatch.dataset.endovis import EndovisDataset


def evaluate(model, loader, mode, cfg, device, func=None):
    torch.manual_seed(53073424)
    criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).to(device)
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    names = ['/home/eugenie/These/data/endovis2018/val/images/seq_2_frame057.png /home/eugenie/These/data/endovis2018/val/annotations/seq_2_frame057.png',
            '/home/eugenie/These/data/endovis2018/val/images/seq_15_frame022.png /home/eugenie/These/data/endovis2018/val/annotations/seq_15_frame022.png',
            '/home/eugenie/These/data/endovis2018/val/images/seq_5_frame014.png /home/eugenie/These/data/endovis2018/val/annotations/seq_5_frame014.png',
            '/home/eugenie/These/data/endovis2018/val/images/seq_5_frame137.png /home/eugenie/These/data/endovis2018/val/annotations/seq_5_frame137.png',
            '/home/eugenie/These/data/endovis2018/val/images/seq_9_frame094.png /home/eugenie/These/data/endovis2018/val/annotations/seq_9_frame094.png',
            '/home/eugenie/These/data/endovis2018/val/images/seq_15_frame126.png /home/eugenie/These/data/endovis2018/val/annotations/seq_15_frame126.png',
            '/home/eugenie/These/data/endovis2018/val/images/seq_2_frame103.png /home/eugenie/These/data/endovis2018/val/annotations/seq_2_frame103.png',
            '/home/eugenie/These/data/endovis2018/val/images/seq_2_frame012.png /home/eugenie/These/data/endovis2018/val/annotations/seq_2_frame012.png',
            '/home/eugenie/These/data/endovis2018/val/images/seq_2_frame090.png /home/eugenie/These/data/endovis2018/val/annotations/seq_2_frame90.png'
            ]
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    i=0
    loss_meter = AverageMeter()
    iou=0
    mIoU=0
    k=0
    with torch.no_grad():
        for img, img_norm, mask, id in loader:
            i+=1
            img_norm = img_norm.to(device) 
            pred = model(img_norm)
            mask = mask.to(device)
            loss = criterion_l(pred, mask)
            pred = pred.argmax(dim=1)
            intersection, union, _ = intersectionAndUnion(pred.cpu().numpy(), mask.cpu().numpy(), cfg['nclass'], 255)
            iouc = 0
            c=0
            interc = 0
            unc=0
            for j in range(1, cfg['nclass']):
                maskiou = mask.cpu().numpy()
                maskiou = np.logical_and(maskiou, maskiou==j)

                prediou = pred.cpu().numpy()
                prediou = np.logical_and(prediou, prediou==j)
                
                if np.sum(maskiou)!=0:
                    c+=1

                inter = np.sum(np.logical_and(maskiou, prediou))
                un = np.sum(np.logical_or(maskiou, prediou))
                interc += inter
                unc += un
                iouc += inter / (un+ 1e-10)
            if c!=0:
                iou += iouc/c
            else:
                iou += 1.0
            mIoU += interc / (unc+ 1e-10)

            
            if func == 'eval' and id[0] in names:
                visualize(cfg, img, pred, mask,k)
                k+=1

            reduced_intersection = torch.from_numpy(intersection).to(device) 
            reduced_union = torch.from_numpy(union).to(device) 

            loss_meter.update(loss.item())

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())

    iou = 100*iou/len(loader)
    mIoU = 100*mIoU/len(loader)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    iou_class_up = np.array([iou_class[0],iou_class[1],iou_class[2],iou_class[3],iou_class[6],iou_class[7],iou_class[8], iou_class[9]])

    mcIoU = np.mean(iou_class_up[1:])
    return mcIoU, iou_class_up, iou, mIoU

def visualize(cfg, img, pred, mask, k):
    im = img.numpy()
    pre = pred.cpu().numpy()
    mas = mask.cpu().numpy()
    im=im.transpose(2,3,1,0).squeeze()
    pre=pre.transpose(1,2,0)
    mas = mas.transpose(1,2,0)
    fig, ax = plt.subplots(1,3)
    ax[0].set_title('Image',fontsize = 5)
    ax[0].imshow(im)
    ax[0].axis('off')

    ax[1].set_title('Ground Truth',fontsize = 5)
    ax[1].imshow(np.zeros(mas.shape),cmap='gray')
    mas = mas.astype('float')
    mas[mas==0]=np.nan
    ax[1].imshow(mas,cmap='Set1',norm=Normalize(1,10))
    ax[1].axis('off')

    ax[2].set_title('Pred',fontsize = 5)
    ax[2].imshow(np.zeros(pre.shape),cmap='gray')
    pre = pre.astype('float')
    pre[pre==0]=np.nan
    ax[2].imshow(pre,cmap='Set1',norm=Normalize(1,10))
    ax[2].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    
    os.makedirs('output/1_{}'.format(cfg['split']), exist_ok=True)
    plt.savefig('output/1_{}/{}.png'.format(cfg['split'],k), dpi=400, bbox_inches='tight')
    plt.close()


def supervised(args, labeled_path, save_path):
    torch.manual_seed(53073424)
    device = args.device
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = 0, 1
    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(save_path)
        
        os.makedirs(save_path, exist_ok=True)

    model = DeepLabV3Plus(cfg)
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    model.to(device) 

    if cfg['criterion']['name'] == 'CELoss':
        criterion = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).to(device) 
    elif cfg['criterion']['name'] == 'OHEM':
        criterion = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).to(device) 
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    trainset = EndovisDataset(cfg['dataset'], cfg['data_root'], 'train_l', id_path=labeled_path)
    valset = EndovisDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler = torch.utils.data.RandomSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
                             pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler)
    valsampler = torch.utils.data.RandomSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    previous_best = 0.0
    epoch = -1


    if os.path.exists(os.path.join(save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        model.train()
        total_loss = AverageMeter()

        for i, (img, mask) in enumerate(trainloader):

            img, mask = img.to(device), mask.to(device)

            pred = model(img)

            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())

            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr

            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss.item(), iters)
            
            if (i % (max(2, len(trainloader) // 8)) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mcIoU, iou_class, IoU, mIoU  = evaluate(model, valloader, eval_mode, cfg, device)

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> IoU: {:.2f}'.format(eval_mode, IoU))
            logger.info('***** Evaluation {} ***** >>>> mIoU: {:.2f}'.format(eval_mode, mIoU))
            logger.info('***** Evaluation {} ***** >>>> mcIoU: {:.2f}\n'.format(eval_mode, mcIoU))
            
            writer.add_scalar('eval/IoU', IoU, epoch)
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            writer.add_scalar('eval/mIoU_inst', mcIoU, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = mcIoU > previous_best
        previous_best = max(mcIoU, previous_best)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(save_path, 'best.pth'))


