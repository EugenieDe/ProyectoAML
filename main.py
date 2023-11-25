import argparse
from UniMatch.unimatch import unimatch
import os
import yaml
from UniMatch.model.semseg.deeplabv3plus import DeepLabV3Plus
import torch
from torch.optim import SGD
from torchvision import transforms
from UniMatch.dataset.transform import resize, normalize
import numpy as np 
from UniMatch.util.utils import intersectionAndUnion, AverageMeter
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
from UniMatch.supervised import supervised

parser = argparse.ArgumentParser(description='Proyecto AML')
parser.add_argument('--config', type=str, default = "/home/eugenie/These/ProyectoAML/UniMatch/configs/endovis2018.yaml")
parser.add_argument('--split', type=str, default = "1_2", choices=['1_2', '1_4', '1_8', 'all'])
parser.add_argument('--save_path', type=str, default= "exp/endovis2018/test/")
parser.add_argument('--device', default='cuda:3')
parser.add_argument('--supervision', type=str, default='semi', choices=['semi', 'fully'])
parser.add_argument('--mode', type=str, default='test', choices=['test', 'demo'])
parser.add_argument('--img', type=str, default='seq_9_frame018.png')

def evaluate(img, mask, cfg, device, model):

    model.eval()

    img, mask = resize(img, mask)
    img_norm, mask = normalize(img, mask)
    img_norm = img_norm.unsqueeze(0)
    mask = mask.unsqueeze(0)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    i=0
    mIoU=0

    with torch.no_grad():
        i+=1
        img_norm = img_norm.to(device) 
        pred = model(img_norm)
        mask = mask.to(device)

        pred = pred.argmax(dim=1)
        intersection, union, _ = intersectionAndUnion(pred.cpu().numpy(), mask.cpu().numpy(), cfg['nclass'], 255)
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
        mIoU += interc / (unc+ 1e-10)

        
        visualize(img, pred, mask)

    mIoU = 100*mIoU
    
    return  mIoU

def visualize(img, pred, mask):
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
    
    os.makedirs('visualization_demo', exist_ok=True)
    plt.savefig('visualization_demo/demo.png', dpi=400, bbox_inches='tight')
    plt.close()

def main():
    args = parser.parse_args()
    if args.mode == 'test':
        save_path = os.join.path(args.save_path, args.supervision, args.split)
        if args.supervision == "semi":
            if args.split == 'all':
                raise NotImplementedError("In the semi-supervised mode, the available splits are 1_2, 1_4 and 1_8.")
            labeled_id_path = os.path.join("/home/eugenie/These/ProyectoAML/UniMatch/splits/endovis2018/unsorted/", args.split, "labeled.txt")
            unlabeled_id_path = os.path.join("/home/eugenie/These/ProyectoAML/UniMatch/splits/endovis2018/unsorted/", args.split, "unlabeled.txt")
            unimatch(args, labeled_id_path, unlabeled_id_path, save_path)
        else:
            labeled_id_path = "/home/eugenie/These/ProyectoAML/UniMatch/splits/endovis2018/unsorted/train.txt"
            supervised(args, labeled_id_path, save_path)
    elif args.mode == "demo":
        img_name = args.img
        img_root = "/home/eugenie/These/data/endovis2018/val/images"
        mask_root = "/home/eugenie/These/data/endovis2018/val/annotations"
        img = Image.open(os.path.join(img_root, img_name)).convert('RGB')
        mask = np.array(Image.open(os.path.join(mask_root, img_name)))
        
        device = args.device
        cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

        os.makedirs(args.save_path, exist_ok=True)

        model = DeepLabV3Plus(cfg)
        
        optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                        {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                        'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

        model.to(device)

        if args.supervision == "semi":
            checkpoint_path = os.path.join("/home/eugenie/These/UniMatch/exp/endovis2018/unimatch/base/r101_OHEM", args.split, "seed")
        else:
            checkpoint_path = os.path.join("/home/eugenie/These/UniMatch/exp/endovis2018/supervised/base/r101", args.split)
        
        if os.path.exists(os.path.join(checkpoint_path, 'best.pth')):
            checkpoint = torch.load(os.path.join(checkpoint_path, 'best.pth'))
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            mIoU = evaluate(img, mask, cfg, device, model)

            print(f"Image {img_name} >>> mIoU: {mIoU}")
        else:
            raise NameError("Make sure that the checkpoints you try to access exists.")

    else:
        raise NotImplementedError("The mode should either be demo or test.")
    return 0

if __name__ == '__main__':
    main()