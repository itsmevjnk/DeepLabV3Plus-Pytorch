# The following script tests the trained models and reports accuracy statistics to results.csv.
# These files in checkpoints/ are expected to exist:
#  - best_mobilenet_scratch.pth
#  - best_mobilenet_separable.pth
#  - best_mobilenet_transfer.pth
#  - best_mobilenet_transfer_single.pth
#  - best_resnet50_scratch.pth
# The training script, as well as the network definitions and other support utilities, are available at:
#   https://github.com/itsmevjnk/DeepLabV3Plus-Pytorch

import torch
from torch.utils import data
from torch import nn

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import network.modeling
from datasets import DexYCB

import utils
from utils import ext_transforms as et

from tqdm import tqdm

import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device to run test on: {device}')

BATCH_SIZE = 24 # change this depending on our VRAM

NUM_CLASSES = 23
MODELS = ['resnet50_scratch', 'mobilenet_scratch', 'mobilenet_separable', 'mobilenet_transfer', 'mobilenet_transfer_single']

test_transforms = et.ExtCompose([
    # et.ExtResize( 512 ),
    et.ExtToTensor(),
    et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
])

test_data = DexYCB(
    root='../DexYCB',
    setup='custom', subjects=[0,2,4,6,8], frame_stride=8, # same as training
    split='test',
    transform=test_transforms
)

test_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f'Testing dataset: {len(test_data)} images')

results = []

try:
    for model_name in MODELS:
        fname = os.path.join('checkpoints', f'best_{model_name}.pth')
        print(f'Testing checkpoint {fname}.')

        entry = {
            'Model': model_name
        }

        # set up model
        if model_name.startswith('mobilenet'): # MobileNetV2 backbone
            model = network.modeling.deeplabv3plus_mobilenet(num_classes=NUM_CLASSES)
        else: # ResNet50 backbone
            model = network.modeling.deeplabv3plus_resnet50(num_classes=NUM_CLASSES)

        if model_name.endswith('separable'): # separable convolution
            network.convert_to_separable_conv(model.classifier)

        # load checkpoint
        checkpoint = torch.load(fname, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        model = nn.DataParallel(model)
        model.to(device)

        # metrics for each batch
        pix_acc = [] # pixel-wise accuracy
        cls_acc_isect = np.zeros(NUM_CLASSES) # class accuracy (and also IoU): intersecting pixel sum
        cls_acc_gt = np.zeros(NUM_CLASSES) # class accuracy: ground truth pixel sum
        iou_union = np.zeros(NUM_CLASSES) # IoU: union pixel sum

        with torch.no_grad():
            model = model.eval() # run in evaluation mode

            for i, (images, labels) in tqdm(enumerate(test_loader)):
                images = images.to(device, dtype=torch.float32)
                # labels = labels.to(device, dtype=torch.long)

                outputs = model(images)

                preds = outputs.detach().max(dim=1)[1].cpu().numpy() # collapse down to [N, H, W]
                labels = labels.cpu().numpy()

                # compute pixel-wise accuracy
                correct = (preds == labels).sum()
                total = np.prod(labels.shape)
                pix_acc.append(float('nan') if total == 0 else correct / total)

                # compute per-class metrics
                for cls in range(NUM_CLASSES):
                    preds_msk = (preds == cls)
                    labels_msk = (labels == cls)

                    # per-class accuracy
                    intersection = np.logical_and(preds_msk, labels_msk).sum() # also true positive
                    gt_count = labels_msk.sum() # ground truth pixel count
                    cls_acc_isect[cls] += intersection
                    cls_acc_gt[cls] += gt_count

                    # IoU
                    union = np.logical_or(preds_msk, labels_msk).sum()
                    iou_union[cls] += union

                # if i >= 1: break

        entry['Pixel accuracy'] = np.nanmean(pix_acc)
        cls_acc = cls_acc_isect / cls_acc_gt # class accuracy (per class)
        entry['Mean class accuracy'] = np.nanmean(cls_acc)
        iou = cls_acc_isect / iou_union # IoU (per class)
        entry['Mean IoU'] = np.nanmean(iou) # mean IoU over entire dataset
        for cls in range(NUM_CLASSES):
            entry[f'{cls} accuracy'] = cls_acc[cls]
            entry[f'{cls} IoU'] = iou[cls]

        results.append(entry)
        print(f' - Pixel accuracy: {entry["Pixel accuracy"]}, mean class accuracy: {entry["Mean class accuracy"]}, mean IoU: {entry["Mean IoU"]}')

finally: # in case we crash
    results_df = pd.DataFrame(results)
    results_df.to_csv('results.csv')


