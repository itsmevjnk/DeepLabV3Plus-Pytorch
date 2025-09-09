# %%
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

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device to run test on: {device}')

# %%
BACKBONES = {
    'mobilenet': (network.modeling.deeplabv3plus_mobilenet, 'MobileNet'),
    'resnet50': (network.modeling.deeplabv3plus_resnet50, 'ResNet50')
}

TRAINING_TYPES = {
    '': 'Scratch',
    'separable': 'Scratch (with separable convolution)',
    'transfer': 'Transfer (updating all layers)',
    'transfer_012': 'Transfer (final layer only)',
}

# %%
BATCH_SIZE=12 # change depending on VRAM

# %%
test_transforms = et.ExtCompose([
    # et.ExtResize( 512 ),
    et.ExtToTensor(),
    et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
])

test_data = DexYCB(
    root='../DexYCB',
    setup='s0', split='test',
    num_sequences=10,
    transform=test_transforms
)

test_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f'Testing dataset: {len(test_data)} images')

NUM_CLASSES = 23

# %%
results = []

try:
    # %%
    for backbone, (model_fn, backbone_name) in BACKBONES.items():
        for train, train_name in TRAINING_TYPES.items():
            fname = f'checkpoints/best_{backbone}{f"_{train}" if len(train) > 0 else ""}.pth'
            print(f'Testing {backbone_name} - {train_name} ({fname}).')

            entry = {
                'Backbone': backbone_name,
                'Training type': train_name
            }

            # set up model
            model = model_fn(num_classes=23)
            if train == 'separable': # separable convolution
                network.convert_to_separable_conv(model.classifier)
            utils.set_bn_momentum(model.backbone, momentum=0.01)

            # load checkpoint
            checkpoint = torch.load(fname, map_location=torch.device('cpu'), weights_only=False)
            model.load_state_dict(checkpoint['model_state'])
            model = nn.DataParallel(model)
            model.to(device)

            # metrics for each batch
            pix_acc = [] # pixel-wise accuracy
            cls_acc = [[] for i in range(NUM_CLASSES)] # class accuracy
            iou = [[] for i in range(NUM_CLASSES)] # IoU for each class

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
                        cls_acc[cls].append(float('nan') if gt_count == 0 else intersection / gt_count)

                        # IoU
                        union = np.logical_or(preds_msk, labels_msk).sum()
                        iou[cls].append(float('nan') if union == 0 else intersection / union)

                    # if i >= 1: break

            entry['Pixel accuracy'] = np.nanmean(pix_acc)
            cls_acc = np.nanmean(cls_acc, axis=1) # average per class
            iou = np.nanmean(iou, axis=1)
            entry['Mean class accuracy'] = np.nanmean(cls_acc)
            entry['Mean IoU'] = np.nanmean(iou)
            for cls in range(NUM_CLASSES):
                entry[f'{cls} accuracy'] = cls_acc[cls]
                entry[f'{cls} IoU'] = iou[cls]

            results.append(entry)
            print(f' - Pixel accuracy: {entry["Pixel accuracy"]}, mean class accuracy: {entry["Mean class accuracy"]}, mean IoU: {entry["Mean IoU"]}')

finally: # in case we crash
    # %%
    results_df = pd.DataFrame(results)

    # results_df

    # %%
    results_df.to_csv('results.csv')


