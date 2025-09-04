import os
import torch
import numpy as np

from glob import glob

from PIL import Image
from torch.utils.data import Dataset

_SUBJECTS = [
    '20200709-subject-01',
    '20200813-subject-02',
    '20200820-subject-03',
    '20200903-subject-04',
    '20200908-subject-05',
    '20200918-subject-06',
    '20200928-subject-07',
    '20201002-subject-08',
    '20201015-subject-09',
    '20201022-subject-10',
]

_SERIALS = [
    '836212060125',
    '839512060362',
    '840412060917',
    '841412060263',
    '932122060857',
    '932122060861',
    '932122061900',
    '932122062010',
]

_YCB_CLASSES = {
     1: '002_master_chef_can',
     2: '003_cracker_box',
     3: '004_sugar_box',
     4: '005_tomato_soup_can',
     5: '006_mustard_bottle',
     6: '007_tuna_fish_can',
     7: '008_pudding_box',
     8: '009_gelatin_box',
     9: '010_potted_meat_can',
    10: '011_banana',
    11: '019_pitcher_base',
    12: '021_bleach_cleanser',
    13: '024_bowl',
    14: '025_mug',
    15: '035_power_drill',
    16: '036_wood_block',
    17: '037_scissors',
    18: '040_large_marker',
    19: '051_large_clamp',
    20: '052_extra_large_clamp',
    21: '061_foam_brick',
}

# TODO: make colour map more specific to the YCB dataset
def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

class DexYCB(Dataset):
    cmap = voc_cmap() # objects (1-21) + hand (22)
    def __init__(
            self, root: str,
            setup: str = 's0', split: str = 'train',
            num_sequences: int = 100,
            subjects: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], # only used if setup = 'custom'
            serials: list[int] = [0, 1, 2, 3, 4, 5, 6, 7], # only used if setup = 'custom'
            transform: torch.nn.Module | None = None
    ):   
        if setup not in ['s0', 's1', 's2', 's3', 'custom']: raise ValueError('Invalid setup')
        if split not in ['train', 'val', 'test']: raise ValueError('Invalid split')       

        self.transform = transform

        # custom setup: specify subjects and serials, and split based on sequence (like below)
        if setup == 'custom':
            subject_ind = subjects
            serial_ind = serials
            if split == 'train':
                sequence_ind = [i for i in range(100) if i % 5 != 4]
            if split == 'val':
                sequence_ind = [i for i in range(100) if i % 5 == 4]
            if split == 'test':
                sequence_ind = [i for i in range(100) if i % 5 == 4]

        # Seen subjects, camera views, grasped objects.
        if setup == 's0':
            if split == 'train':
                subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = [i for i in range(100) if i % 5 != 4]
            if split == 'val':
                subject_ind = [0, 1]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = [i for i in range(100) if i % 5 == 4]
            if split == 'test':
                subject_ind = [2, 3, 4, 5, 6, 7, 8, 9]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = [i for i in range(100) if i % 5 == 4]

        # Unseen subjects.
        if setup == 's1':
            if split == 'train':
                subject_ind = [0, 1, 2, 3, 4, 5, 9]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = list(range(100))
            if split == 'val':
                subject_ind = [6]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = list(range(100))
            if split == 'test':
                subject_ind = [7, 8]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = list(range(100))

        # Unseen camera views.
        if setup == 's2':
            if split == 'train':
                subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                serial_ind = [0, 1, 2, 3, 4, 5]
                sequence_ind = list(range(100))
            if split == 'val':
                subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                serial_ind = [6]
                sequence_ind = list(range(100))
            if split == 'test':
                subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                serial_ind = [7]
                sequence_ind = list(range(100))

        # Unseen grasped objects.
        if setup == 's3':
            if split == 'train':
                subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = [
                i for i in range(100) if i // 5 not in (3, 7, 11, 15, 19)
                ]
            if split == 'val':
                subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = [i for i in range(100) if i // 5 in (3, 19)]
            if split == 'test':
                subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = [i for i in range(100) if i // 5 in (7, 11, 15)]
            subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
            sequence_ind = [i for i in range(100) if i // 5 in (7, 11, 15)]

        if num_sequences != 100: # rescale sequence indices
            sequence_ind = np.array(sequence_ind)
            sequence_ind = np.unique((sequence_ind / (100 / num_sequences)).astype(int)) # divide, round down, and filter unique sequences
            sequence_ind = np.unique((sequence_ind * (100 / num_sequences)).astype(int)) # then multiply back, round down, and filter unique sequences (just in case)
            sequence_ind = sequence_ind.tolist() # convert back to list

        self.image_paths = []
        self.label_paths = []
        for sub_idx in subject_ind:
            subject = _SUBJECTS[sub_idx]
            seq = [
                os.path.join(subject, s)
                for s in sorted(os.listdir(os.path.join(root, subject)))
            ]
            for seq_idx in sequence_ind:
                sequence = seq[seq_idx]
                for ser_idx in serial_ind:
                    serial = _SERIALS[ser_idx]
                    self.image_paths.extend(
                        sorted(glob(os.path.join(root, sequence, serial, 'color_*.jpg')))
                    )
                    self.label_paths.extend(
                        sorted(glob(os.path.join(root, sequence, serial, 'labels_*.npz')))
                    )
    
    def __len__(self):
        return len(self.image_paths)
    
    @classmethod
    def decode_target(cls: 'DexYCB', mask):
        mask[mask == 255] = 22
        return cls.cmap[mask]
    
    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert('RGB')

        # DexYCB stores segmentation data in npz files instead
        target: np.ndarray = np.load(self.label_paths[index])['seg']
        target[target == 255] = 22 # reassign hand ID to 22 instead of 255
        target = Image.fromarray(target) # img and target should be PIL images

        if self.transform is not None:
            img, target = self.transform(img, target)
        
        return img, target