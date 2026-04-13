"""
SynPAIN Dataset Loader
======================
Returns pair (neutral + expressive) samples with demographic metadata.

Expected metadata CSV schema (minimum required columns):
    neutral_path, expr_path, label[, split, age_group, gender, ethnicity, subject_id]

- neutral_path / expr_path: relative to root_path
- label: integer 0 (no pain) or 1 (pain)
- split: 'train' | 'val' | 'test'  (if absent, ALL rows go to every split)
- demographic columns are optional; missing ones default to 'unknown'

Compatible with the existing image_train / image_test transforms from utils.py.
"""

import csv
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import image_train, image_test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


# Demographic keys expected (safe 'unknown' placeholder when absent)
_DEMO_KEYS = ('age_group', 'gender', 'ethnicity', 'subject_id')


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class SynPAIN(Dataset):
    """
    SynPAIN pair-input dataset.

    Args:
        root_path (str):       Root directory; image paths in CSV are relative to this.
        split (str):           'train', 'val', or 'test'.
        metadata_file (str):   CSV filename inside root_path (default 'metadata.csv').
        transform_train:       Callable matching utils.image_train signature
                               (img, flip, offset_x, offset_y) → tensor.
        transform_test:        Callable matching utils.image_test signature
                               (img) → tensor.
        crop_size (int):       Spatial crop size for image_train augmentation.
        loader:                Image loader (default PIL RGB loader).
    """

    def __init__(
        self,
        root_path,
        split='train',
        metadata_file='metadata.csv',
        transform_train=None,
        transform_test=None,
        crop_size=224,
        loader=_pil_loader,
    ):
        assert split in ('train', 'val', 'test'), f"Invalid split '{split}'"
        self.root_path  = root_path
        self.split      = split
        self.is_train   = (split == 'train')
        self.crop_size  = crop_size
        self.loader     = loader

        # Fall back to default transforms if not supplied
        self.transform_train = transform_train or image_train(crop_size=crop_size)
        self.transform_test  = transform_test  or image_test(crop_size=crop_size)

        meta_path = os.path.join(root_path, metadata_file)
        self.data_list = self._load_metadata(meta_path, split)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_metadata(self, meta_path, split):
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"SynPAIN metadata not found: {meta_path}\n"
                "Expected a CSV with columns: neutral_path, expr_path, label"
                "[, split, age_group, gender, ethnicity, subject_id]"
            )

        samples = []
        with open(meta_path, newline='') as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            for row in reader:
                # Filter by split column when present
                if 'split' in fields and row.get('split', split) != split:
                    continue
                sample = {
                    'neutral_path': row['neutral_path'].strip(),
                    'expr_path':    row['expr_path'].strip(),
                    'label':        int(row['label']),
                }
                # Parse demographics with safe fallback
                for key in _DEMO_KEYS:
                    sample[key] = row.get(key, 'unknown').strip() if key in fields else 'unknown'
                samples.append(sample)

        if not samples:
            raise RuntimeError(
                f"No samples found for split='{split}' in {meta_path}. "
                "Check the 'split' column values."
            )
        return samples

    def _apply_pair_transform(self, img_neu, img_expr):
        """
        Apply IDENTICAL random augmentation to both images so spatial
        correspondence is preserved between the neutral and expressive frame.
        """
        if self.is_train:
            w, h = img_neu.size
            offset_x = random.randint(0, max(0, w - self.crop_size))
            offset_y = random.randint(0, max(0, h - self.crop_size))
            flip = random.randint(0, 1)
            t_neu  = self.transform_train(img_neu,  flip, offset_x, offset_y)
            t_expr = self.transform_train(img_expr, flip, offset_x, offset_y)
        else:
            t_neu  = self.transform_test(img_neu)
            t_expr = self.transform_test(img_expr)
        return t_neu, t_expr

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __getitem__(self, index):
        s = self.data_list[index]

        img_neu  = self.loader(os.path.join(self.root_path, s['neutral_path']))
        img_expr = self.loader(os.path.join(self.root_path, s['expr_path']))

        # Synchronised transforms keep spatial alignment between the pair
        x_neu, x_expr = self._apply_pair_transform(img_neu, img_expr)

        return {
            'x_neu':      x_neu,           # [3, H, W] tensor
            'x_expr':     x_expr,          # [3, H, W] tensor
            'label':      s['label'],      # int 0/1
            'age_group':  s['age_group'],  # str or 'unknown'
            'gender':     s['gender'],
            'ethnicity':  s['ethnicity'],
            'subject_id': s['subject_id'],
        }

    def __len__(self):
        return len(self.data_list)


# ---------------------------------------------------------------------------
# Custom collate — stacks tensors, keeps string demographics as lists
# ---------------------------------------------------------------------------

def synpain_collate_fn(batch):
    """
    Collate function for DataLoader.
    Stacks tensor fields; keeps demographic strings as plain lists
    (torch.stack cannot handle variable-length strings).
    """
    return {
        'x_neu':      torch.stack([b['x_neu']  for b in batch]),   # [B, 3, H, W]
        'x_expr':     torch.stack([b['x_expr'] for b in batch]),   # [B, 3, H, W]
        'label':      torch.tensor([b['label'] for b in batch], dtype=torch.long),
        'age_group':  [b['age_group']  for b in batch],
        'gender':     [b['gender']     for b in batch],
        'ethnicity':  [b['ethnicity']  for b in batch],
        'subject_id': [b['subject_id'] for b in batch],
    }


# ---------------------------------------------------------------------------
# Single-image variant for full-model pain pretraining
# ---------------------------------------------------------------------------

class SynPAINSingle(Dataset):
    """SynPAIN single-image dataset.

    Returns ``(img, one_hot_label)`` so it is a drop-in replacement for the
    UNBC stage-3 loader, allowing ``pain_estimation_full*.py`` to pretrain on
    SynPAIN with binary pain supervision (no AU labels, no pair input).

    Only the *expressive* frame (`expr_path`) is used; the neutral frame and
    demographic columns are ignored. The label column is the pain status of
    the expressive frame, identical to the value the pair-input loader uses.
    """

    def __init__(
        self,
        root_path,
        split='train',
        metadata_file='metadata.csv',
        transform=None,
        crop_size=224,
        loader=_pil_loader,
    ):
        assert split in ('train', 'val', 'test'), f"Invalid split '{split}'"
        self.root_path = root_path
        self.split     = split
        self.is_train  = (split == 'train')
        self.crop_size = crop_size
        self.loader    = loader
        self._transform = transform  # callable: train uses (img, flip, ox, oy); test uses (img,)

        meta_path = os.path.join(root_path, metadata_file)
        self.data_list = self._load_metadata(meta_path, split)

    def _load_metadata(self, meta_path, split):
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"SynPAIN metadata not found: {meta_path}")
        samples = []
        with open(meta_path, newline='') as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            for row in reader:
                if 'split' in fields and row.get('split', split) != split:
                    continue
                samples.append({
                    'expr_path': row['expr_path'].strip(),
                    'label':     int(row['label']),
                })
        if not samples:
            raise RuntimeError(f"No samples for split='{split}' in {meta_path}")
        return samples

    def __getitem__(self, index):
        s = self.data_list[index]
        img = self.loader(os.path.join(self.root_path, s['expr_path']))
        if self.is_train:
            w, h = img.size
            offset_y = random.randint(0, max(0, h - self.crop_size))
            offset_x = random.randint(0, max(0, w - self.crop_size))
            flip = random.randint(0, 1)
            if self._transform is not None:
                img = self._transform(img, flip, offset_x, offset_y)
        else:
            if self._transform is not None:
                img = self._transform(img)
        # One-hot binary label, matching the UNBC stage-3 pspi format.
        label = [1.0, 0.0] if s['label'] == 0 else [0.0, 1.0]
        return img, torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.data_list)
