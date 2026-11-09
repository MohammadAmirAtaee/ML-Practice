import pickle
import tarfile
import urllib.request
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .augmentations import augment

URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CLASSES = ('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')


def download_cifar10(root='data') -> Path:
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    folder = root / 'cifar-10-batches-py'
    if folder.exists(): return folder
    archive = root / 'cifar-10-python.tar.gz'
    urllib.request.urlretrieve(URL, archive)
    with tarfile.open(archive) as f: f.extractall(root)
    return folder


class CIFAR10(Dataset):
    def __init__(self, root='data', train=True, download=False, two_views=False, limit=None):
        folder = download_cifar10(root) if download else Path(root) / 'cifar-10-batches-py'
        if not folder.exists(): raise FileNotFoundError(f'CIFAR-10 not found at {folder}. Run: python main.py download')
        files = [folder / f'data_batch_{i}' for i in range(1,6)] if train else [folder / 'test_batch']
        arrays, labels = [], []
        for file in files:
            with file.open('rb') as h:
                item = pickle.load(h, encoding='bytes')
            arrays.append(item[b'data']); labels.extend(item[b'labels'])
        self.images = np.concatenate(arrays).reshape(-1,3,32,32).astype('float32') / 255
        self.labels = np.asarray(labels, dtype='int64')
        if limit: self.images, self.labels = self.images[:limit], self.labels[:limit]
        self.two_views = two_views

    def __len__(self): return len(self.labels)
    def __getitem__(self, index):
        x, y = torch.from_numpy(self.images[index]), int(self.labels[index])
        if self.two_views: return augment(x.clone()), augment(x.clone()), y
        return x, y
