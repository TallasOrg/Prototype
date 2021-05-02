import torch
from torch.utils.data import Dataset
import numpy as np

RNA_alphabet = ['A', 'C', 'G', 'U']

class HueskenDataset(Dataset):

    def __init__(self, datafile, transform=None):
        x, y = self._load_huesken_data(datafile)
        self.x = x
        self.y = y
        self.n_samples = x.shape[0]
        self.transform = transform
    
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return self.n_samples
    
    def _letter_to_index(self, letter):
        return RNA_alphabet.index(letter)

    def _to_feature(self, guide_strand, free_energy):
        feature = np.zeros(len(guide_strand) + 1)
        feature[-1] = free_energy
        for i, letter in enumerate(guide_strand):
            feature[i] = self._letter_to_index(letter)
        return feature

    def _load_huesken_data(self, datafile):
        xy = np.genfromtxt(datafile, delimiter=',', dtype=None, names=True, encoding='utf-8')
        y = np.zeros((xy.size, 1), dtype=float)
        y[:,0] = xy['norm_inhibitory_activity']
        guide_strand = xy['guide_strand']
        free_energy = xy['free_energy']
        x = np.zeros((len(free_energy), len(guide_strand[0]) + 1))
        for i, guide_strand in enumerate(guide_strand):
            feature = self._to_feature(guide_strand, free_energy[i])
            x[i, :] = feature
        return x, y

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs.astype(np.float32)), torch.from_numpy(targets.astype(np.float32))