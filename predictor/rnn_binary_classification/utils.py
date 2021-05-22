import torch
from torch.utils.data import Dataset
import numpy as np

RNA_alphabet = ['A', 'C', 'G', 'U']

class HueskenDataset(Dataset):

    def __init__(self, datafile, transform=None):
        guide_strands, free_energy, y = self._load_huesken_data(datafile)
        self.guide_strands = guide_strands
        self.free_energy = free_energy
        self.y = y
        self.n_samples = y.shape[0]
        self.transform = transform
    
    def __getitem__(self, index):
        sample = self._to_feature(self.guide_strands[index], self.free_energy[index]), self.y[index]
        if self.transform:
            sample = self.transform(sample)
        inputs, labels = sample
        return sample
    
    def __len__(self):
        return self.n_samples
    
    def _letter_to_index(self, letter):
        return RNA_alphabet.index(letter)

    def _to_feature(self, guide_strand, free_energy):
        feature = np.zeros((len(guide_strand), 1, len(RNA_alphabet) + 1))
        for i, letter in enumerate(guide_strand):
            feature[i][0][self._letter_to_index(letter)] = 1
            feature[i][0][-1] = free_energy
        return feature

    def _load_huesken_data(self, datafile):
        xy = np.genfromtxt(datafile, delimiter=',', dtype=None, names=True, encoding='utf-8')
        y = np.zeros((xy.size, 1), dtype=float)
        y[:,0] = np.rint(xy['norm_inhibitory_activity'])
        guide_strands = xy['guide_strand']
        free_energy = xy['free_energy']
        return guide_strands, free_energy, y

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs.astype(np.float32)), torch.from_numpy(targets.astype(np.float32))