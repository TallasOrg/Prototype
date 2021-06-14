import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset
import numpy as np

smiles_alphabet = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
    '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
    '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n', ' ']

class HueskenDataset(Dataset):

    def __init__(self, datafile, transform=None):
        guide_strands, free_energy, y = self._load_huesken_data(datafile)
        self.guide_strands = guide_strands
        self.free_energy = free_energy
        self.mean = np.mean(free_energy)
        self.min = np.min(free_energy)
        self.max = np.max(free_energy)
        self.y = y
        self.n_samples = y.shape[0]
        self.transform = transform
    
    def __getitem__(self, index):
        sample = self._to_feature(self.guide_strands[index], self.free_energy[index]), self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return self.n_samples
    
    def _letter_to_index(self, letter):
        return smiles_alphabet.index(letter)

    def _to_feature(self, guide_strand, free_energy):
        feature = np.zeros((len(guide_strand), len(smiles_alphabet) + 1))
        for i, letter in enumerate(guide_strand):
            feature[i][self._letter_to_index(letter)] = 1
            feature[i][-1] = (free_energy - self.mean)/(self.max - self.min)
        return feature

    def _load_huesken_data(self, datafile):
        xy = np.genfromtxt(datafile, delimiter=',', dtype=None, names=True, encoding='utf-8')
        y = np.zeros((xy.size, 1), dtype=float)
        y[:,0] = xy['norm_inhibitory_activity']
        guide_strands = []
        for strand in xy['guide_strand']:
            smiles_strand = ''
            for letter in strand:
                smiles_letter = self._alphabet_to_smiles(letter)
                smiles_strand += smiles_letter
            guide_strands.append(smiles_strand)
        free_energy = xy['free_energy']
        return guide_strands, free_energy, y
    
    def _alphabet_to_smiles(self, letter):
        if letter == 'A':
            return 'C1=NC2=NC=NC(=C2N1)N'
        elif letter == 'C':
            return 'C1=C(NC(=O)N=C1)N'
        elif letter == 'G':
            return 'C1=NC2=C(N1)C(=O)NC(=N2)N'
        elif letter == 'U':
            return 'C1=CNC(=O)NC1=O'
        else:
            return ''

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs.astype(np.float32)), torch.from_numpy(targets.astype(np.float32))

def collate_fn(batch):
    features, labels, lengths = zip(*[(a, b, a.size(0)) for (a,b) in sorted(batch, key=lambda x: x[0].size(0), reverse=True)])
    max_len, n_feats = features[0].size()
    features = [torch.cat((f, torch.zeros(max_len - f.size(0), n_feats)), 0) if f.size(0) != max_len else f for f in features]
    features = torch.stack(features, 0)
    labels = torch.stack(labels, 0)
    return features, labels
