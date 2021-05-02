import torch
from torch.utils.data import Dataset
import numpy as np

RNA_alphabet = ['A', 'C', 'G', 'U']

def letter_to_index(letter):
    return RNA_alphabet.index(letter)

def to_feature(guide_strand, free_energy):
    feature = np.zeros(85)
    feature[-1] = free_energy
    for j, letter in enumerate(guide_strand):
        letter_array = np.zeros(4)
        letter_array[letter_to_index(letter)] = 1
        feature[4*j:4*(j+1)] = letter_array
    return feature

def load_huesken_data(datafile, n_samples=None):
    xy = np.genfromtxt(datafile, delimiter=',', dtype=None, names=True, encoding='utf-8')
    y = np.zeros((xy.size, 1), dtype=int)
    y[:,0] = np.rint(xy['norm_inhibitory_activity'])
    guide_strand = xy['guide_strand']
    free_energy = xy['free_energy']
    x = np.zeros((len(free_energy), 85))
    for i, guide_strand in enumerate(guide_strand):
        feature = to_feature(guide_strand, free_energy[i])
        x[i, :] = feature
    if n_samples:
        x = x[0:n_samples, :]
        y = y[0:n_samples]
    return x, y
