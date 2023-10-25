import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import os
from datetime import datetime
import math
import matplotlib.pyplot as plt

class PennTreebankDataset(Dataset):
    def __init__(self, data_lines, sequence_length, batch_size, vocab, device):
        self.data_lines = data_lines
        self.data = None
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.split_length = None
        self.vocab = vocab
        self.device = device
        self.init_epoch()

    def __len__(self):
        return self.split_length  // self.sequence_length * self.batch_size
    
    def __getitem__(self, idx):
        # Split the long sequence into batch size equal-length sequences
        # For each split sequence, cut them into continuous parts
        # Batch k is composed of the kth part of all the sequences.
        batch_x = idx % self.batch_size
        batch_y = idx // self.batch_size
        inputs = self.data[batch_x * self.split_length + batch_y*self.sequence_length: batch_x * self.split_length + (batch_y+1)*self.sequence_length]
        outputs = self.data[batch_x * self.split_length + batch_y*self.sequence_length + 1: batch_x * self.split_length + (batch_y+1)*self.sequence_length + 1]
        return inputs, outputs
    
    def init_epoch(self):
        self.data = self.shuffle_lines(self.data_lines)
        self.split_length = len(self.data) // self.batch_size

    def shuffle_lines(self, data_lines):
        np.random.shuffle(data_lines)
        data = [token for line in data_lines for token in line]
        return self.tensor_from_tokens(data)
    
    def tensor_from_tokens(self, tokens):
        token_ids = [self.vocab[token] for token in tokens]
        return torch.tensor(token_ids, dtype=torch.long, device=self.device)