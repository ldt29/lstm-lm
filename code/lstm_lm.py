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

# Train and Test an LSTM Language Model
class LSTM_LM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_prob, glove_embeddings, device, tie_weight=False):
        super(LSTM_LM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.device = device

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding.weight.data.copy_(glove_embeddings)
        self.embedding.weight.requires_grad = False
        self.dropout = nn.Dropout(self.dropout_prob)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.num_layers, dropout=self.dropout_prob, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, self.vocab_size)
        if tie_weight:
            self.linear.weight = self.embedding.weight
            self.linear.weight.requires_grad = False
        self.softmax = nn.LogSoftmax(dim=2)


    def forward(self, inputs, hidden):
        embedding = self.embedding(inputs)
        embedding = self.dropout(embedding)
        lstm_out, hidden = self.lstm(embedding, hidden)
        lstm_out = self.dropout(lstm_out)
        outputs = self.linear(lstm_out)
        outputs = self.softmax(outputs)
        return outputs, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().to(self.device),
                weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().to(self.device))
        
    def reuse_hidden(self, hidden):
        return (self.dropout(hidden[0]).detach(), hidden[1].detach())