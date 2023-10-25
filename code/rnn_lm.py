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

# Train and Test an RNN Language Model
class RNN_LM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_prob, glove_embeddings, device):
        super(RNN_LM, self).__init__()
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
        self.rnn = nn.RNN(self.embedding_dim, self.hidden_dim, self.num_layers, dropout=self.dropout_prob, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)
        

    def forward(self, inputs, hidden):
        embedding = self.embedding(inputs)
        embedding = self.dropout(embedding)
        rnn_out, hidden = self.rnn(embedding, hidden)
        rnn_out = self.dropout(rnn_out)
        outputs = self.linear(rnn_out)
        outputs = self.softmax(outputs)
        return outputs, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().to(self.device)
    
    def reuse_hidden(self, hidden):
        return hidden.detach()