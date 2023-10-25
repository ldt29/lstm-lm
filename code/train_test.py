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

# Train the model
def train(model, train_dataLoader, criterion, optimizer, device, reuse=True):
    model.train()
    epoch_loss = 0
    hidden = model.init_hidden(train_dataLoader.dataset.batch_size)
    for inputs, targets in train_dataLoader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        if reuse:
            hidden = model.reuse_hidden(hidden)
        else:
            hidden = model.init_hidden(train_dataLoader.dataset.batch_size)
        model.zero_grad()
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs.view(-1, model.vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_dataLoader)

# Test the model
def test(model, test_dataLoader, criterion, device):
    model.eval()
    epoch_loss = 0
    hidden = model.init_hidden(test_dataLoader.dataset.batch_size)
    with torch.no_grad():
        for inputs, targets in test_dataLoader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs.view(-1, model.vocab_size), targets.view(-1))
            epoch_loss += loss.item()
    return epoch_loss / len(test_dataLoader)