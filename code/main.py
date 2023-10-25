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
from ptb_dataset import PennTreebankDataset
from lstm_lm import LSTM_LM
from rnn_lm import RNN_LM
from train_test import train, test
import arguments 

## change here
# args = arguments.args_shuffled_batching
# args = arguments.args_continuous_batching
# args = arguments.args_continuous_batching_reuse
# args = arguments.args_continuous_batching_reuse_rnn
# args = arguments.args_dropout_len7
# args = arguments.args_layers2
# args = arguments.args_layers3
# args = arguments.args_layers4
# args = arguments.args_weight_tying
# args = arguments.args_lr_decay01
# args = arguments.args_lr_decay05

# best model
args = arguments.args_lr_decay098

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Read the training, validation, and testing files, and replace newlines with the "<eos>" token.
def read_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line.split() + ['<eos>'] for line in lines]
        return lines
ptb_train_lines = read_file('penn-treebank/ptb.train.txt')
ptb_valid_lines = read_file('penn-treebank/ptb.valid.txt')
ptb_test_lines = read_file('penn-treebank/ptb.test.txt')


# Concatenate all articles in the training set into one long sequence.
ptb_train_tokens = [token for line in ptb_train_lines for token in line]


# Initialize the word embeddings with Glove 
print('Loading Glove embeddings...')
vocab = {}
glove_embeddings = []
with open('glove.6B/glove.6B.300d.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip().split()
        vocab[line[0]] = len(vocab)
        glove_embeddings.append([float(x) for x in line[1:]])
print('Done.')
glove_embeddings = torch.tensor(glove_embeddings, dtype=torch.float, device=device)

# For words not found in Glove, use random initializations.
for token in ptb_train_tokens:
    if token not in vocab.keys():
        vocab[token] = len(vocab)
        glove_embeddings = torch.cat((glove_embeddings, torch.randn(1, glove_embeddings.shape[1], device=device)), dim=0)
vocab_size = len(vocab)

## Train 
train_dataLoader = DataLoader(dataset=PennTreebankDataset(ptb_train_lines, args['sequence_length'], args['batch_size'], vocab, device), batch_size=args['batch_size'], shuffle=args['shuffle'])
valid_dataLoader = DataLoader(dataset=PennTreebankDataset(ptb_valid_lines, args['sequence_length'], args['batch_size'], vocab, device), batch_size=args['batch_size'], shuffle=False)
test_dataLoader = DataLoader(dataset=PennTreebankDataset(ptb_test_lines, args['sequence_length'], args['batch_size'], vocab, device), batch_size=args['batch_size'], shuffle=False)

model = LSTM_LM(vocab_size, args['embedding_dim'], args['hidden_dim'], args['num_layers'], args['dropout_prob'], glove_embeddings, device, args['tie_weight'])
model.to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=args['lr'])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma = args['gamma'])
train_losses = []
valid_losses = []
for epoch in range(args['epochs']):
    train_dataLoader.dataset.init_epoch()
    train_loss = train(model, train_dataLoader, criterion, optimizer, device, reuse=args['reuse_hidden'])
    valid_loss = test(model, valid_dataLoader, criterion, device)
    scheduler.step()
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    valid_perplexity = math.exp(valid_loss)
    print('Epoch: {}, Train Loss: {}, Valid Loss: {}, Valid Perplexity: {}'.format(epoch, train_loss, valid_loss, valid_perplexity))

## plot the loss curve and save the model
plt.plot(train_losses, label='train loss')
plt.plot(valid_losses, label='valid loss')
plt.legend()
## save with utc time
torch.save(model.state_dict(), args['save_path'] + 'models/LSTM_LM_' + datetime.utcnow().strftime("%Y-%m-%d-%H%MZ") + '.pt')
plt.savefig(args['save_path'] + 'curves/LSTM_loss_curve_' + datetime.utcnow().strftime("%Y-%m-%d-%H%MZ") + '.png')

## Test
criterion = nn.NLLLoss()
test_loss = test(model, test_dataLoader, criterion, device)
valid_loss = test(model, valid_dataLoader, criterion, device)
test_perplexity = math.exp(test_loss)
valid_perplexity = math.exp(valid_loss)
print('Test Loss: {:.4f}, Valid Loss: {:.4f}, Test Perplexity: {:.4f}, Valid Perplexity: {:.4f}'.format(test_loss, valid_loss, test_perplexity, valid_perplexity))