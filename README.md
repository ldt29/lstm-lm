##

### disclosure
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
https://nlp.stanford.edu/projects/glove/
https://medium.com/mlearning-ai/load-pre-trained-glove-embeddings-in-torch-nn-embedding-layer-in-under-2-minutes-f5af8f57416a
https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#lstm
https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
### requirements
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
GPU need more than 10G memory

### examples
```python
rnn = nn.LSTM(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))
```

### batch

example:
tokens_len = 1000
batch_size = 20
squence_length =10
batch_num = 10

idx = 11

batch_x = idx % batch_size = 11
batch_y = idx // batch_size = 0

    