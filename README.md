## lstm-lm

code all in language_model_with_lstm.ipynb

### disclosure
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
https://nlp.stanford.edu/projects/glove/
https://medium.com/mlearning-ai/load-pre-trained-glove-embeddings-in-torch-nn-embedding-layer-in-under-2-minutes-f5af8f57416a
https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#lstm
https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html

### requirements
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
GPU need more than 16G memory

### examples
LSTM:  
```python
rnn = nn.LSTM(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))
```
### args
change args to run code  
```bash
python code/main.py
```
## glove.6B
https://nlp.stanford.edu/projects/glove/


    