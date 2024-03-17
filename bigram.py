# pytorch=1.12.1 (py3.9_cuda11.3_cudnn8.3.2_0)
import torch
from torch import nn
from torch.nn import functional as F
import os

# hyperparameters
epochs = 0
n_train_split = 0.9
random_seed = 1999
batch_size = 8 # how many independend seq will we process in parallel
block_size = 8 # what is the maximum context length for predictions
eval_interval = 300
eval_iters = 300 # for estimate loss
train_steps = 50000
final_loss = 2.5
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(random_seed)

# load data from wget
def read_data(fpath='input.txt'):
    with open(fpath, 'r') as f:
        text = f.read()
    return text

try:
    text = read_data()
    print(f'Finish readed data. Have {len(text)} charactors.')
except FileNotFoundError:
    os.system('wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')
    text = read_data()
    print(f'Finish downloaded and readed data. Have {len(text)} charactors.')

print("-- Sample Data --", text[:1000], sep='\n')

# encode decode
print("", "-- Encode Decode --", sep='\n')
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("all_chars:", ''.join(chars))
print("vocab_size:", vocab_size)

c2i = {chars[i]:i for i in range(vocab_size)}
i2c = {i:chars[i] for i in range(vocab_size)}
encode = lambda s: [c2i[c] for c in s]
decode = lambda l: [i2c[i] for i in l]

test_enc_dec_text = "Hello World"
print('')
print("test_enc_dec_text:",  test_enc_dec_text)
print('encode:', encode(test_enc_dec_text))
print('decode:', decode(encode(test_enc_dec_text)))

data = torch.tensor(encode(text), dtype=torch.long).to(device=device)
print('data:', data[:100],)
print('')

# train and test split
n_split = int(len(data) * n_train_split)
train_data = data[:n_split]
val_data = data[n_split:]
print("-- Data Spliting --")
print("train_data:", train_data.__len__())
print("val_data:  ", val_data.__len__())

# batch data function
def get_batch(split: str='train'):
    data = train_data if split == 'train' else val_data
    
    ix = torch.randint(len(data) - block_size, size=(batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x, y

xb, yb = get_batch('train')
print("xb:", xb.shape, xb, '', sep='\n')
print("yb:", yb.shape, yb, '', sep='\n')

# simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx):
        # idx are (B,T) tensor of int
        logits = self.token_embedding_table(idx)
        return logits
    
    def generate(self, idx, max_new_token: int):

        for _ in range(max_new_token):
            # forward pass
            logits = self(idx)
            
            # get last time step
            logits = logits[:,-1,:] 

            # get prob
            probs = F.softmax(logits, dim=1)

            # sample from distribution (multinomial)
            samples = torch.multinomial(probs, num_samples=1)

            # append to last time steps
            idx = torch.concat([idx, samples], dim=1)
        return idx

# define loss function
def loss_fn(logits, targets):

    B,T,C = logits.shape
    logits = logits.view(B*T, C)
    targets = targets.view(B*T)

    loss = F.cross_entropy(logits, targets)
    return loss

# estimate loss by eval (train and test)
@torch.no_grad()
def estimate_loss(model: nn.Module):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#  create optimizer
model = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)

model.train()
# training loops
for iter in range(train_steps):

    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step: {iter:02d} | train_loss: {losses['train']:.3f} | test_loss: {losses['val']:.3f}")

    # sample batch data
    xb, yb = get_batch('train')

    # forward pass
    logits = model(xb)

    # cal loss 
    loss = loss_fn(logits, yb)

    # opt zero grad
    optimizer.zero_grad()

    # loss backward
    loss.backward()

    # opt step
    optimizer.step()

    if losses['val'] < final_loss:
        break
    
# generate from the model
y_pred = model.generate(
    torch.zeros(1, 1, dtype=torch.long).to(device),
    max_new_token=1000
).to('cpu').tolist()[0]

generated_text = "".join(decode(y_pred))
print("-- Generate Text --")
print(generated_text)