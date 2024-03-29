# pytorch=1.12.1 (py3.9_cuda11.3_cudnn8.3.2_0)
import torch
from torch import nn
from torch.nn import functional as F
import os

# hyperparameters
epochs = 0
n_train_split = 0.9
random_seed = 1999
batch_size = 1024 # how many independend seq will we process in parallel
block_size = 128 # what is the maximum context length for predictions
eval_interval = 300
eval_iters = 300 # for estimate loss
train_steps = 50000
final_loss = 1.0
learning_rate = 1e-3
n_emb = 384 # number of embedding
n_head = 6 # number of head
n_blocks = 6
dropout = 0.2
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

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        # register a buffer that not a model parameters
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei =  q @ k.transpose(-2, -1)
        wei = wei * (T **-0.5) # scaled attention
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float('-inf')
        )

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out 

class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_size * num_heads, n_emb)

    def forward(self, x):
        out = torch.cat(
            [h(x) for h in self.heads], dim=-1
        )
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    
    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.net(x)
        return x

class Block(nn.Module):

    def __init__(self, n_emb, n_head):
        super().__init__()
        head_size = n_emb // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
        self.pos_embedding_table = nn.Embedding(block_size, n_emb)
        
        # Blocks
        self.block = nn.Sequential(*[Block(n_emb=n_emb, n_head=n_head) for _ in range(n_blocks)])
        self.ln_final = nn.LayerNorm(n_emb)
        
        # LM Head
        self.lm_head = nn.Linear(n_emb, vocab_size)

    def forward(self, idx):
        # idx are (B,T) tensor of int
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.pos_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb
        x = self.block(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
    
    def generate(self, idx, max_new_token: int):

        for _ in range(max_new_token):
            # crop idx to the last of the blocksize token
            idx_cond = idx[:, -block_size:] 

            # forward pass
            logits = self(idx_cond)
            
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
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)

model.train()
# training loops
for iter in range(train_steps):

    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        loss_printed = f"step: {iter:02d} | train_loss: {losses['train']:.3f} | test_loss: {losses['val']:.3f}"
        print(loss_printed)

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
    max_new_token=10000
).to('cpu').tolist()[0]

generated_text = "".join(decode(y_pred))
print("-- Generate Text --")
print(generated_text)
with open('generated.txt', 'w') as f:
    f.write(loss_printed)
    f.write('\n')
    f.write(generated_text)