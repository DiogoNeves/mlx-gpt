"""Bigram model implementation."""

from enum import Enum

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from data import fetch_input_data


class DataSplit(Enum):
    TRAIN = "train"
    VAL = "val"


# Hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_epochs = 3000
learning_rate = 1e-2
eval_interval = 300
eval_iters = 200
m_embd = 32
# ---------------

mx.random.seed(1337)

# Data
INPUT_DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
                  "master/data/tinyshakespeare/input.txt")
input_text = fetch_input_data(INPUT_DATA_URL, "input.txt")
# ----


# here are all the unique characters that occur in this text
chars = sorted(list(set(input_text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# encoder: take a string, output a list of integers
encode = lambda s: [stoi[c] for c in s]
# decoder: take a list of integers, output a string
decode = lambda l: ''.join([itos[i] for i in l])

# train and test splits
data = mx.array(encode(input_text), dtype=mx.int64)
n = int(0.9 * len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


def get_batch(split: DataSplit) -> tuple[mx.array, mx.array]:
    """Generate a small batch of data of inputs x and targets y"""
    data = train_data if split == DataSplit.TRAIN else val_data
    ix = mx.random.randint(0, len(data) - block_size, [batch_size])
    # gets `batch_size` blocks stacked
    x = mx.stack([data[i.item():i.item() + block_size] for i in ix])
    # it's shifted to compute the target vectorized
    y = mx.stack([data[i.item() + 1:i.item() + block_size + 1] for i in ix])
    return x, y


class BigramLanguageModel(nn.Module):
    """Super-simple Bigram model."""
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, m_embd)
        self.lm_head = nn.Linear(m_embd, vocab_size)
    
    def __call__(self, idx: mx.array) -> mx.array:
        token_embeddings = self.token_embedding(idx)  # (B, T, m_embd)
        return self.lm_head(token_embeddings)  # (B, T, vocab_size)
    
    def generate(self, idx: mx.array, max_new_tokens: int) -> mx.array:
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]
            # sample from the distribution
            idx_next = mx.random.categorical(logits, num_samples=1, axis=-1)
            # append sampled index to the running sequence
            idx = mx.concatenate([idx, idx_next], axis=1)
        # this is actually going to return 101 tokens since the input counts
        return idx


def loss_fn(model: nn.Module, x: mx.array, y: mx.array) -> mx.array:
    return mx.mean(nn.losses.cross_entropy(model(x), y))


def estimate_loss():
    out = {}
    for split in [DataSplit.TRAIN, DataSplit.VAL]:
        losses = mx.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            loss = loss_fn(model, xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out


model = BigramLanguageModel()
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
optimizer = optim.AdamW(learning_rate=learning_rate)

for epoch in range(max_epochs):
    # every once in a while evaluate the loss on train and val sets
    if epoch % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {epoch}: train loss {losses[DataSplit.TRAIN]}, val loss {losses[DataSplit.VAL]}")

    xb, yb = get_batch(DataSplit.TRAIN)
    loss, grads = loss_and_grad_fn(model, xb, yb)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)


# generate from the model
context = mx.zeros((1, 1), dtype=mx.int64)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
