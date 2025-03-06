# import torch
import json
import os
import glob
from model import LLM

chars = """\t\n !"$%&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]`abcdefghijklmnopqrstuvwxyz|~–—‘’“”…"""
print(chars[0], chars[1], chars[2], chars[3])
ctoi = {ch:i for i, ch in enumerate(chars)}
itoc = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [ctoi[c] for c in s]
decode = lambda l: "".join([itoc[x] for x in l])

print(encode("hello"))
print(decode([0, 1, 66, 63, 70, 70, 73, 0, 1]))

shard_filenames = sorted(glob.glob(os.path.join('TinyStories', "*.json")))
with open(shard_filenames[0], "r") as f:
    data = json.load(f)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
batch_size = 4
def get_batch(split):
    # generate a small bunch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
# hyperparameters
batch_size = 64 # independent sequences processed in parallel
block_size = 256 # max context length
max_iters = 3000 
eval_interval = 100
learning_rate = 1e-3
eval_iters = 200
n_embd = 384
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.0

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Config:
    n_embd = 384
    n_embed = 384
    n_head = 6
    n_layer = 6
    attention_dropout = 0.0
    expert_dropout = 0.0
    head_size = 64

config = Config()

model = LLM(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            X = X.to(device)
            Y = Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

for iter in range(max_iters):

    # print the loss on train and val datasets
    if iter % 100 == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')
    xb = xb.to(device)
    yb = yb.to(device)

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def train(args, model, config, train_data, val_data):
    block_size = config.block_size
    batch_size = config.batch_size
    max_iters = config.max_iters
    eval_interval = config.eval_interval
    learning_rate = config.learning_rate
    eval_iters = config.eval_iters  

def main(args):

    with open(args.config, "r") as f:
        config = json.load(f)

    shard_filenames = sorted(glob.glob(os.path.join('TinyStories', "*.json")))
    with open(shard_filenames[0], "r") as f:
        data = json.load(f)

    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]
    model = LLM(config).to(device)

    if args.action == "train":
        train(args, model, config, train_data, val_data)
    elif args.action == "eval:"
        eval(args, model, config, val_data)
    else:
        raise ValueError(f"Invalid action: {args.action}")
    

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--action", type=str, default="train")
    args = parser.parse_args()

    main(config)