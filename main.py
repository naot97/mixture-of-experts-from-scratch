# import torch
import json
import os
import glob
from model import LLM

def get_tokenizer(vocab_file):
    with open(vocab_file, "r") as f:
        chars = f.read().strip()

    print(chars)
    print(f"Vocab size: {len(chars)}")
    ctoi = {ch:i for i, ch in enumerate(chars)}
    itoc = {i:ch for i,ch in enumerate(chars)}

    encode = lambda s: [ctoi[c] for c in s]
    decode = lambda l: "".join([itoc[x] for x in l])
    return chars, encode, decode

def get_batch(split, data, batch_size, block_size, encode):
    # generate a small bunch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x, y

# xb, yb = get_batch('train')
# # hyperparameters
# batch_size = 64 # independent sequences processed in parallel
# block_size = 256 # max context length
# max_iters = 3000 
# eval_interval = 100
# learning_rate = 1e-3
# eval_iters = 200
# n_embd = 384
# n_embed = 384
# n_head = 6
# n_layer = 6
# dropout = 0.0

# # set device
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# class Config:
#     n_embd = 384
#     n_embed = 384
#     n_head = 6
#     n_layer = 6
#     attention_dropout = 0.0
#     expert_dropout = 0.0
#     head_size = 64

# config = Config()

# model = LLM(config).to(device)

@torch.no_grad()
def eval(args, model, config, val_data, encode):
    batch_size = config.batch_size
    block_size = config.block_size
    eval_iters = config.eval_iters
    
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(val_data, batch_size, block_size, encode)
        X = X.to(device)
        Y = Y.to(device)
        logits, loss = model(X, Y)
        losses[k] = loss.item()

    return losses.mean().item()


def train(args, model, config, train_data, val_data, encode):
    batch_size = config.batch_size
    block_size = config.block_size
    max_iters = config.max_iters
    eval_interval = config.eval_interval
    learning_rate = config.learning_rate
    eval_iters = config.eval_iters  
    device = model.device
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
    model.train()
    
    for iter in range(max_iters):
        

        # sample a batch of data
        xb, yb = get_batch('train')
        xb = xb.to(device)
        yb = yb.to(device)

        # evaluate the loss
        loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # print the loss on train and val datasets
        if iter % 100 == 0 or iter == max_iters - 1:
            losses = eval(args, model, config, val_data, encode)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            model.train()


def main(args):

    with open(args.config, "r") as f:
        config = json.load(f)

    shard_filenames = sorted(glob.glob(os.path.join('data', "*.json")))
    with open(shard_filenames[0], "r") as f:
        data = json.load(f)

    encode, decode = get_tokenizer()
    tokenized_data = torch.tensor(encode(data), dtype = torch.long)

    n = int(0.8*len(data))
    train_data = data[:n]
    val_data = data[n:]
    model = LLM(config).to(device)

    if args.action == "train":
        train(args, model, config, train_data, val_data)
    elif args.action == "eval":
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