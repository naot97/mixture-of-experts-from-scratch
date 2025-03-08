import argparse
import glob
import json
import os
import unicodedata

from tqdm import tqdm
import torch
import yaml

from src.model import LLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_tokenizer(vocab_file):
    with open(vocab_file, "r", encoding="utf-8") as f:
        chars = f.read()

    if '\t' not in chars:
        chars += '\t'  # Add the tab character if missing

    ctoi = {ch: i for i, ch in enumerate(chars)}
    itoc = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [ctoi[c] for c in s]
    decode = lambda l: "".join([itoc[x] for x in l])
    return chars, encode, decode

def get_batch(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

@torch.no_grad()
def evaluate(model, config, val_data):
    model.eval()
    losses = torch.zeros(config.eval_iters)
    for k in tqdm(range(config.eval_iters), desc="[INFO] Evaluating"):
        X, Y = get_batch(val_data, config.batch_size, config.block_size)
        X, Y = X.to(device), Y.to(device)
        loss = model(X, Y)
        losses[k] = loss.item()
    return losses.mean().item()

def train(model, config, train_data, val_data):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    train_losses = []
    print("[INFO] Starting training with {config.max_iters} steps...")
    for iter in tqdm(range(config.max_iters), desc="[INFO] Training"):
        xb, yb = get_batch(train_data, config.batch_size, config.block_size)
        xb, yb = xb.to(device), yb.to(device)

        loss = model.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if iter % 100 == 0 or iter == config.max_iters - 1:
            eval_loss = evaluate(model, config, val_data)
            train_loss = sum(train_losses) / len(train_losses)
            print(f"[INFO] step {iter}: train loss {train_loss:.4f}, val loss {eval_loss:.4f}")
            model.train()

def main(args):
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
        config = Config(**config_dict)

    shard_filenames = sorted(glob.glob(os.path.join('data', "*.json")))
    with open(shard_filenames[0], "r") as f:
        data = json.load(f)

    data = "\n".join([x['story'] for x in data])
    data = unicodedata.normalize("NFKC", data)

    _, encode, decode = get_tokenizer(config.vocab_file)
    tokenized_data = torch.tensor(encode(data), dtype=torch.long)

    n = int(0.8 * len(tokenized_data))
    train_data, val_data = tokenized_data[:n], tokenized_data[n:]
    model = LLM(config).to(device)

    if args.action == "train":
        train(model, config, train_data, val_data)
        print("[INFO] Training complete.")
    elif args.action == "eval":
        eval_loss = evaluate(model, config, val_data)
        print("[INFO] val loss: {eval_loss:.4f}")
        print("[INFO] Evaluation complete.")
    else:
        raise ValueError(f"[ERROR] Invalid action: {args.action}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--action", type=str, default="train")
    args = parser.parse_args()

    main(args)
