import argparse
import yaml
import os
import torch

from src.model import LLM
from src.tokenizer import get_tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def load_config(config_path):
    with open(config_path, "r") as f:
        return Config(**yaml.safe_load(f))

def load_model(config):
    model = LLM(config)
    model.load_state_dict(torch.load(os.path.join(config.model_dir, config.model_name), map_location="cpu"))
    model.to(device)
    model.eval()
    return model

def generate_output(model, tokenizer, input_text, max_new_tokens):
    _, encode, decode = tokenizer
    tokenized_input = torch.tensor(encode(input_text), dtype=torch.long).unsqueeze(0).to(device)
    output = model.generate(tokenized_input, max_new_tokens=max_new_tokens)
    return decode(output.squeeze(0).tolist())

def main(args):
    config = load_config(args.config)
    tokenizer = get_tokenizer(config.vocab_file)
    model = load_model(config)
    output = generate_output(model, tokenizer, args.input, args.max_new_token)
    print(f"[INFO] Generate output: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--max_new_token", type=int, default=128)
    args = parser.parse_args()
    main(args)