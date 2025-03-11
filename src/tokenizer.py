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

