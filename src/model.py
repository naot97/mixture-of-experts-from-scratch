import torch
from torch import nn
from torch.nn import functional as F

class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_embed = config.n_embed
        dropout = config.expert_dropout
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class MoeLayer(nn.Module):
    def __init__(self, experts, gate, k=1):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.k = k

    def forward(self, inputs):
        inputs_squashed = inputs.view(-1, inputs.shape[-1])
        gate_logits = self.gate(inputs_squashed)
        topk_values, selected_experts = torch.topk(gate_logits, self.k)
        weights = F.softmax(topk_values, dim=1).type_as(inputs)
        results = torch.zeros_like(inputs_squashed)
        
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert] * expert(inputs_squashed[batch_idx])
        
        return results.view_as(inputs)

class Head(nn.Module):
    def __init__(self, config, head_size):
        super().__init__()
        n_embed = config.n_embed
        block_size = config.block_size
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(config.attention_dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_head = config.n_head
        head_size = config.n_embed // n_head
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(n_head)])
        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(out)
        return out

class BlockLayer(nn.Module):
    def __init__(self, config, idx):
        super().__init__()
        n_embed = config.n_embed
        n_experts = config.n_experts
        self.sa_head = MultiHeadAttention(config)
        self.smoe = MoeLayer(
            experts=[Expert(config) for _ in range(n_experts)],
            gate=nn.Linear(n_embed, n_experts, bias=False)
        )
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        moe_output = x + self.ln1(self.sa_head(x))
        moe_output = moe_output + self.ln2(self.smoe(moe_output))
        return moe_output

class LLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vocab_size = config.vocab_size
        n_embed = config.n_embed
        n_layer = config.n_layer
        block_size = config.block_size
        self.embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.encoders = nn.ModuleList([BlockLayer(config, layer_i) for layer_i in range(n_layer)])
        self.register_buffer('position_ids', torch.arange(block_size).to(self.device))
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, labels=None):
        B, T = idx.shape
        token_embedding_vec = self.embedding(idx)
        pos_embedding_vec = self.position_embedding_table(self.position_ids[:T])
        x = token_embedding_vec + pos_embedding_vec
        
        for encoder in self.encoders:
            x = encoder(x)

        logits = self.lm_head(x)
        
        if labels is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = labels.view(B * T)
            loss = F.cross_entropy(logits, targets)
            return loss
        else:
            return logits
