import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    """One head of self-attention."""
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size)
        self.query = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)
        weights = F.softmax(q @ k.transpose(-2, -1), dim=-1)
        v = self.value(x)
        return weights @ v

    def get_weights(self, x):
        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1)
        weights = F.softmax(weights, dim=-1)
        return weights

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    def __init__(self, n_embd, n_heads, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.proj(x)
        return self.dropout(x)

    def get_weights_stacked(self, x):
        weights = [h.get_weights(x) for h in self.heads]
        return torch.stack(weights, dim=0)

class FeedForward(nn.Module):
    """Feed-forward network."""
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear2 = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)

class Block(nn.Module):
    """Transformer block combining multi-head attention and feed-forward network."""
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.attention = MultiHeadAttention(n_embd, n_head, head_size, dropout)
        self.feed_forward = FeedForward(n_embd, dropout)
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x

    def get_attention_matrix(self, x):
        return self.attention.get_weights_stacked(self.norm1(x))

class PangenomeTransformerModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, dropout):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.first_block = Block(n_embd, n_head, dropout)  # First block
        self.second_block = Block(n_embd, n_head, dropout)  # Second block
        self.third_block = Block(n_embd, n_head, dropout)  # Third block
        self.last_block = Block(n_embd, n_head, dropout)  # Last block
        self.final_layers = nn.Linear(n_embd, 10)  # Output layer

    def forward(self, x, return_all_blocks=True):
        x = self.token_embeddings(x)
        outputs = []
        x = self.first_block(x)  # Pass through the first block
        if return_all_blocks:
            outputs.append(self.final_layers(x))
        x = self.second_block(x)  # Pass through the second block
        if return_all_blocks:
            outputs.append(self.final_layers(x))
        x = self.third_block(x)  # Pass through the third block
        if return_all_blocks:
            outputs.append(self.final_layers(x))
        x = self.last_block(x)  # Pass through the last block
        if return_all_blocks:
            outputs.append(self.final_layers(x))
            return outputs
        return self.final_layers(x)

    def get_attention_matrix(self, x):
        x = self.token_embeddings(x)
        return self.last_block.get_attention_matrix(x)
