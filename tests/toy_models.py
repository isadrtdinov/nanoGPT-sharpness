import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
# Simple toy models (CPU tests)
# ------------------------------------------------------------------ #

class ClassificationModel(nn.Module):
    """
    Single linear layer, no bias, cross-entropy loss.
        logits = X W^T
        loss = cross_entropy(logits, targets)
    """
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes, bias=False)

    def forward(self, inputs, targets):
        logits = self.linear(inputs)
        loss = F.cross_entropy(logits, targets)
        return logits, loss


class LinearRegressionModel(nn.Module):
    """
    Single linear layer, no bias, MSE loss.
        preds = X w
        loss = (1/N) || Xw - y ||^2
    Hessian w.r.t. w: (2/N) X^T X
    """
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1, bias=False)

    def forward(self, inputs, targets):
        preds = self.linear(inputs).squeeze(-1)
        loss = 0.5 * F.mse_loss(preds, targets)
        return preds, loss


class TwoLayerClassificationModel(nn.Module):
    """
    Two linear layers with bias, cross-entropy loss.
    Has multiple parameter tensors: weight1, bias1, weight2, bias2.
    """
    def __init__(self, in_features, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs, targets):
        x = torch.relu(self.fc1(inputs))
        logits = self.fc2(x)
        loss = F.cross_entropy(logits, targets)
        return logits, loss


# ------------------------------------------------------------------ #
# GPT-style toy models (GPU integration tests)
# ------------------------------------------------------------------ #

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        )

    def forward(self, x):
        return self.attn(x, x, x, attn_mask=self.causal_mask)[0]


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len, ffn_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, seq_len)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class SmallGPT(nn.Module):
    """
    Small GPT-style transformer language model.
    logits shape: [N, T, C]
    loss: cross-entropy averaged over N*T predictions
    """
    def __init__(self, vocab_size, embed_dim, num_heads, seq_len, num_layers, ffn_dim):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(seq_len, embed_dim)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, seq_len, ffn_dim)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.seq_len = seq_len

    def forward(self, inputs, targets):
        N, T = inputs.shape
        positions = torch.arange(T, device=inputs.device).unsqueeze(0)  # [1, T]
        x = self.token_embedding(inputs) + self.pos_embedding(positions)  # [N, T, embed_dim]
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)                                             # [N, T, C]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1)
        )
        return logits, loss


# ------------------------------------------------------------------ #
# Helper
# ------------------------------------------------------------------ #

def matvec_to_matrix(matvec_fn, model, inputs, targets):
    """Materialize the matrix defined by matvec_fn into a dense [P, P] matrix."""
    params = list(model.parameters())
    param_sizes = [p.numel() for p in params]
    num_params = sum(param_sizes)
    device = next(model.parameters()).device

    columns = []
    for i in range(num_params):
        e_i_flat = torch.zeros(num_params, device=device)
        e_i_flat[i] = 1.0
        e_i, offset = [], 0
        for p, size in zip(params, param_sizes):
            e_i.append(e_i_flat[offset:offset + size].view_as(p))
            offset += size
        Mv = matvec_fn(model, inputs, targets, e_i)
        columns.append(torch.cat([v.flatten() for v in Mv]))

    return torch.stack(columns, dim=1)  # [P, P]
