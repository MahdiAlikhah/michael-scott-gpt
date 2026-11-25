import argparse
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embed: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embed // n_head

        self.c_attn = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.c_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, kv_cache=None):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)

        is_causal = (T > 1) and (kv_cache is None)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(y)
        return out, new_cache


class FeedForward(nn.Module):
    def __init__(self, n_embed: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = CausalSelfAttention(n_embed, n_head, dropout=dropout)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ffwd = FeedForward(n_embed, dropout=dropout)

    def forward(self, x: torch.Tensor, kv_cache=None):
        attn_out, new_cache = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.ffwd(self.ln2(x))
        return x, new_cache


class MichaelScottGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 2048,
        n_embed: int = 256,
        n_head: int = 8,
        n_layer: int = 6,
        block_size: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [Block(n_embed, n_head, dropout=dropout) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

        # Weight tying between token embeddings and output projection
        self.token_embedding.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets=None, kv_cache=None):
        B, T = idx.shape

        past_len = kv_cache[0][0].size(2) if kv_cache is not None else 0
        assert (
            past_len + T <= self.block_size
        ), f"Sequence length {past_len + T} exceeds block_size {self.block_size}"

        pos = torch.arange(past_len, past_len + T, device=idx.device)
        x = self.drop(self.token_embedding(idx) + self.position_embedding(pos))

        new_caches = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, updated_cache = block(x, kv_cache=layer_cache)
            new_caches.append(updated_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

        return logits, loss, new_caches


def load_artifacts(
    checkpoint_path: str = "michael_scott_sft.pt",
    tokenizer_path: str = "michael_scott_tokenizer.json",
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    eot_token = tokenizer.token_to_id("<|endoftext|>")

    raw_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Support both raw state_dict (from SFT) and wrapped dict (from pretraining)
    if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt:
        state_dict = raw_ckpt["model_state_dict"]
        cfg = raw_ckpt.get("config", {})
    else:
        state_dict = raw_ckpt
        cfg = {}

    model = MichaelScottGPT(
        vocab_size=cfg.get("vocab_size", vocab_size),
        n_embed=cfg.get("n_embed", 256),
        n_head=cfg.get("n_head", 8),
        n_layer=cfg.get("n_layer", 6),
        block_size=cfg.get("block_size", 256),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, tokenizer, eot_token, device


@torch.no_grad()
def generate_reply(
    model: MichaelScottGPT,
    tokenizer: Tokenizer,
    eot_token: int,
    conversation_history: str,
    device: str,
    temperature: float = 0.72,
    top_k: int = 40,
    top_p: float = 0.9,
    rep_penalty: float = 1.15,
    min_new_tokens: int = 2,
    max_new_tokens: int = 70,
) -> str:
    prompt_ids = tokenizer.encode(conversation_history).ids
    idx = torch.tensor(
        [prompt_ids[-(model.block_size - max_new_tokens) :]],
        dtype=torch.long,
        device=device,
    )

    # Phase 1: Prefill KV cache with conversation history
    logits, _, kv_cache = model(idx, kv_cache=None)
    reply_ids = []

    # Phase 2: Incremental single-token decoding using KV cache
    for _ in range(max_new_tokens):
        next_logits = logits[:, -1, :].clone()

        for prev_id in set(reply_ids):
            next_logits[0, prev_id] /= rep_penalty

        next_logits = next_logits / max(temperature, 1e-5)

        if len(reply_ids) < min_new_tokens and eot_token is not None:
            next_logits[0, eot_token] = -float("Inf")

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = -float("Inf")

        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            next_logits[indices_to_remove] = -float("Inf")

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token_id = next_token.item()

        if token_id == eot_token:
            break

        piece = tokenizer.decode([token_id])
        if "\n" in piece:
            before_nl = piece.split("\n")[0]
            if before_nl:
                reply_ids.extend(tokenizer.encode(before_nl).ids)
            break

        reply_ids.append(token_id)
        if kv_cache[0][0].size(2) >= model.block_size:
            break

        logits, _, kv_cache = model(next_token, kv_cache=kv_cache)

    return tokenizer.decode(reply_ids).strip()


def main():
    parser = argparse.ArgumentParser(description="Chat with MichaelScottGPT")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="michael_scott_sft.pt",
        help="Path to the trained PyTorch weights (.pt)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="michael_scott_tokenizer.json",
        help="Path to the trained BPE tokenizer (.json)",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="Jim",
        help="Character name you roleplay as in the prompt (default: Jim)",
    )
    parser.add_argument("--temperature", type=float, default=0.72)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--rep-penalty", type=float, default=1.15)
    parser.add_argument("--max-tokens", type=int, default=70)
    args = parser.parse_args()

    model, tokenizer, eot_token, device = load_artifacts(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
    )

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded MichaelScottGPT ({param_count:.2f}M params) on {device}.")
    print("Type 'reset' to clear conversation history, or 'quit' to exit.")
    print("-" * 60)

    history = ""
    while True:
        try:
            user_input = input(f"{args.speaker}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nMichael: Catch you on the flippity flip.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Michael: Catch you on the flippity flip.")
            break
        if user_input.lower() == "reset":
            history = ""
            print("--- Conversation memory cleared ---\n")
            continue

        history += f"{args.speaker}: {user_input}\nMichael:"
        reply = generate_reply(
            model=model,
            tokenizer=tokenizer,
            eot_token=eot_token,
            conversation_history=history,
            device=device,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            rep_penalty=args.rep_penalty,
            max_new_tokens=args.max_tokens,
        )
        history += f" {reply}\n"
        print(f"Michael: {reply}\n")


if __name__ == "__main__":
    main()
