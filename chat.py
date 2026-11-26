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
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y), new_cache


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


class MichaelScottTWSS_Chat(nn.Module):
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

        self.blocks = nn.ModuleList([Block(n_embed, n_head, dropout=dropout) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight

        # Stage 3: TWSS Detection Heads
        self.twss_head = nn.Sequential(
            nn.LayerNorm(n_embed),
            nn.Linear(n_embed, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.proj_head = nn.Sequential(
            nn.Linear(n_embed, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )

    def forward(self, idx: torch.Tensor, kv_cache=None, eval_twss=False):
        B, T = idx.shape

        past_len = kv_cache[0][0].size(2) if kv_cache is not None else 0
        assert past_len + T <= self.block_size, f"Sequence length exceeds block_size"

        pos = torch.arange(past_len, past_len + T, device=idx.device)
        x = self.drop(self.token_embedding(idx) + self.position_embedding(pos))

        new_caches = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, updated_cache = block(x, kv_cache=layer_cache)
            new_caches.append(updated_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        twss_prob = None
        if eval_twss:
            # Evaluate TWSS probability on the final token of the user's prompt
            h_last = x[:, -1, :]
            twss_logit = self.twss_head(h_last).squeeze(-1)
            twss_prob = torch.sigmoid(twss_logit).item()

        return logits, new_caches, twss_prob


def load_artifacts(checkpoint_path: str, tokenizer_path: str, device: str):
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    eot_token = tokenizer.token_to_id("<|endoftext|>")

    raw_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = raw_ckpt["model_state_dict"] if "model_state_dict" in raw_ckpt else raw_ckpt

    model = MichaelScottTWSS_Chat(vocab_size=vocab_size).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, tokenizer, eot_token


@torch.no_grad()
def generate_reply(
    model: MichaelScottTWSS_Chat,
    tokenizer: Tokenizer,
    eot_token: int,
    conversation_history: str,
    device: str,
    temperature: float = 0.72,
    top_k: int = 40,
    rep_penalty: float = 1.15,
    max_new_tokens: int = 70,
    twss_threshold: float = 0.65,
):
    prompt_ids = tokenizer.encode(conversation_history).ids
    idx = torch.tensor([prompt_ids[-(model.block_size - max_new_tokens) :]], dtype=torch.long, device=device)

    # Phase 1: Prefill KV cache and evaluate TWSS
    logits, kv_cache, twss_prob = model(idx, kv_cache=None, eval_twss=True)
    
    reply_str = ""
    # If the setup clears the detector threshold, Michael interrupts with the joke!
    if twss_prob > twss_threshold:
        joke = "That's what she said! "
        reply_str += joke
        # We must append the joke to the KV cache so the model knows it just said it
        joke_ids = tokenizer.encode(joke).ids
        for j_id in joke_ids:
            j_tensor = torch.tensor([[j_id]], dtype=torch.long, device=device)
            logits, kv_cache, _ = model(j_tensor, kv_cache=kv_cache, eval_twss=False)

    reply_ids = []

    # Phase 2: Incremental single-token decoding using KV cache
    for _ in range(max_new_tokens):
        next_logits = logits[:, -1, :].clone()

        for prev_id in set(reply_ids):
            next_logits[0, prev_id] /= rep_penalty

        next_logits = next_logits / max(temperature, 1e-5)

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = -float("Inf")

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

        logits, kv_cache, _ = model(next_token, kv_cache=kv_cache, eval_twss=False)

    return reply_str + tokenizer.decode(reply_ids).strip()


def main():
    parser = argparse.ArgumentParser(description="Chat with MichaelScottGPT (TWSS Edition)")
    parser.add_argument("--checkpoint", type=str, default="weights/michael_scott_twss.pt")
    parser.add_argument("--tokenizer", type=str, default="weights/michael_scott_tokenizer.json")
    parser.add_argument("--speaker", type=str, default="Jim")
    parser.add_argument("--temperature", type=float, default=0.72)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model, tokenizer, eot_token = load_artifacts(args.checkpoint, args.tokenizer, device)
    except FileNotFoundError as e:
        print(f"Error: {e}\nPlease ensure your .pt and .json files are inside a 'weights' folder.")
        return

    print(f"Loaded MichaelScottGPT (TWSS-Enabled) on {device}.")
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
        )
        
        history += f" {reply}\n"
        print(f"Michael: {reply}\n")


if __name__ == "__main__":
    main()