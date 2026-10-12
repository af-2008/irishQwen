import re
import math
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from irishQwen.irishQwen import IrishQwen
from irishQwen.tokenizer import Regex_Tokenizer
from irishQwen import optimizers


def load_shakespeare(path):
    """Complete works, minus the Project Gutenberg legal boilerplate."""
    text = open(path, encoding="utf-8", errors="replace").read()
    text = re.sub(r"<<.*?>>", "", text, flags=re.S)        # copyright notice between every work
    text = text[text.index("THE SONNETS"):]                 # skip the Gutenberg header
    end = text.find("End of this Etext")
    if end != -1:
        text = text[:end]
    text = re.sub(r"^[ \t]+", "", text, flags=re.M)         # drop the indentation
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


class ShakespeareWindows(Dataset):
    """
    The whole corpus as one long token stream, cut into fixed windows of
    ctx_len + 1 tokens. No padding needed: X = window[:-1], Y = window[1:].
    """
    def __init__(self, token_ids, ctx_len, vocab_size, device):
        self.vocab_size = vocab_size
        self.device = device
        tokens = torch.tensor(token_ids, dtype=torch.long)
        n = (len(tokens) - 1) // ctx_len
        self.windows = [tokens[i * ctx_len : (i + 1) * ctx_len + 1] for i in range(n)]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]

    def collate(self, batch):
        W = torch.stack(batch)                                       # (B, T+1)
        X = W[:, :-1]
        Y = F.one_hot(W[:, 1:], num_classes=self.vocab_size).float() # (B, T, V)
        return X.to(self.device), Y.to(self.device)


def validation_loss(chat, loader):
    """Average loss on held-out windows (forward only, no weight updates)."""
    losses = [chat.cross_entropy(chat.forward(X), Y).item() for X, Y in loader]
    return sum(losses) / len(losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a small Qwen3-architecture IrishQwen on Shakespeare")
    parser.add_argument("--dataset", default="Datasets/shakespeare.txt")
    parser.add_argument("--tokenizer", default="Datasets/shakespeare_vocab.json")
    parser.add_argument("--ctx_len", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_kv_heads", type=int, default=4)
    parser.add_argument("--ffn_dim", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-3)
    parser.add_argument("--optimizer", choices=["sgd", "momentum", "adam", "adamw"], default="adam")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW only")
    parser.add_argument("--val_frac", type=float, default=0.05, help="Fraction of the corpus held out for validation")
    parser.add_argument("--save", default="Datasets/qwen_shakespeare.pt")
    args = parser.parse_args()

    tokenizer = Regex_Tokenizer()
    tokenizer.load(args.tokenizer)
    vocab_size = len(tokenizer.vocab)

    # Regex_Tokenizer.encode expects text wrapped in special tokens
    token_ids = tokenizer.encode("<|sos|>" + load_shakespeare(args.dataset) + "<|eos|>")
    print(f"{len(token_ids):,} tokens, vocab {vocab_size}")

    # A small Qwen3: same architecture as Qwen3-0.6B, just narrower and shallower
    chat = IrishQwen(
        vocab_size=vocab_size,
        ctx_len=args.ctx_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        head_dim=args.d_model // args.n_heads,
        ffn_dim=args.ffn_dim,
        rope_theta=10_000.0,
        padding_idx=tokenizer.special_tokens["<|pad|>"],
    )

    # Hold out the END of the corpus (not random windows) so no validation text leaks into training
    split = int(len(token_ids) * (1 - args.val_frac))
    train_set = ShakespeareWindows(token_ids[:split], args.ctx_len, vocab_size, chat.device)
    val_set = ShakespeareWindows(token_ids[split:], args.ctx_len, vocab_size, chat.device)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=train_set.collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=val_set.collate)
    print(f"{len(train_set):,} train / {len(val_set):,} validation windows of {args.ctx_len} tokens")

    if args.optimizer == "adamw":
        optimizer = optimizers.AdamW(chat, lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = {
            "sgd": optimizers.SGD,           # try --learning_rate 0.2
            "momentum": optimizers.Momentum, # try --learning_rate 0.02
            "adam": optimizers.Adam,         # try --learning_rate 3e-3
        }[args.optimizer](chat, lr=args.learning_rate)

    best_val = float("inf")
    for epoch in range(args.epochs):
        # Cosine schedule: start at learning_rate, glide smoothly down to ~0 by the last epoch
        optimizer.lr = args.learning_rate * 0.5 * (1 + math.cos(math.pi * epoch / args.epochs))

        history = chat.train(train_loader, 1, optimizer.lr, verbose=False, optimizer=optimizer)
        val = validation_loss(chat, val_loader)

        # Only keep the epoch that does best on text it has never trained on
        marker = ""
        if val < best_val:
            best_val = val
            chat.save_converted_qwen3_checkpoint(args.save)
            marker = "  <- best, saved"
        print(f"epoch {epoch + 1}/{args.epochs}  lr {optimizer.lr:.1e}  "
              f"train loss {history['loss_history'][0]:.3f}  val loss {val:.3f}{marker}", flush=True)

    # Generate from the best checkpoint, not necessarily the last epoch
    chat = IrishQwen.from_checkpoint_config(args.save, ctx_len=args.ctx_len)
    chat.load_converted_qwen3_checkpoint(args.save)
    print(f"Loaded best model (val loss {best_val:.3f}) from {args.save}")

    for prompt in ["ROMEO.", "To be, or not to be", "KING HENRY.", "Shall I compare thee"]:
        prompt_ids = tokenizer.encode("<|sos|>" + prompt + "<|eos|>")[1:-1]
        out = chat.chat(prompt_ids, max_new_tokens=150, temperature=0.8, top_k=40, eos_token_id=None)
        print("=" * 60)
        print(tokenizer.decode(out))
