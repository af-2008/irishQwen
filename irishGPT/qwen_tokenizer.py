import json
import unicodedata
import regex as re

from .gpt2_tokenizer import GPT2Tokenizer, bytes_to_unicode


class QwenTokenizer(GPT2Tokenizer):
    """
    Qwen's tokenizer is the same byte-level BPE as GPT-2 (same bpe() / decode()),
    with three differences:
      1) A different pre-tokenization regex (e.g. digits are split one at a time)
      2) A much bigger vocab (~151k) and NFC unicode normalization
      3) Special chat tokens like <|im_start|>, <|im_end|>, <think>
    """
    QWEN_PATTERN = re.compile(
        r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    )

    def __init__(self, tokenizer_json_path):
        # tokenizer.json is written by scripts/import_qwen3.py
        with open(tokenizer_json_path, "r", encoding="utf-8") as f:
            tok = json.load(f)

        self.encoder = tok["model"]["vocab"]
        merges = [tuple(m.split()) if isinstance(m, str) else tuple(m) for m in tok["model"]["merges"]]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))

        self.special_tokens = {t["content"]: t["id"] for t in tok["added_tokens"]}
        self.encoder.update(self.special_tokens)
        self.decoder = {v: k for k, v in self.encoder.items()}

        # Split text on special tokens first so they are never BPE'd
        self.special_pattern = re.compile("(" + "|".join(re.escape(t) for t in self.special_tokens) + ")")

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.cache = {}

    def encode(self, text):
        bpe_tokens = []
        for chunk in re.split(self.special_pattern, unicodedata.normalize("NFC", text)):
            if chunk in self.special_tokens:
                bpe_tokens.append(self.special_tokens[chunk])
                continue
            for token in re.findall(self.QWEN_PATTERN, chunk):
                token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
                bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def decode(self, token_ids):
        text = ""
        byte_values = bytearray()
        for token_id in token_ids:
            piece = self.decoder[token_id]
            if piece in self.special_tokens:
                text += byte_values.decode("utf-8", errors="replace") + piece
                byte_values = bytearray()
            else:
                byte_values.extend(self.byte_decoder[c] for c in piece)
        return text + byte_values.decode("utf-8", errors="replace")

    def chat_prompt(self, user_message, enable_thinking=False):
        """
        Wrap a message in Qwen3's ChatML template, ready for the assistant to reply.
        With enable_thinking=False we pre-fill an empty <think></think> block so
        the model answers directly instead of reasoning first.
        """
        prompt = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        if not enable_thinking:
            prompt += "<think>\n\n</think>\n\n"
        return self.encode(prompt)
