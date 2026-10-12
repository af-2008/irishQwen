"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""

import json
import regex as re
from collections import Counter

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py


"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py
"""

class Simple_Tokenizer():
    def __init__(self):
        self.merges = {} # (int, int) -> int

        self.special_tokens = {'<|sos|>': 256, '<|eos|>': 257} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = {idx: bytes([idx]) for idx in range(256)} | {idx: special.encode("utf-8") for special, idx in self.special_tokens.items()}


    def _count_pairs(self, tokens, counts=None):
        pair_counts = Counter()

        for pair in zip(tokens, tokens[1:]):
            pair_counts[pair] += 1

        return pair_counts

    def _merge_pair(self, tokens, target_pair, combined_id):
        merged_tokens = []

        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == target_pair:
                merged_tokens.append(combined_id)
                i += 2
            else:
                merged_tokens.append(tokens[i])
                i += 1

        return merged_tokens

    def train(self, text, max_vocab_size):
        vocab_size = len(self.vocab)
        num_merges = max_vocab_size - vocab_size

        text_bytes = text.encode('utf-8')
        tokens = list(text_bytes)

        for i in range(num_merges):
            pairs = self._count_pairs(ids)

            pair = pairs.most_common()[0][0]

            merged_token_id = vocab_size + i

            ids = self._merge_pair(tokens, pair, merged_token_id)

            self.merges[pair] = merged_token_id
            self.vocab[merged_token_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def decode(self, ids):
        # given ids (list of integers), return Python string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        import re
        special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
        special_chunks = re.split(special_pattern, text)

        encoded_str = []
        for chunk in special_chunks[1:-1]:
            if chunk in self.special_tokens:
                # this is a special token, encode it separately as a special case
                encoded_str.append(self.special_tokens[chunk])
            else:
                # given a string text, return the token ids
                text_bytes = chunk.encode("utf-8") # raw bytes
                chunk_ids = list(text_bytes) # list of integers in range 0..255

                while len(chunk_ids) >= 2:
                    # find the pair with the lowest merge index
                    counted_pairs = self._count_pairs(chunk_ids)
                    earliest_pair = min(counted_pairs, key=lambda p: self.merges.get(p, float("inf")))

                    # just the first pair in the list, arbitrarily
                    # we can detect this terminating case by a membership check
                    if earliest_pair not in self.merges:
                        break # nothing else can be merged anymore

                    # otherwise let's merge the best pair (lowest merge index)
                    pair_idx = self.merges[earliest_pair]
                    chunk_ids = self._merge_pairs(chunk_ids, earliest_pair, pair_idx)

                encoded_str += chunk_ids

        return encoded_str

class Regex_Tokenizer():
    GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

    def __init__(self):
        self.merges = {} # (int, int) -> int
        self.special_tokens = {'<|pad|>': 256, '<|sos|>': 257, '<|eos|>': 258} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = {idx: bytes([idx]) for idx in range(256)} | {idx: special.encode("utf-8") for special, idx in self.special_tokens.items()}

        self.pattern = self.GPT4_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)

    def _count_pairs(self, ids, counts=None):
        """
        Given a list of integers, return a dictionary of counts of consecutive pairs
        Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
        Optionally allows to update an existing dictionary of counts
        """
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]): # iterate consecutive elements
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge_pairs(self, ids, pair, idx):
        """
        In the list of integers (ids), replace all consecutive occurrences
        of pair with the new integer token idx
        Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
        """
        newids = []
        i = 0
        while i < len(ids):
            # if not at the very last position AND the pair matches, replace it
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def chunk(self, text):
        return re.findall(self.compiled_pattern, text)

    def train(self, text, max_vocab_size, verbose=False):
        pretrain_vocab_size = len(self.vocab)
        num_merges = max_vocab_size - pretrain_vocab_size

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # iteratively merge the most common pairs to create new tokens
        for i in range(num_merges):
            # count the number of times every consecutive pair appears
            stats = {}
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts
                self._count_pairs(chunk_ids, stats)

            # find the pair with the highest count
            pair = max(stats, key=stats.get)

            # mint a new token: assign it the next available id
            idx = pretrain_vocab_size + i

            # replace all occurrences of pair in ids with idx
            ids = [self._merge_pairs(chunk_ids, pair, idx) for chunk_ids in ids]

            # save the merge
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def decode(self, ids):
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            else:
                part_bytes.append('<|unk|>')

        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")

        return text

    def _encode_chunk(self, text_bytes):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)

        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = self._count_pairs(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = self._merge_pairs(ids, pair, idx)

        return ids

    def encode(self, text):
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
        special_chunks = re.split(special_pattern, text)

        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks[1:-1]:
            if part in self.special_tokens:
                # this is a special token, encode it separately as a special case
                ids.append(self.special_tokens[part])
            else:
                # this is an ordinary sequence, encode it normally
                text_chunks = self.chunk(part)

                for chunk in text_chunks:
                    chunk_bytes = chunk.encode("utf-8") # raw bytes
                    chunk_ids = self._encode_chunk(chunk_bytes)
                    ids.extend(chunk_ids)

        return ids

    def save(self, filepath):
        """Save the tokenizer state to a JSON file."""
        data = {
            "pattern": self.pattern,
            "special_tokens": self.special_tokens,
            "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath):
        """Load the tokenizer state from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        self.pattern = data["pattern"]
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = data["special_tokens"]
        self.merges = {tuple(map(int, k.split(","))): v for k, v in data["merges"].items()}

        # Rebuild vocab from base bytes + special tokens + merges
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for special, idx in self.special_tokens.items():
            self.vocab[idx] = special.encode("utf-8")
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def visualize_tokenization(self, ids):
        """Small helper function useful in debugging: visualize the tokenization of render_conversation"""
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        GRAY = '\033[90m'

        tokens = []
        for token_id in ids:
            token_str = self.decode([token_id])
            tokens.append(f"{GREEN}{token_str}{GRAY}({token_id}){RESET}")

        return ' | '.join(tokens)