from types import ClassMethodDescriptorType
import torch
import pandas as pd
import numpy as np
import os
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer

"""
Text Feature Extractor for VIBE model
"""


class TextFeatureExtractor:

    def __init__(self, model_name: str, device: str, **kwargs):
        self.model = ((AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
                       .to(device))
                      .eval())

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    @classmethod
    def read_transcripts_(cls, path: str, sep: str = '\t', transcript_column_name: str = 'text_per_tr'):
        df = pd.read_csv(path, sep=sep)
        # 'text_per_tr' is assumed to contain the text within each 1.49s TR.
        tr_series = df[transcript_column_name]
        # Normalize NaNs/empties to a single space.
        return [" " if (isinstance(t, float) and np.isnan(t)) else (t if isinstance(t, str) and len(t) > 0 else " ")
                for t in tr_series.tolist()]

    @classmethod
    def build_prompt(cls, transcripts: list[str]) -> tuple[str, list[tuple[int, int]]]:
        """
        Concatenate TR lines with newline separators to form a single string
        that we feed into the tokenizer/model. We also keep track of the
        (start, end) character span of each TR line in the concatenated string.

        Returns
        -------
        full_text : str
            The concatenated text ("TR0\\nTR1\\n...").
        char_spans : list[(start, end)]
            Character spans for each TR inside full_text. These spans let us map
            token offsets back to their corresponding TR, enabling per-TR pooling.
        """
        char_spans = []
        parts = []
        pos = 0

        for line in transcripts:
            start = pos
            parts.append(line + "\n")  # newline marks TR boundary
            pos += len(line) + 1  # +1 for '\n'
            char_spans.append((start, pos))

        return "".join(parts), char_spans

    @classmethod
    def tokens_by_tr(cls, offsets: torch.Tensor, char_spans: list[tuple[int, int]]):
        """
        Map each token index to a TR index by comparing token character offsets to TR character spans.

        Parameters
        ----------
        offsets : Tensor (seq_len, 2)
            Character start/end of each token in the concatenated string.
            Special tokens often have (0, 0) and should be ignored.
        char_spans : list[(start, end)]
            Character spans per TR in the concatenated string.

        Returns
        -------
        list[list[int]]
            A list of length num_TRs; each element is a list of token indices belonging to that TR.
        """
        per_tr = [[] for _ in char_spans]

        for tok_idx, (tok_start, tok_end) in enumerate(offsets.tolist()):
            # Ignore special tokens with empty or non-overlapping spans
            if tok_end <= char_spans[0][0]:  # typically catches specials at the very beginning
                continue

            # Assign the token to the first TR it overlaps.
            # (Overlap condition: token_start < TR_end and token_end > TR_start)
            for tr_idx, (start, end) in enumerate(char_spans):
                if tok_start < end and tok_end > start:
                    per_tr[tr_idx].append(tok_idx)
                    break

        return per_tr

    @torch.no_grad()  # inference only, no gradients needed
    def extract_features(
            self,
            path: str,
            sep: str = '\t',
            transcript_column_name: str = 'text_per_tr'
    ):
        """
        Extract per-TR text features from a transcript TSV and save as a NumPy array.

        Pipeline
        --------
        1) Load one string per TR.
        2) Concatenate with newlines; record character spans per TR.
        3) Tokenize with offsets.
        4) Run LLM once to get hidden states.
        5) Average the last 4 layers (semantic features).
        6) For each TR, average features of all tokens assigned to that TR.

        Saved File
        ----------
        A .npy file containing an array of shape (num_TR, hidden_size).

        Notes
        -------------------------
        - Averaging token embeddings within the same TR is a simple 'temporal pooling' step,
          analogous to averaging words heard within that TR. More sophisticated pooling
          (e.g., attention-weighted) can be explored later.
        """
        transcripts = TextFeatureExtractor.read_transcripts_(path, sep,
                                                             transcript_column_name)  # load_transcript(tsv_path)
        num_tr = len(transcripts)

        # Build a single text blob + TR character spans.
        fname = os.path.basename(path)
        full_text, char_spans = TextFeatureExtractor.build_prompt(transcripts)

        # Tokenize with character offsets
        encoded = self.tokenizer(
            full_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True,
            padding=False,
            truncation=False
        )

        input_ids = encoded["input_ids"].to(self.model.device)  # (1, seq_len)
        offsets = encoded["offset_mapping"][0]  # (seq_len, 2)

        # --- Safety: extremely long inputs ---
        # If the tokenized length exceeds the model's context window, we truncate with a warning.
        # (Later you can switch to chunking; truncation here is a minimal safeguard.)
        max_len = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
        if isinstance(max_len, int) and input_ids.size(1) > max_len:
            logging.warning(
                f"{fname}: tokenized length {input_ids.size(1)} > model max {max_len}. Truncating to first {max_len} tokens."
            )
            input_ids = input_ids[:, :max_len]
            offsets = offsets[:max_len]

        # Forward pass to get hidden states; inference_mode is slightly faster than no_grad
        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, output_hidden_states=True)

        # hidden_states is a tuple (n_layers+1) of tensors with shape (1, seq_len, hidden_size).
        # Average the last 4 layers (a robust default for semantic embeddings).
        hidden_last4 = torch.stack(outputs.hidden_states[-4:], dim=0).mean(dim=0)[0]  # (seq_len, hidden_size)

        # Token -> TR assignment, then per-TR mean pooling.
        per_tr_tokens = TextFeatureExtractor.tokens_by_tr(offsets, char_spans)  # list[list[int]]
        features = np.zeros((num_tr, hidden_last4.size(1)), dtype=np.float32)

        for i, token_idxs in enumerate(per_tr_tokens):
            if token_idxs:  # non-empty TR
                features[i] = hidden_last4[token_idxs].mean(dim=0).float().cpu().numpy()
            # else: silent TR -> remain zeros

        return features
