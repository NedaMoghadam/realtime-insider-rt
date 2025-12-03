"""
TinyLlama Model Wrapper
Handles model loading, inference, and embedding generation.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Dict
import numpy as np

from logger import get_logger
from exceptions import ModelLoadError, InferenceError

logger = get_logger(__name__)


class TinyLlamaModel:
    """
    Wrapper for TinyLlama model with embedding extraction.
    """

    def __init__(
        self,
        model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        cache_dir: Optional[str] = None,
        use_gpu: bool = True,
        max_length: int = 256,
    ):
        """
        Initialize TinyLlama model.

        Args:
            model_id: HuggingFace model identifier
            cache_dir: Directory to cache model weights
            use_gpu: Whether to use GPU if available
            max_length: Maximum sequence length
        """
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.max_length = max_length

        # Choose device
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU")

        # Load model + tokenizer
        self._load_model()

        logger.info(f"Model initialized: {model_id}")
        logger.info(f"Hidden size: {self.hidden_size}")
        logger.info(f"Num layers: {self.num_layers}")
        logger.info(f"Vocab size: {self.vocab_size}")

    def _load_model(self):
        """Load model and tokenizer from HuggingFace."""
        try:
            logger.info(f"Loading tokenizer: {self.model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
            )

            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"Loading model: {self.model_id}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
            )

            self.model.to(self.device)
            self.model.eval()

            # Freeze parameters
            for p in self.model.parameters():
                p.requires_grad = False

            # Store config
            self.hidden_size = self.model.config.hidden_size
            self.num_layers = self.model.config.num_hidden_layers
            self.vocab_size = self.model.config.vocab_size

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelLoadError(f"Could not load model {self.model_id}: {e}")

    def tokenize(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize input texts.

        Returns a dict with input_ids and attention_mask on the correct device.
        """
        if max_length is None:
            max_length = self.max_length

        try:
            encoded = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            return encoded

        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            raise InferenceError(f"Tokenization error: {e}")

    @torch.no_grad()
    def get_embeddings(
        self,
        texts: List[str],
        layer: int = -1,
        pooling: str = "last",
    ) -> np.ndarray:
        """
        Get embeddings from a given layer.

        pooling: "last", "mean", or "max"
        """
        try:
            encoded = self.tokenize(texts)

            outputs = self.model(
                **encoded,
                output_hidden_states=True,
            )

            hidden_states = outputs.hidden_states[layer]  # (batch, seq_len, hidden)

            if pooling == "last":
                # Last token
                embeddings = hidden_states[:, -1, :]
            elif pooling == "mean":
                # Mean over valid tokens
                mask = encoded["attention_mask"].unsqueeze(-1)  # (b, t, 1)
                embeddings = (hidden_states * mask).sum(1) / mask.sum(1).clamp_min(1)
            elif pooling == "max":
                embeddings = hidden_states.max(1)[0]
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling}")

            return embeddings.cpu().numpy()

        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            raise InferenceError(f"Could not extract embeddings: {e}")

    @torch.no_grad()
    def compute_nll(self, texts: List[str]) -> List[float]:
        """
        Compute negative log-likelihood (NLL) per text.
        Lower NLL = more normal; higher NLL = more surprising.
        """
        try:
            encoded = self.tokenize(texts)
            outputs = self.model(**encoded)
            logits = outputs.logits  # (b, t, vocab)

            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()

            log_probs = torch.log_softmax(shift_logits, dim=-1)

            token_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=shift_labels.unsqueeze(-1),
            ).squeeze(-1)

            nll_scores = []
            for i in range(input_ids.size(0)):
                mask = shift_mask[i].float()
                nll = -(token_log_probs[i] * mask).sum() / mask.sum().clamp_min(1)
                nll_scores.append(nll.item())

            return nll_scores

        except Exception as e:
            logger.error(f"NLL computation failed: {e}")
            raise InferenceError(f"Could not compute NLL: {e}")


if __name__ == "__main__":
    # Quick self-test
    model = TinyLlamaModel()
    texts = [
        "User logged into system.",
        "File access detected.",
        "Abnormal network activity from external IP.",
    ]

    print("Testing embeddings...")
    embs = model.get_embeddings(texts)
    print("Embeddings shape:", embs.shape)

    print("\nTesting NLL...")
    nll = model.compute_nll(texts)
    print("NLL:", nll)
