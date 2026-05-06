"""
PubMedBERT Disease Classifier
Model: microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
Trained on: PubMed from scratch (no general-English pre-training)
Best on: BLURB benchmark, complex medical terminology
Weakness for ZeaCares: academic language ≠ clinical notes style
"""
import logging
import time
from dataclasses import dataclass
from typing import Optional
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"


@dataclass
class EmbeddingResult:
    embedding: np.ndarray
    inference_time_ms: float
    model_name: str


class PubMedBERTClassifier:
    def __init__(self, device: Optional[str] = None):
        self.model_name = MODEL_NAME
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        logger.info(f"Loading PubMedBERT from {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self._loaded = True
        logger.info("PubMedBERT loaded")

    def _mean_pool(self, outputs, attention_mask) -> torch.Tensor:
        token_emb = outputs.last_hidden_state
        mask_exp = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
        return torch.sum(token_emb * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)

    def embed(self, text: str) -> EmbeddingResult:
        self.load()
        start = time.time()
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = self._mean_pool(outputs, inputs["attention_mask"]).squeeze().cpu().numpy()

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        elapsed_ms = (time.time() - start) * 1000
        return EmbeddingResult(embedding=embedding, inference_time_ms=elapsed_ms,
                               model_name=self.model_name)

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[EmbeddingResult]:
        self.load()
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            start = time.time()
            inputs = self.tokenizer(
                batch, return_tensors="pt", truncation=True,
                max_length=128, padding=True,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self._mean_pool(outputs, inputs["attention_mask"]).cpu().numpy()

            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.where(norms > 0, norms, 1)

            elapsed_ms = (time.time() - start) * 1000 / len(batch)
            for emb in embeddings:
                results.append(EmbeddingResult(embedding=emb, inference_time_ms=elapsed_ms,
                                               model_name=self.model_name))
        return results

    def unload(self) -> None:
        if self._loaded:
            del self.model
            del self.tokenizer
            if self.device == "cuda":
                torch.cuda.empty_cache()
            self._loaded = False
