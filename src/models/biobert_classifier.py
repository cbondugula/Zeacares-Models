"""
BioBERT Disease Classifier
Model: dmis-lab/biobert-base-cased-v1.2
Trained on: PubMed abstracts + PMC full-text (4.5B words)
"""
import logging
import time
from dataclasses import dataclass
from typing import Optional
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"


@dataclass
class EmbeddingResult:
    embedding: np.ndarray
    inference_time_ms: float
    model_name: str


class BioBERTClassifier:
    def __init__(self, device: Optional[str] = None):
        self.model_name = MODEL_NAME
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        logger.info(f"Loading BioBERT from {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self._loaded = True
        logger.info("BioBERT loaded")

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
            # Mean pooling over token embeddings
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        elapsed_ms = (time.time() - start) * 1000
        return EmbeddingResult(
            embedding=embedding,
            inference_time_ms=elapsed_ms,
            model_name=self.model_name,
        )

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[EmbeddingResult]:
        self.load()
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            start = time.time()
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            elapsed_ms = (time.time() - start) * 1000 / len(batch)
            for emb in embeddings:
                results.append(EmbeddingResult(
                    embedding=emb,
                    inference_time_ms=elapsed_ms,
                    model_name=self.model_name,
                ))
        return results

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def classify_disease(self, disease_text: str, icd10_descriptions: list[str],
                          icd10_codes: list[str]) -> tuple[str, float]:
        query_emb = self.embed(disease_text).embedding
        candidate_embs = [self.embed(d).embedding for d in icd10_descriptions]
        similarities = [self.cosine_similarity(query_emb, c) for c in candidate_embs]
        best_idx = int(np.argmax(similarities))
        return icd10_codes[best_idx], similarities[best_idx]

    def unload(self) -> None:
        if self._loaded:
            del self.model
            del self.tokenizer
            if self.device == "cuda":
                torch.cuda.empty_cache()
            self._loaded = False
