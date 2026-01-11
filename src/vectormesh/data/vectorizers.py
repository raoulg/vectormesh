"""Text vectorization components using HuggingFace models."""

from abc import ABC, abstractmethod
from collections import Counter
from typing import Any

import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from loguru import logger
from pydantic import Field, PrivateAttr, model_validator
from torch import Tensor
from transformers import AutoConfig, AutoModel, AutoTokenizer

from vectormesh.types import VectorMeshComponent, VectorMeshError


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class BaseVectorizer(VectorMeshComponent, ABC):
    """
    Base class for all vectorizers.

    All vectorizers must:
    - Have a model_name, device, and metadata
    - Implement __call__ that returns dict[str, list[Float[Tensor, "..."]]]
    - The exact tensor dimensionality can vary by implementation
    """

    model_name: str
    col_name: str
    device: str = Field(default_factory=detect_device)

    _metadata: Any = PrivateAttr()

    @abstractmethod
    @model_validator(mode="after")
    def initialize_model(self):
        """
        Initialize the model/API connection.
        Must set self._metadata with at least:
        - hidden_size or dim: output dimension
        - max_position_embeddings (optional): context window size

        For API-based models, this might just set up API keys.
        For local models, this loads the model weights.
        """
        pass

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def __call__(
        self, texts: list[str], batchsize: int
    ) -> dict[str, list[Float[Tensor, "..."]]]:
        """
        Process texts and return embedding.

        Args:
            texts: List of input texts
            batchsize: Batch size for processing

        Returns:
            Dict with '{self.col_name : list[Tensor]}'.
            Tensor dimensionality varies by implementation
        """
        pass

    @property
    def get_metadata(self) -> dict:
        """
        Return metadata about the model.
        Subclasses can override to add more fields.
        """
        return {
            "model_name": self.model_name,
            "col_name": self.col_name,
            "hidden_size": getattr(self._metadata, "hidden_size"),
            "context_size": getattr(self._metadata, "max_position_embeddings"),
        }

    @property
    def get_hidden_size(self) -> int:
        return getattr(self._metadata, "hidden_size")

    @property
    def get_context_size(self) -> int:
        return getattr(self._metadata, "max_position_embeddings")


class Vectorizer(BaseVectorizer):
    model_name: str
    col_name: str
    device: str = Field(default_factory=detect_device)

    _metadata: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()
    _stride: int = PrivateAttr()
    chunk_sizes: Counter = Counter()

    @model_validator(mode="after")
    def initialize_model(self):
        self._metadata = AutoConfig.from_pretrained(self.model_name)
        max_pos = getattr(self._metadata, "max_position_embeddings")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        self._stride = max_pos // 10
        logger.info(f"Using device: {self.device}")
        logger.info(
            f"using stride: {self._stride}, based on max_position_embeddings: {max_pos} // 10"
        )
        return self

    @jaxtyped(typechecker=beartype)
    def tokenize(
        self, text: list[str]
    ) -> tuple[
        Int[Tensor, "batch tokens"],
        Int[Tensor, "batch tokens"],
        Int[Tensor, "batch"],
    ]:
        """
        Receives a batch of texts Σ* , where Σ is an alphabet, and * represents that the strings
        can be concatenated in any order to create a sequence to create sentences.

        The output is:
        - input_ids: A 2D tensor of token ids (batch_size * chunks, max_length) ∈ ℕ
        - attention_mask: A 2D tensor of attention mask (batch_size * chunks, max_length) ∈ {0, 1}
        - overflow_to_sample_mapping: A 1D tensor of document indices (batch_size * chunks,) ∈ ℕ  (eg 0, 0 ,0, 1, 1, 2, ...)

        Because of the context window (eg 512 tokens), we "overflow" the tokens into
        a (batch * chunks, max_length) tensor.
        eg batch might be 32, but some documents are 3 * 512 tokens, others are 5 * 512 tokens, etc.
        So we end up with eg (115, 512) tokens from an input of (32) documents

        We will later reconstruct into
        (chunks, max_length) for each document with the help of the overflow indices
        """
        max_length = getattr(self._metadata, "max_position_embeddings")
        if max_length is None:
            raise VectorMeshError(
                message=f"Model '{self.model_name}' does not have a max_position_embeddings attribute.",
                hint="Check that the model ID is correct and supports sentence-transformers.",
                fix=f"See: `BaseVectorizer(model_name='{self.model_name}`._metadata for max_poistion_embeddings",
            )
        tokens = self._tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            stride=self._stride,
            return_overflowing_tokens=True,
            return_tensors="pt",
            padding="max_length",
        )
        input_ids = tokens["input_ids"]
        attention = tokens["attention_mask"]
        overflow = tokens["overflow_to_sample_mapping"]
        return input_ids, attention, overflow

    @jaxtyped(typechecker=beartype)
    def embed(
        self,
        input_ids: Int[Tensor, "batch tokens"],
        attention: Int[Tensor, "batch tokens"],
        batchsize: int,
    ) -> tuple[Float[Tensor, "batch tokens dim"], Int[Tensor, "batch tokens"]]:
        """
        This function turns a 2D tensor (batch * chunks, tokens) ∈ ℕ  into an embedding
        (batch * chunks, tokens, dim) ∈ ℝ

        The attention mask is used to mask out padding tokens.
        batchsize is the number of chunks to be processed at once
        """
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            attention_mask = attention.to(self.device)
            chunks = input_ids.shape[0]
            embs = []
            for i in range(0, chunks, batchsize):
                input_ids_batch = input_ids[i : i + batchsize]
                attention_mask_batch = attention_mask[i : i + batchsize]
                outputs = self._model(
                    input_ids_batch, attention_mask=attention_mask_batch
                )
                embs.append(outputs.last_hidden_state)
        embeddings = torch.cat(embs, dim=0)

        return embeddings, attention_mask

    @jaxtyped(typechecker=beartype)
    def aggregate(
        self,
        embeddings: Float[Tensor, "batch tokens dim"],
        attention: Int[Tensor, "batch tokens"],
    ) -> Float[Tensor, "batch dim"]:
        """
        This function turns a 3D tensor (batch, tokens, dim) ∈ ℝ
        into an embedding (batch, dim) ∈ ℝ by aggregating over the tokens dimension.

        We can do this because due to the attention mechanism, all
        tokens have been "mixed" like a hologram and
        sort-of contain the information of the full contextwindow.
        """
        mask_expand = attention.unsqueeze(-1)
        sum_emb = torch.sum(embeddings * mask_expand, dim=1)
        sum_mask = torch.sum(mask_expand, dim=1)
        return sum_emb / sum_mask

    @jaxtyped(typechecker=beartype)
    def extend(
        self,
        agg: Float[Tensor, "batch dim"],
        overflow: Int[Tensor, "batch"],
        num_docs: int,
    ) -> dict[str, list[Float[Tensor, "_ dim"]]]:
        """
        With the help of the overflow indices, we can regroup the embeddings back into
        a (chunks, dim) ∈ ℝ tensor per document where chunk varies per document.
        """
        regrouped = []
        for doc_idx in range(num_docs):
            idx = overflow == doc_idx
            embed = agg[idx]
            self.chunk_sizes[embed.shape[0]] += 1
            regrouped.append(embed)
        return {self.col_name: regrouped}

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, texts: list[str], batchsize: int
    ) -> dict[str, list[Float[Tensor, "_ dim"]]]:
        input_ids, attention, overflow = self.tokenize(texts)
        embedded, attention = self.embed(input_ids, attention, batchsize=batchsize)
        agg = self.aggregate(embedded, attention)
        return self.extend(agg, overflow, num_docs=len(texts))

    @property
    def get_model(self):
        return self._model

    @property
    def get_tokenizer(self):
        return self._tokenizer
