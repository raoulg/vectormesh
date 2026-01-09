"""Text vectorization components using HuggingFace models."""

from typing import Optional, Literal, Any
import torch
from beartype.typing import List
from jaxtyping import jaxtyped
from beartype import beartype as typechecker
from pydantic import ConfigDict, Field, PrivateAttr, model_validator
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, AutoConfig
from datasets import Dataset
from loguru import logger

from vectormesh.types import VectorMeshComponent, VectorMeshError, TwoDTensor, ThreeDTensor, NDTensor
from vectormesh.utils.model_info import get_model_metadata, ModelMetadata

def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

class Vectorizer(VectorMeshComponent):

    model_name: str
    device: str = Field(default_factory=detect_device)

    _metadata: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()
    _stride: int = PrivateAttr()

    
    @model_validator(mode='after')
    def initialize_model(self):
        self._metadata = AutoConfig.from_pretrained(self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        self._stride = self.get_metadata['max_position_embeddings'] // 10 
        logger.info(f"using stride: {self._stride}, based on max_position_embeddings: {self.get_metadata['max_position_embeddings']} // 10")
        return self
    
    def tokenize(self, text: str) -> dict[str, torch.Tensor]:
        max_length = self.get_metadata['max_position_embeddings']
        if max_length is None:
            raise VectorMeshError(
                message=f"Model '{self.model_name}' does not have a max_position_embeddings attribute.",
                hint="Check that the model ID is correct and supports sentence-transformers.",
                fix=f"See: `BaseVectorizer(model_name='{self.model_name}`._metadata for max_poistion_embeddings"
            )
        tokens = self._tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            stride=self._stride,
            return_overflowing_tokens=True,
            return_tensors="pt",
            padding="max_length"
        )
        return tokens 
    
    def embed(self, tokens: dict[str, torch.Tensor], batchsize: int) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            input_ids = tokens["input_ids"].to(self.device)
            attention_mask = tokens["attention_mask"].to(self.device)
            chunks = input_ids.shape[0]
            embs = []
            for i in range(0, chunks, batchsize):
                    input_ids_batch = input_ids[i:i+batchsize]
                    attention_mask_batch = attention_mask[i:i+batchsize]
                    outputs = self._model(input_ids_batch, attention_mask=attention_mask_batch)
                    embs.append(outputs.last_hidden_state)
        return {"embeddings": torch.cat(embs, dim=0), "attention_mask": attention_mask}
    
    def aggregate(self, embedded: dict[str, torch.Tensor]) -> TwoDTensor:
        mask_expand = embedded["attention_mask"].unsqueeze(-1)
        sum_emb = torch.sum(embedded["embeddings"] * mask_expand, dim=1)
        sum_mask = torch.sum(mask_expand, dim=1)
        return sum_emb / sum_mask


    def __call__(self, text: str) -> TwoDTensor:
        tokens = self.tokenize(text)
        embedded = self.embed(tokens, batchsize=16)
        return self.aggregate(embedded)
    
    @property
    def get_model(self):
        return self._model
    
    @property
    def get_tokenizer(self):
        return self._tokenizer
    
    @property
    def get_metadata(self) -> dict:
        config = {
            "hidden_size" : getattr(self._metadata, "hidden_size", getattr(self._metadata, "dim", None)),
            "max_position_embeddings" : getattr(self._metadata, "max_position_embeddings", None),
        }
        return config
    
    @property
    def get_contextsize(self) -> int:
        return self._metadata["max_position_embeddings"]
    
    @property
    def get_hidden_size(self) -> int:
        hidden_size=getattr(self._metadata, "hidden_size", getattr(self._metadata, "dim", None))
        return hidden_size

    def _detect_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

