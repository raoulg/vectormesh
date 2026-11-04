"""Chunking of long texts: every token of a document lands in some chunk.

`Vectorizer` and `ChunkedRegexVectorizer` split a text longer than the context window
into overlapping chunks with the tokenizer's `return_overflowing_tokens`. These tests
run offline on a tiny word-level tokenizer and a one-layer BERT saved to `tmp_path`,
so they exercise the real `tokenizers` backend the course installs.
"""

from itertools import pairwise
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from vectormesh import ChunkedRegexVectorizer, Vectorizer
from vectormesh.data.vectorizers import (
    build_legal_reference_pattern,
    harmonize_legal_reference,
)

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
WORDS = [f"w{i}" for i in range(400)]
MAX_LENGTH = 40  # 38 content tokens per chunk after [CLS] and [SEP]
STRIDE = MAX_LENGTH // Vectorizer.STRIDE_DIVISOR  # 4 tokens of overlap
CONTENT = MAX_LENGTH - 2


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A word-level BERT tokenizer and a tiny BERT, saved like a hub checkpoint."""
    path = tmp_path_factory.mktemp("tiny-bert")
    vocab = {token: i for i, token in enumerate(SPECIALS + WORDS)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.BertProcessing(
        ("[SEP]", vocab["[SEP]"]), ("[CLS]", vocab["[CLS]"])
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        model_max_length=MAX_LENGTH,
    )
    tokenizer.save_pretrained(path)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=16,
        max_position_embeddings=MAX_LENGTH,
    )
    torch.manual_seed(0)
    BertModel(config).save_pretrained(path)
    return path


def expected_chunks(n_words: int) -> int:
    """Windows of CONTENT tokens, each starting CONTENT - STRIDE after the last."""
    if n_words <= CONTENT:
        return 1
    step = CONTENT - STRIDE
    return 1 + -(-(n_words - CONTENT) // step)


@pytest.fixture(scope="module")
def vectorizer(model_dir: Path) -> Vectorizer:
    return Vectorizer(model_name=str(model_dir), col_name="emb", device="cpu")


def test_every_token_lands_in_a_chunk(vectorizer: Vectorizer):
    """A 400-word text yields 12 overlapping chunks that cover all 400 words."""
    assert vectorizer.get_stride == STRIDE
    text = " ".join(WORDS)
    input_ids, attention, overflow = vectorizer.tokenize([text])

    assert input_ids.shape[0] == expected_chunks(len(WORDS))
    assert (overflow == 0).all()

    special = set(vectorizer.get_tokenizer.all_special_ids)
    chunks = [
        [t for t in row[mask.bool()].tolist() if t not in special]
        for row, mask in zip(input_ids, attention)
    ]
    covered = {t for chunk in chunks for t in chunk}
    assert covered == set(vectorizer.get_tokenizer.convert_tokens_to_ids(WORDS))
    for prev, nxt in pairwise(chunks):
        assert prev[-STRIDE:] == nxt[:STRIDE]


def test_chunk_counts_per_document_in_a_batch(vectorizer: Vectorizer):
    """Short and long texts in one batch each keep all their chunks."""
    lengths = [400, 10, 200, CONTENT, CONTENT + 1]
    texts = [" ".join(WORDS[:n]) for n in lengths]
    out = vectorizer(texts, batchsize=8)["emb"]
    assert [t.shape[0] for t in out] == [expected_chunks(n) for n in lengths]


def test_chunked_regex_covers_the_whole_text(model_dir: Path):
    """ChunkedRegexVectorizer's chunks, mapped back to text, reach the last word."""
    regex = ChunkedRegexVectorizer(
        pattern_builder=build_legal_reference_pattern,
        harmonizer=harmonize_legal_reference,
        min_doc_frequency=1,
        max_features=2,
        training_texts=["artikel 1 Boek 3 BW"],
        tokenizer_name=str(model_dir),
        context_size=MAX_LENGTH,
        stride=STRIDE,
        device="cpu",
    )
    text = " ".join(WORDS)
    chunk_texts, overflow = regex._chunk_texts([text])
    assert len(chunk_texts) == expected_chunks(len(WORDS))
    assert set(overflow) == {0}
    assert chunk_texts[0].startswith("w0 ")
    assert chunk_texts[-1].endswith(WORDS[-1])
