import json
from pathlib import Path

from vectormesh import TwoDVectorizer, VectorCache


def load_jsonl_texts(file_path: Path, limit: int = None) -> list[str]:
    """Load text fields from JSONL file.

    Args:
        file_path: Path to JSONL file
        limit: Optional limit on number of records

    Returns:
        List of text strings
    """
    texts = []
    with open(file_path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            record = json.loads(line)
            texts.append(record["text"])
    return texts


def main():
    from vectormesh.zoo.models import Models
    from transformers import AutoConfig, AutoTokenizer, AutoModel

    texts = load_jsonl_texts(Path("assets/train.jsonl"), limit=2)
    print(len(texts[0]))

    zoo = Models.MINILM.value
    zoo = Models.BERT_MULTILINGUAL.value
    print(zoo.model_id)
    config = AutoConfig.from_pretrained(zoo.model_id)
    print(config)
    device = "mps"
    model = AutoModel.from_pretrained(zoo.model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(zoo.model_id)
    tokens = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_position_embeddings,
    ).to(device)
    print(tokens.input_ids.shape)
    embedding = model(**tokens)
    print(embedding.last_hidden_state.shape)
    print(embedding.pooler_output.shape)

    # print(config)


if __name__ == "__main__":
    main()
