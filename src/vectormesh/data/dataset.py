import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
from datasets import Dataset
from loguru import logger
from tqdm import tqdm


class LabelEncoder:
    """
    Helper to map raw sparse integer codes (504, 508, etc.) to dense indices (0, 1, ..., N) and back.
    Reserves index 0 for 'Unknown' codes.
    """

    def __init__(self, train_codes: List[int]):
        self.unique_codes: List[int] = sorted(list(set(train_codes)))
        self.unknown_idx: int = 0

        self.code2idx: Dict[int, int] = {
            code: i + 1 for i, code in enumerate(self.unique_codes)
        }
        self.idx2code: Dict[int, int] = {
            i + 1: code for i, code in enumerate(self.unique_codes)
        }
        self.idx2code[self.unknown_idx] = 0

    def __len__(self) -> int:
        return len(self.idx2code)

    def onehot(self, codes: List[int]) -> torch.Tensor:
        """Converts a list of raw codes into a Multi-Hot Tensor."""
        # Create a vector of zeros with length equal to total classes
        vector = torch.zeros(len(self), dtype=torch.float32)

        # Set indices to 1 for present codes
        for code in codes:
            if code in self.code2idx:
                vector[self.code2idx[code]] = 1.0
            else:
                # Map to unknown if not found
                vector[self.unknown_idx] = 1.0
        return vector

    def encode(self, codes: list[int]) -> list[int]:
        """Converts a list of raw codes into a list of dense indices."""
        indices = []
        for code in codes:
            if code in self.code2idx:
                indices.append(self.code2idx[code])
            else:
                indices.append(self.unknown_idx)
        return indices

    def decode(self, vector: torch.Tensor, threshold: float) -> List[int]:
        """Converts a Multi-Hot Tensor/Probabilities back to raw codes."""
        indices = (vector > threshold).nonzero(as_tuple=True)[0]
        decoded = []
        for idx in indices:
            idx_val = idx.item()
            try:
                decoded.append(self.idx2code[idx_val])
            except KeyError:
                raise KeyError(f"Index {idx_val} not found in idx2code mapping.")
        return decoded

    def save(self, filepath: Path) -> None:
        """Save the encoder mappings to a JSON file."""
        filepath = Path(filepath)
        data = {
            "code2idx": {str(k): v for k, v in self.code2idx.items()},
            "idx2code": {str(k): v for k, v in self.idx2code.items()},
            "unique_codes": self.unique_codes,
            "unknown_idx": self.unknown_idx,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_file(cls, filepath: str | Path) -> "LabelEncoder":
        """Load an encoder from a JSON file."""
        filepath = Path(filepath)
        with open(filepath, "r") as f:
            data = json.load(f)

        # Create a minimal instance without calling __init__
        encoder = cls.__new__(cls)

        # Restore attributes, converting string keys back to integers
        encoder.code2idx = {int(k): v for k, v in data["code2idx"].items()}
        encoder.idx2code = {int(k): v for k, v in data["idx2code"].items()}
        encoder.unique_codes = data["unique_codes"]
        encoder.unknown_idx = data["unknown_idx"]

        return encoder


def aktes_threshold(file_path: Path, threshold: int) -> Tuple[Dataset, Set[int]]:
    # load texts from jsonl
    texts = []
    with open(file_path) as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            texts.append((record["text"], record["rechtsfeitcodes"]))

    # count all codes
    all_codes = Counter()
    for _, codes in tqdm(texts):
        all_codes.update(set(codes))

    # apply threshold
    codes = {k for k, v in all_codes.items() if v >= threshold}

    # select texts that have the codes over the threshold
    selected = []
    for text, label_list in texts:
        # Keep only codes that passed the threshold
        filtered_labels = [c for c in label_list if c in codes]

        # Only add to dataset if there's at least one label left after filtering
        if filtered_labels:
            selected.append({"text": text, "target": filtered_labels})
    dataset = Dataset.from_list(selected)

    return dataset, codes


def generate_splits(
    path: Path, threshold: int, trainsplit: float, testvalsplit: float
) -> Tuple[Dict[str, Dataset], Set[int]]:
    """
    Args:
        path (Path): location of the jsonl file
        threshold (int): minimum frequency for a code to be included
        trainsplit (float): the fraction of data to use for training
        testvalsplit (float): the fraction of the remaining data (after train split) to use for validation

    Returns:
        Tuple[Dict[str, Dataset], Set[int]]: _description_
    """
    data, codes = aktes_threshold(file_path=path, threshold=threshold)
    full_set = data.train_test_split(train_size=trainsplit)
    train = full_set["train"]

    test_valid = full_set["test"].train_test_split(train_size=testvalsplit)
    valid = test_valid["train"]
    test = test_valid["test"]

    return {
        "train": train,
        "valid": valid,
        "test": test,
    }, codes


def build(
    input_file: Path,
    threshold: int,
    trainsplit: float,
    testvalsplit: float,
    output_dir: Path,
) -> None:
    datasets, codes = generate_splits(
        path=input_file,
        threshold=threshold,
        trainsplit=trainsplit,
        testvalsplit=testvalsplit,
    )
    datasettag = f"theshold_{threshold}_{datasets['train']._fingerprint}"
    datadir = output_dir / Path(f"aktes_{datasettag}")
    datadir = datadir.resolve()
    logger.info(f"Saving processed data to {datadir}")
    if not datadir.exists():
        datadir.mkdir(parents=True, exist_ok=True)
    le = LabelEncoder(codes)
    le_file = Path("labelencoder.json")
    le.save(datadir / le_file)
    for split in ["train", "test", "valid"]:
        dataset = datasets[split]
        labeled = dataset.map(lambda x: {"labels": le.encode(x["target"])})
        filepath = datadir / Path(f"{split}")
        labeled.save_to_disk(filepath)
    logger.success(f"Processed data saved to {datadir}")
