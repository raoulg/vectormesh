import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set

import torch
from datasets import Dataset
from loguru import logger
from tqdm import tqdm
import json
from collections import Counter


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

    def encode(self, codes: List[int]) -> torch.Tensor:
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

    def decode(self, vector: torch.Tensor) -> List[int]:
        """Converts a Multi-Hot Tensor/Probabilities back to raw codes."""
        # Get indices where value is high (assuming threshold 0.5 for logits)
        indices = (vector > 0.5).nonzero(as_tuple=True)[0]
        decoded = []
        for idx in indices:
            idx_val = idx.item()
            if idx_val in self.idx2code:
                decoded.append(self.idx2code[idx_val])
            else:
                # Should not happen if logic is correct, but safe fallback
                decoded.append(0)
        return decoded


# class RechtsfeitDataset():
#     def __init__(self, hf_dataset: HFDataset, encoder: LabelEncoder):
#         self.dataset = hf_dataset
#         self.encoder = encoder

#     def __len__(self) -> int:
#         return len(self.dataset)

#     def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
#         item = self.dataset[idx]
#         text: str = item["text"]
#         raw_codes: List[int] = item["rechtsfeitcodes"]

#         # Transform raw codes to Multi-Hot Tensor
#         label_tensor: torch.Tensor = self.encoder.encode(raw_codes)

#         return text, label_tensor

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
    codes = {k for k,v in all_codes.items() if v >= threshold}

    # select texts that have the codes over the threshold
    selected = []
    for text, label_list in texts:
        # Keep only codes that passed the threshold
        filtered_labels = [c for c in label_list if c in codes]
        
        # Only add to dataset if there's at least one label left after filtering
        if filtered_labels:
            selected.append({
                "text": text, 
                "target": filtered_labels
            })
    dataset = Dataset.from_list(selected)
        
    return dataset, codes


def generate_splits(path: Path, threshold: int, trainsplit: float, testvalsplit: float) -> Tuple[Dict[str, Dataset], Set[int]]:
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

    
    
