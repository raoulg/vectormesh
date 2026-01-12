import sys
from pathlib import Path

from vectormesh.data.dataset import build

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/build_dataset.log", rotation="10 MB", level="DEBUG")

if __name__ == "__main__":
    input_file = Path("assets/aktes.jsonl").resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Input file {input_file} does not exist.")
    threshold = 50
    trainsplit = 0.8
    testvalsplit = 0.5
    output_dir = Path("assets/").resolve()

    build(
        input_file=input_file,
        threshold=threshold,
        trainsplit=trainsplit,
        testvalsplit=testvalsplit,
        output_dir=output_dir,
    )
    logger.success("Dataset build complete.")
