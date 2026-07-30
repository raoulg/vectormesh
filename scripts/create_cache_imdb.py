import sys
from pathlib import Path

from datasets import load_dataset
from loguru import logger

from vectormesh import VectorCache, Vectorizer

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/embed_imdb.log", rotation="10 MB", level="DEBUG")


def cache_for_model(model_name: str):
    assets = Path("assets/imdb")
    dataset = load_dataset("stanfordnlp/imdb", cache_dir=str(assets))
    tag = model_name.split("/")[-1]

    for mode in ["train", "test"]:
        vectorizer = Vectorizer(model_name=model_name, col_name=tag, max_length=512)

        # data = dataset[mode].select(range(64))

        vectorcache = VectorCache.create(
            cache_dir=Path("artefacts"),
            vectorizer=vectorizer,
            dataset=dataset[mode],
            dataset_tag=f"imdb_{mode}",
        )
        logger.success(f"Created vector cache at: {vectorcache.cache_dir}")


if __name__ == "__main__":
    models = [
        "ibm-granite/granite-embedding-small-english-r2",
        "sentence-transformers/all-MiniLM-L6-v2",
    ]
    for model in models:
        logger.info(f"Creating cache for model: {model}")
        cache_for_model(model)
