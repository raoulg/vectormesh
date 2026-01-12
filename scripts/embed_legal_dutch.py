import sys
from pathlib import Path

from datasets import Dataset, load_from_disk
from loguru import logger

from vectormesh import VectorCache, Vectorizer

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/embed_legal_dutch.log", rotation="10 MB", level="DEBUG")


def main():
    """
    Gerwin/legal-bert-dutch-english:

        Trained on 184,000 legal documents including regulations, decisions, directives, and parliamentary questions in both Dutch and English PromptLayer
        Built on top of mBERT with 60,000 training steps PromptLayer
        Tested on a proprietary dataset of 8,000 long legal documents (2,000 Dutch & 6,000 English) with 30 classes Hugging Face
        Also evaluated on Multi-EURLEX task
    """
    assets = Path("assets")
    if not assets.exists():
        logger.error(f"Assets folder not found at: {assets.resolve()}")
        raise FileNotFoundError(f"Assets folder not found at: {assets.resolve()}")
    tag = next(assets.glob("aktes_*/"))
    trainpath = tag / "train"
    if not trainpath.exists():
        logger.error(f"Train dataset not found at: {trainpath}")
        raise FileNotFoundError(f"Train dataset not found at: {trainpath}")
    logger.info(f"Loading data from: {trainpath}")

    train = load_from_disk(trainpath)
    assert type(train) is Dataset
    model_name = "Gerwin/legal-bert-dutch-english"
    vectorizer = Vectorizer(model_name=model_name, col_name="legal_dutch")

    vectorcache = VectorCache.create(
        cache_dir=Path("artefacts"),
        vectorizer=vectorizer,
        dataset=train,
        dataset_tag=tag.name,
    )
    logger.success(f"Created vector cache at: {vectorcache.cache_dir}")


if __name__ == "__main__":
    main()
