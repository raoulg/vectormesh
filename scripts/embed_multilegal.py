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
    joelniklaus/legal-dutch-roberta-base:

        Trained on MultiLegalPile, a 689GB multilingual corpus covering 24 languages from 17 jurisdictions Hugging FaceHugging Face
        The complete dataset consists of four subsets: Native Multi Legal Pile (112GB), Eurlex Resources (179GB), Legal MC4 (106GB), and Pile of Law (292GB) Hugging Face
        The Dutch-specific model was trained on the Dutch language subset of this massive corpus
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
    model_name = "joelniklaus/legal-dutch-roberta-base"
    vectorizer = Vectorizer(model_name=model_name, col_name="multilegalpile")

    vectorcache = VectorCache.create(
        cache_dir=Path("artefacts"),
        vectorizer=vectorizer,
        dataset=train,
        dataset_tag=tag.name,
    )
    logger.success(f"Created vector cache at: {vectorcache.cache_dir}")


if __name__ == "__main__":
    main()
