"""Infer which dataset columns map onto vectormesh's input/label roles.

HuggingFace datasets name their columns inconsistently: the model input may live
in ``text``, ``texts``, ``sentence``, ``image``, ...; the label in ``label``,
``labels``, ``target``, ``targets``, ... Rather than hardcode one name, infer the
mapping once (overridable) and feed the result into the column params the rest of
the library already expects (``vectorizer.input_col``, ``OneHot(label_col=...)``,
``Collate(target_col=...)``).
"""

from typing import Callable, ClassVar, Optional

from datasets import ClassLabel, Dataset, Value
from loguru import logger
from pydantic import BaseModel

from vectormesh.types import VectorMeshError


class DatasetSchema(BaseModel):
    """Maps a HuggingFace dataset's real column names onto vectormesh roles.

    Build one with :meth:`infer` right after ``load_dataset``. Detection is
    name-first (alias priority, case-insensitive); if no alias matches it falls
    back to the column's HuggingFace feature *type* (a string ``Value`` for the
    input, a ``ClassLabel`` for the label). Any role can be pinned explicitly to
    override detection.
    """

    input_col: str
    label_col: Optional[str] = None

    # Priority order: when several are present, the earlier alias wins.
    INPUT_ALIASES: ClassVar[tuple[str, ...]] = (
        "text",
        "texts",
        "sentence",
        "sentences",
        "content",
        "document",
        "image",
        "img",
    )
    LABEL_ALIASES: ClassVar[tuple[str, ...]] = (
        "label",
        "labels",
        "target",
        "targets",
        "class",
        "classes",
        "category",
    )

    @classmethod
    def infer(
        cls,
        dataset: Dataset,
        *,
        input_col: Optional[str] = None,
        label_col: Optional[str] = None,
    ) -> "DatasetSchema":
        """Detect the input and label columns, honoring any explicit override.

        The input column is required (we cannot vectorize nothing); the label
        column is optional, since a dataset may be unlabeled (inference only).

        Args:
            dataset: the loaded HuggingFace dataset (a single split).
            input_col: pin the input column, skipping detection for it.
            label_col: pin the label column, skipping detection for it.

        Raises:
            VectorMeshError: if no input column can be inferred and none is given.
        """
        cols = list(dataset.column_names)
        features = dataset.features

        resolved_input = (
            input_col
            or cls._pick_by_name(cols, cls.INPUT_ALIASES)
            or cls._pick_by_type(
                features, lambda f: isinstance(f, Value) and f.dtype == "string"
            )
        )
        if resolved_input is None:
            raise VectorMeshError(
                f"Could not infer an input column from {cols}.",
                hint=f"None of {cls.INPUT_ALIASES} matched, and no string column was found.",
                fix="Pass input_col=... explicitly to DatasetSchema.infer().",
            )

        resolved_label = (
            label_col
            or cls._pick_by_name(cols, cls.LABEL_ALIASES)
            or cls._pick_by_type(features, lambda f: isinstance(f, ClassLabel))
        )
        if resolved_label is None:
            logger.warning(
                f"No label column inferred from {cols}; continuing without one. "
                "Pass label_col=... if this dataset is actually labeled."
            )

        schema = cls(input_col=resolved_input, label_col=resolved_label)
        logger.info(
            f"Inferred schema: input_col='{schema.input_col}', "
            f"label_col='{schema.label_col}'"
        )
        return schema

    @staticmethod
    def _pick_by_name(cols: list[str], aliases: tuple[str, ...]) -> Optional[str]:
        """First alias present in cols (case-insensitive), original casing kept."""
        lower = {c.lower(): c for c in cols}
        for alias in aliases:
            if alias in lower:
                return lower[alias]
        return None

    @staticmethod
    def _pick_by_type(features, predicate: Callable[[object], bool]) -> Optional[str]:
        """First column whose HuggingFace feature satisfies predicate."""
        for name, feature in features.items():
            if predicate(feature):
                return name
        return None
