import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sentence_transformers
import torch
import transformers

from pun_detection.config import (
    DATA,
    EMBEDDINGS,
    PATHS,
)
from pun_detection.embeddings import (
    get_embedding_config,
    validate_embeddings,
)


def validate_embedding_split(
    split_name: str,
) -> None:
    if split_name not in EMBEDDINGS.allowed_splits:
        raise ValueError(
            f"Embedding split is not allowed: {split_name}"
        )


def dataset_fingerprint(
    dataframe: pd.DataFrame,
) -> str:
    digest = hashlib.sha256()

    for row in dataframe.itertuples(index=False):
        instance_id = str(
            getattr(row, DATA.id_column)
        )

        text = str(
            getattr(row, DATA.text_column)
        )

        payload = json.dumps(
            [instance_id, text],
            ensure_ascii=False,
            separators=(",", ":"),
        )

        digest.update(
            payload.encode("utf-8")
        )

        digest.update(b"\n")

    return digest.hexdigest()


def embeddings_fingerprint(
    embeddings: np.ndarray,
) -> str:
    contiguous = np.ascontiguousarray(
        embeddings,
        dtype=np.float32,
    )

    return hashlib.sha256(
        contiguous.tobytes()
    ).hexdigest()


def get_embedding_cache_dir(
    model_name: str,
    split_name: str,
) -> Path:
    validate_embedding_split(
        split_name
    )

    get_embedding_config(
        model_name
    )

    return (
        PATHS.embeddings_dir
        / model_name
        / split_name
    )


def build_embedding_metadata(
    model_name: str,
    split_name: str,
    dataframe: pd.DataFrame,
    embeddings: np.ndarray,
    max_seq_length: int,
    tokenization_statistics: dict,
) -> dict:
    config = get_embedding_config(
        model_name
    )

    return {
        "model_name": model_name,
        "model_id": config.model_id,
        "model_revision": config.revision,
        "split": split_name,
        "rows": len(dataframe),
        "embedding_dimension": (
            config.expected_dimension
        ),
        "dtype": str(
            embeddings.dtype
        ),
        "precision": EMBEDDINGS.precision,
        "normalize_embeddings": (
            config.normalize_embeddings
        ),
        "prompt": config.prompt,
        "batch_size": config.batch_size,
        "max_seq_length": max_seq_length,
        "dataset_fingerprint": (
            dataset_fingerprint(
                dataframe
            )
        ),
        "embeddings_fingerprint": (
            embeddings_fingerprint(
                embeddings
            )
        ),
        "tokenization": (
            tokenization_statistics
        ),
        "runtime": {
            "python": (
                sys.version.split()[0]
            ),
            "platform": (
                platform.platform()
            ),
            "torch": torch.__version__,
            "torch_cuda": (
                torch.version.cuda
            ),
            "transformers": (
                transformers.__version__
            ),
            "sentence_transformers": (
                sentence_transformers.__version__
            ),
        },
    }


def save_embedding_cache(
    model_name: str,
    split_name: str,
    dataframe: pd.DataFrame,
    embeddings: np.ndarray,
    max_seq_length: int,
    tokenization_statistics: dict,
    overwrite: bool = False,
) -> dict:
    config = get_embedding_config(
        model_name
    )

    validate_embedding_split(
        split_name
    )

    validate_embeddings(
        embeddings=embeddings,
        expected_rows=len(dataframe),
        expected_dimension=(
            config.expected_dimension
        ),
        normalized=(
            config.normalize_embeddings
        ),
    )

    output_dir = get_embedding_cache_dir(
        model_name,
        split_name,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings_path = (
        output_dir
        / "embeddings.npy"
    )

    metadata_path = (
        output_dir
        / "metadata.json"
    )

    if not overwrite:
        existing_paths = [
            path
            for path in (
                embeddings_path,
                metadata_path,
            )
            if path.exists()
        ]

        if existing_paths:
            raise FileExistsError(
                "Embedding cache already exists: "
                + ", ".join(
                    str(path)
                    for path in existing_paths
                )
            )

    np.save(
        embeddings_path,
        embeddings,
        allow_pickle=False,
    )

    metadata = build_embedding_metadata(
        model_name=model_name,
        split_name=split_name,
        dataframe=dataframe,
        embeddings=embeddings,
        max_seq_length=max_seq_length,
        tokenization_statistics=(
            tokenization_statistics
        ),
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    return metadata


def load_embedding_cache(
    model_name: str,
    split_name: str,
    dataframe: pd.DataFrame,
) -> np.ndarray:
    config = get_embedding_config(
        model_name
    )

    output_dir = get_embedding_cache_dir(
        model_name,
        split_name,
    )

    embeddings_path = (
        output_dir
        / "embeddings.npy"
    )

    metadata_path = (
        output_dir
        / "metadata.json"
    )

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Missing embedding cache: "
            f"{embeddings_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing embedding metadata: "
            f"{metadata_path}"
        )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    expected_dataset_fingerprint = (
        dataset_fingerprint(
            dataframe
        )
    )

    expected_values = {
        "model_name": model_name,
        "model_id": config.model_id,
        "model_revision": config.revision,
        "split": split_name,
        "rows": len(dataframe),
        "embedding_dimension": (
            config.expected_dimension
        ),
        "dtype": "float32",
        "precision": EMBEDDINGS.precision,
        "normalize_embeddings": (
            config.normalize_embeddings
        ),
        "prompt": config.prompt,
        "batch_size": config.batch_size,
        "dataset_fingerprint": (
            expected_dataset_fingerprint
        ),
    }

    for key, expected_value in (
        expected_values.items()
    ):
        actual_value = metadata.get(
            key
        )

        if actual_value != expected_value:
            raise ValueError(
                f"Embedding cache metadata mismatch "
                f"for {key}: "
                f"actual={actual_value!r}, "
                f"expected={expected_value!r}"
            )

    embeddings = np.load(
        embeddings_path,
        allow_pickle=False,
    )

    validate_embeddings(
        embeddings=embeddings,
        expected_rows=len(dataframe),
        expected_dimension=(
            config.expected_dimension
        ),
        normalized=(
            config.normalize_embeddings
        ),
    )

    actual_embeddings_fingerprint = (
        embeddings_fingerprint(
            embeddings
        )
    )

    expected_embeddings_fingerprint = (
        metadata.get(
            "embeddings_fingerprint"
        )
    )

    if (
        actual_embeddings_fingerprint
        != expected_embeddings_fingerprint
    ):
        raise ValueError(
            "Embedding cache fingerprint mismatch"
        )

    return embeddings