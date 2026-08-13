import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from pun_detection.base_views import (
    BASE_VIEW_NAMES,
    BaseViewMatrices,
    validate_base_view_matrix,
)
from pun_detection.config import (
    BASE_MODELS,
    EMBEDDING_MODELS,
    EXPERIMENT,
    GRAPHS,
    PATHS,
    TFIDF,
)
from pun_detection.fingerprints import (
    array_fingerprint,
    json_fingerprint,
    supervised_dataset_fingerprint,
)
from pun_detection.selection import (
    load_embedding_selection,
)
from pun_detection.text.features import (
    load_portuguese_stopwords,
)


BASE_VIEW_CACHE_VERSION = 1


def get_base_view_cache_dir(
    seed: int,
) -> Path:
    return (
        PATHS.base_views_dir
        / f"seed_{seed}"
    )


def get_base_view_cache_paths(
    seed: int,
) -> tuple[Path, Path]:
    cache_dir = get_base_view_cache_dir(
        seed
    )

    return (
        cache_dir / "matrices.npz",
        cache_dir / "metadata.json",
    )


def get_base_view_cache_state(
    seed: int,
) -> str:
    matrices_path, metadata_path = (
        get_base_view_cache_paths(
            seed=seed,
        )
    )

    matrices_exists = (
        matrices_path.is_file()
    )

    metadata_exists = (
        metadata_path.is_file()
    )

    if (
        matrices_exists
        and metadata_exists
    ):
        return "complete"

    if (
        matrices_exists
        or metadata_exists
    ):
        return "partial"

    return "missing"


def base_classifier_config(
    seed: int,
) -> dict:
    return {
        "type": "logistic_regression",
        "C": BASE_MODELS.logistic_c,
        "solver": BASE_MODELS.logistic_solver,
        "max_iter": BASE_MODELS.logistic_max_iter,
        "seed": seed,
    }


def tfidf_config_metadata() -> dict:
    metadata = asdict(
        TFIDF
    )

    if TFIDF.use_portuguese_stopwords:
        stop_words = (
            load_portuguese_stopwords()
        )

        metadata[
            "stopwords_fingerprint"
        ] = json_fingerprint(
            stop_words
        )
    else:
        metadata[
            "stopwords_fingerprint"
        ] = None

    return metadata


def matrix_metadata(
    matrix: np.ndarray,
) -> dict:
    return {
        "shape": list(
            matrix.shape
        ),
        "dtype": str(
            matrix.dtype
        ),
        "fingerprint": array_fingerprint(
            matrix
        ),
    }


def build_base_view_context(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    seed: int,
) -> dict:
    selection = load_embedding_selection(
        train=train,
        validation=validation,
    )

    selected_model = selection[
        "selected_model"
    ]

    embedding_config = asdict(
        EMBEDDING_MODELS[
            selected_model
        ]
    )

    return {
        "artifact_type": "base_view_matrices",
        "version": BASE_VIEW_CACHE_VERSION,
        "seed": seed,
        "oof": {
            "folds": EXPERIMENT.oof_folds,
            "split_seed": seed,
        },
        "columns": list(
            BASE_VIEW_NAMES
        ),
        "selected_embedding_model": (
            selected_model
        ),
        "embedding_selection_fingerprint": (
            json_fingerprint(
                selection
            )
        ),
        "datasets": {
            "train": (
                supervised_dataset_fingerprint(
                    train
                )
            ),
            "validation": (
                supervised_dataset_fingerprint(
                    validation
                )
            ),
        },
        "configuration": {
            "base_classifier": (
                base_classifier_config(
                    seed=seed,
                )
            ),
            "tfidf": (
                tfidf_config_metadata()
            ),
            "graphs": asdict(
                GRAPHS
            ),
            "selected_embedding": (
                embedding_config
            ),
        },
    }


def build_base_view_metadata(
    matrices: BaseViewMatrices,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    seed: int,
) -> dict:
    metadata = build_base_view_context(
        train=train,
        validation=validation,
        seed=seed,
    )

    if (
        matrices.selected_embedding_model
        != metadata[
            "selected_embedding_model"
        ]
    ):
        raise ValueError(
            "Base view matrices use an unexpected "
            "embedding model"
        )

    if matrices.columns != BASE_VIEW_NAMES:
        raise ValueError(
            "Base view matrix columns are invalid"
        )

    validate_base_view_matrix(
        matrix=matrices.train_oof,
        expected_rows=len(train),
        matrix_name="train_oof",
    )

    validate_base_view_matrix(
        matrix=matrices.validation,
        expected_rows=len(validation),
        matrix_name="validation",
    )

    metadata[
        "matrices"
    ] = {
        "train_oof": matrix_metadata(
            matrices.train_oof
        ),
        "validation": matrix_metadata(
            matrices.validation
        ),
    }

    return metadata


def save_base_view_cache(
    matrices: BaseViewMatrices,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    seed: int,
    overwrite: bool = False,
) -> dict:
    matrices_path, metadata_path = (
        get_base_view_cache_paths(
            seed=seed,
        )
    )

    cache_dir = matrices_path.parent

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not overwrite:
        existing_paths = [
            path
            for path in (
                matrices_path,
                metadata_path,
            )
            if path.exists()
        ]

        if existing_paths:
            raise FileExistsError(
                "Base view cache already exists: "
                + ", ".join(
                    str(path)
                    for path in existing_paths
                )
            )

    metadata = build_base_view_metadata(
        matrices=matrices,
        train=train,
        validation=validation,
        seed=seed,
    )

    np.savez_compressed(
        matrices_path,
        train_oof=matrices.train_oof,
        validation=matrices.validation,
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


def load_base_view_cache(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    seed: int,
) -> BaseViewMatrices:
    matrices_path, metadata_path = (
        get_base_view_cache_paths(
            seed=seed,
        )
    )

    if not matrices_path.is_file():
        raise FileNotFoundError(
            f"Missing base view matrices: "
            f"{matrices_path}"
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing base view metadata: "
            f"{metadata_path}"
        )

    try:
        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(
                file
            )
    except json.JSONDecodeError as error:
        raise ValueError(
            "Invalid base view metadata JSON"
        ) from error

    expected_context = build_base_view_context(
        train=train,
        validation=validation,
        seed=seed,
    )

    for key, expected_value in (
        expected_context.items()
    ):
        actual_value = metadata.get(
            key
        )

        if actual_value != expected_value:
            raise ValueError(
                "Base view cache metadata mismatch "
                f"for {key}"
            )

    with np.load(
        matrices_path,
        allow_pickle=False,
    ) as archive:
        expected_files = {
            "train_oof",
            "validation",
        }

        if set(
            archive.files
        ) != expected_files:
            raise ValueError(
                "Base view cache contains "
                "unexpected arrays"
            )

        train_oof = np.array(
            archive[
                "train_oof"
            ],
            copy=True,
        )

        validation_matrix = np.array(
            archive[
                "validation"
            ],
            copy=True,
        )

    validate_base_view_matrix(
        matrix=train_oof,
        expected_rows=len(train),
        matrix_name="train_oof",
    )

    validate_base_view_matrix(
        matrix=validation_matrix,
        expected_rows=len(validation),
        matrix_name="validation",
    )

    expected_matrix_metadata = {
        "train_oof": matrix_metadata(
            train_oof
        ),
        "validation": matrix_metadata(
            validation_matrix
        ),
    }

    if metadata.get(
        "matrices"
    ) != expected_matrix_metadata:
        raise ValueError(
            "Base view matrix fingerprint mismatch"
        )

    return BaseViewMatrices(
        selected_embedding_model=(
            expected_context[
                "selected_embedding_model"
            ]
        ),
        train_oof=train_oof,
        validation=validation_matrix,
        columns=BASE_VIEW_NAMES,
    )