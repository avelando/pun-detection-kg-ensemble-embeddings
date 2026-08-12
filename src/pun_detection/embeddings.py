from collections.abc import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from pun_detection.config import (
    EMBEDDINGS,
    EMBEDDING_MODELS,
    EmbeddingModelConfig,
)


def get_embedding_config(
    model_name: str,
) -> EmbeddingModelConfig:
    if model_name not in EMBEDDING_MODELS:
        raise ValueError(
            f"Unknown embedding model: {model_name}"
        )

    return EMBEDDING_MODELS[model_name]


def load_embedding_model(
    model_name: str,
) -> SentenceTransformer:
    config = get_embedding_config(
        model_name
    )

    token = (
        True
        if config.requires_auth
        else None
    )

    model = SentenceTransformer(
        config.model_id,
        revision=config.revision,
        device=EMBEDDINGS.device,
        token=token,
        trust_remote_code=False,
        model_kwargs={
            "torch_dtype": torch.float32,
        },
    )

    model.eval()

    dimension = (
        model.get_embedding_dimension()
    )

    if dimension != config.expected_dimension:
        raise ValueError(
            f"{model_name} has dimension "
            f"{dimension}, expected "
            f"{config.expected_dimension}"
        )

    return model


def validate_embeddings(
    embeddings: np.ndarray,
    expected_rows: int,
    expected_dimension: int,
    normalized: bool,
) -> None:
    expected_shape = (
        expected_rows,
        expected_dimension,
    )

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Embedding shape is "
            f"{embeddings.shape}, expected "
            f"{expected_shape}"
        )

    if embeddings.dtype != np.float32:
        raise ValueError(
            f"Embedding dtype is "
            f"{embeddings.dtype}, expected float32"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embeddings contain non-finite values"
        )

    if normalized:
        norms = np.linalg.norm(
            embeddings,
            axis=1,
        )

        if not np.allclose(
            norms,
            1.0,
            rtol=1e-3,
            atol=1e-3,
        ):
            raise ValueError(
                "Embeddings are not L2 normalized"
            )


def encode_texts(
    model: SentenceTransformer,
    model_name: str,
    texts: Sequence[str],
    show_progress_bar: bool = True,
) -> np.ndarray:
    config = get_embedding_config(
        model_name
    )

    normalized_texts = [
        str(text)
        for text in texts
    ]

    embeddings = model.encode(
        normalized_texts,
        prompt=config.prompt,
        batch_size=config.batch_size,
        show_progress_bar=show_progress_bar,
        precision=EMBEDDINGS.precision,
        convert_to_numpy=True,
        convert_to_tensor=False,
        normalize_embeddings=(
            config.normalize_embeddings
        ),
        device=EMBEDDINGS.device,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    validate_embeddings(
        embeddings=embeddings,
        expected_rows=len(normalized_texts),
        expected_dimension=(
            config.expected_dimension
        ),
        normalized=(
            config.normalize_embeddings
        ),
    )

    return embeddings