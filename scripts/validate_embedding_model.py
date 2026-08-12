import argparse
import gc

import numpy as np
import torch

from pun_detection.config import (
    DATA,
    EMBEDDING_MODELS,
)
from pun_detection.data import load_train_split
from pun_detection.embeddings import (
    encode_texts,
    get_embedding_config,
    load_embedding_model,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(
            EMBEDDING_MODELS.keys()
        ),
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=8,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    train = load_train_split()

    texts = (
        train[
            DATA.text_column
        ]
        .astype(str)
        .head(args.samples)
        .tolist()
    )

    config = get_embedding_config(
        args.model
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = load_embedding_model(
        args.model
    )

    embeddings = encode_texts(
        model=model,
        model_name=args.model,
        texts=texts,
        show_progress_bar=False,
    )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    allocated_gb = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    peak_gb = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    print(
        f"model={args.model}"
    )

    print(
        f"model_id={config.model_id}"
    )

    print(
        f"revision={config.revision}"
    )

    print(
        f"prompt={config.prompt!r}"
    )

    print(
        f"shape={embeddings.shape}"
    )

    print(
        f"dtype={embeddings.dtype}"
    )

    print(
        f"norm_min={norms.min():.6f}"
    )

    print(
        f"norm_max={norms.max():.6f}"
    )

    print(
        f"gpu_allocated_gb={allocated_gb:.3f}"
    )

    print(
        f"gpu_peak_gb={peak_gb:.3f}"
    )

    print(
        f"max_seq_length={model.max_seq_length}"
    )

    del embeddings
    del model

    gc.collect()
    torch.cuda.empty_cache()

    print(
        "Embedding model validation is valid"
    )


if __name__ == "__main__":
    main()