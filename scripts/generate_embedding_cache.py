import argparse
import gc

import torch

from pun_detection.config import (
    DATA,
    EMBEDDING_MODELS,
    EMBEDDINGS,
)
from pun_detection.data import (
    load_train_split,
    load_validation_split,
)
from pun_detection.embedding_cache import (
    save_embedding_cache,
)
from pun_detection.embeddings import (
    compute_tokenization_statistics,
    encode_texts,
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
        "--split",
        required=True,
        choices=sorted(
            EMBEDDINGS.allowed_splits
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()


def load_split(
    split_name: str,
):
    if split_name == "train":
        return load_train_split()

    if split_name == "validation":
        return load_validation_split()

    raise ValueError(
        f"Unsupported split: {split_name}"
    )


def main():
    args = parse_args()

    dataframe = load_split(
        args.split
    )

    texts = (
        dataframe[
            DATA.text_column
        ]
        .astype(str)
        .tolist()
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = load_embedding_model(
        args.model
    )

    tokenization_statistics = (
        compute_tokenization_statistics(
            model=model,
            model_name=args.model,
            texts=texts,
        )
    )

    print(
        f"model={args.model}, "
        f"split={args.split}, "
        f"rows={len(dataframe)}, "
        f"max_tokens="
        f"{tokenization_statistics['maximum_tokens']}, "
        f"max_seq_length="
        f"{tokenization_statistics['max_seq_length']}, "
        f"truncated="
        f"{tokenization_statistics['truncated_instances']}"
    )

    embeddings = encode_texts(
        model=model,
        model_name=args.model,
        texts=texts,
        show_progress_bar=True,
    )

    metadata = save_embedding_cache(
        model_name=args.model,
        split_name=args.split,
        dataframe=dataframe,
        embeddings=embeddings,
        max_seq_length=int(
            model.max_seq_length
        ),
        tokenization_statistics=(
            tokenization_statistics
        ),
        overwrite=args.force,
    )

    peak_gb = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    print(
        f"shape={embeddings.shape}, "
        f"dtype={embeddings.dtype}, "
        f"gpu_peak_gb={peak_gb:.3f}"
    )

    print(
        "dataset_fingerprint="
        f"{metadata['dataset_fingerprint']}"
    )

    print(
        "embeddings_fingerprint="
        f"{metadata['embeddings_fingerprint']}"
    )

    del embeddings
    del model

    gc.collect()
    torch.cuda.empty_cache()

    print(
        "Embedding cache generation is valid"
    )


if __name__ == "__main__":
    main()