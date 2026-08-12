import numpy as np

from pun_detection.config import (
    DATA,
    FINE_TUNING,
)
from pun_detection.data import (
    load_development_splits,
)
from pun_detection.fine_tuning import (
    load_fine_tuning_tokenizer,
)


def summarize_lengths(
    split_name,
    dataframe,
    tokenizer,
):
    encoded = tokenizer(
        dataframe[
            DATA.text_column
        ].astype(str).tolist(),
        add_special_tokens=True,
        truncation=False,
    )

    lengths = np.asarray(
        [
            len(input_ids)
            for input_ids in encoded[
                "input_ids"
            ]
        ],
        dtype=int,
    )

    truncated = int(
        np.sum(
            lengths
            > FINE_TUNING.max_length
        )
    )

    print(
        f"{split_name}: "
        f"rows={len(lengths)}, "
        f"max={lengths.max()}, "
        f"p95={np.percentile(lengths, 95):.0f}, "
        f"p99={np.percentile(lengths, 99):.0f}, "
        f"over_max_length={truncated}"
    )


def main():
    splits = load_development_splits()

    tokenizer = load_fine_tuning_tokenizer()

    print(
        f"model_id={FINE_TUNING.model_id}"
    )

    print(
        f"revision={FINE_TUNING.revision}"
    )

    print(
        f"max_length={FINE_TUNING.max_length}"
    )

    print(
        f"tokenizer_class="
        f"{tokenizer.__class__.__name__}"
    )

    print(
        f"vocab_size="
        f"{tokenizer.vocab_size}"
    )

    summarize_lengths(
        "train",
        splits.train,
        tokenizer,
    )

    summarize_lengths(
        "validation",
        splits.validation,
        tokenizer,
    )


if __name__ == "__main__":
    main()