from huggingface_hub import hf_hub_download
from transformers import BertTokenizer

from pun_detection.config import FINE_TUNING


def load_fine_tuning_tokenizer():
    vocab_path = hf_hub_download(
        repo_id=FINE_TUNING.model_id,
        filename=FINE_TUNING.tokenizer_vocab_filename,
        revision=FINE_TUNING.revision,
    )

    tokenizer = BertTokenizer(
        vocab=vocab_path,
        do_lower_case=False,
    )

    if tokenizer.vocab_size <= 0:
        raise ValueError(
            "Fine-tuning tokenizer vocabulary is empty"
        )

    required_tokens = (
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
    )

    vocabulary = tokenizer.get_vocab()

    missing_tokens = [
        token
        for token in required_tokens
        if token not in vocabulary
    ]

    if missing_tokens:
        raise ValueError(
            "Fine-tuning tokenizer vocabulary is missing "
            f"required tokens: {missing_tokens}"
        )

    return tokenizer