import os
import random

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch.optim import AdamW
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from pun_detection.config import (
    DATA,
    FINE_TUNING,
)


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


def set_fine_tuning_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(
        True
    )


def get_fine_tuning_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the configured "
            "fine-tuning experiment"
        )

    return torch.device(
        "cuda"
    )


def tokenize_fine_tuning_dataframe(
    dataframe,
    tokenizer,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        dataframe[
            DATA.text_column
        ].astype(str).tolist(),
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=FINE_TUNING.max_length,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="pt",
    )

    expected_keys = {
        "input_ids",
        "attention_mask",
        "token_type_ids",
    }

    if set(
        encoded
    ) != expected_keys:
        raise ValueError(
            "Unexpected BERTimbau tokenization fields: "
            f"{sorted(encoded)}"
        )

    expected_shape = (
        len(dataframe),
        FINE_TUNING.max_length,
    )

    for tensor in encoded.values():
        if tuple(
            tensor.shape
        ) != expected_shape:
            raise ValueError(
                "Unexpected BERTimbau "
                "tokenized tensor shape"
            )

    return {
        key: tensor.to(
            dtype=torch.long
        )
        for key, tensor in encoded.items()
    }


def make_fine_tuning_dataset(
    dataframe,
    tokenizer,
) -> TensorDataset:
    encoded = (
        tokenize_fine_tuning_dataframe(
            dataframe=dataframe,
            tokenizer=tokenizer,
        )
    )

    labels_array = dataframe[
        DATA.label_column
    ].astype(int).to_numpy(
        copy=True
    )

    labels = torch.from_numpy(
        labels_array
    ).to(
        dtype=torch.long
    )

    if labels.ndim != 1:
        raise ValueError(
            "BERTimbau labels must be "
            "one-dimensional"
        )

    if len(
        labels
    ) != len(
        dataframe
    ):
        raise ValueError(
            "BERTimbau label count mismatch"
        )

    unique_labels = set(
        labels.unique().tolist()
    )

    if (
        not unique_labels
        or not unique_labels.issubset(
            {
                0,
                1,
            }
        )
    ):
        raise ValueError(
            "BERTimbau labels must be binary"
        )

    return TensorDataset(
        encoded[
            "input_ids"
        ],
        encoded[
            "attention_mask"
        ],
        encoded[
            "token_type_ids"
        ],
        labels,
    )


def make_train_loader(
    dataset: TensorDataset,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    return DataLoader(
        dataset,
        batch_size=(
            FINE_TUNING.train_batch_size
        ),
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


def make_evaluation_loader(
    dataset: TensorDataset,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=(
            FINE_TUNING.evaluation_batch_size
        ),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


def load_fine_tuning_model(
    seed: int,
    device: torch.device,
):
    set_fine_tuning_seed(
        seed
    )

    model = (
        BertForSequenceClassification.from_pretrained(
            FINE_TUNING.model_id,
            revision=FINE_TUNING.revision,
            num_labels=(
                FINE_TUNING.num_labels
            ),
            id2label={
                0: "non_pun",
                1: "pun",
            },
            label2id={
                "non_pun": 0,
                "pun": 1,
            },
        )
    )

    if (
        model.config.num_labels
        != FINE_TUNING.num_labels
    ):
        raise ValueError(
            "Unexpected number of "
            "classification labels"
        )

    return model.to(
        device
    )


def make_fine_tuning_optimizer(
    model,
):
    no_decay = (
        "bias",
        "LayerNorm.weight",
    )

    parameter_groups = [
        {
            "params": [
                parameter
                for name, parameter
                in model.named_parameters()
                if not any(
                    excluded in name
                    for excluded in no_decay
                )
            ],
            "weight_decay": (
                FINE_TUNING.weight_decay
            ),
        },
        {
            "params": [
                parameter
                for name, parameter
                in model.named_parameters()
                if any(
                    excluded in name
                    for excluded in no_decay
                )
            ],
            "weight_decay": 0.0,
        },
    ]

    return AdamW(
        parameter_groups,
        lr=FINE_TUNING.learning_rate,
    )


def train_fine_tuning_model(
    model,
    train_loader: DataLoader,
    device: torch.device,
) -> dict:
    optimizer = (
        make_fine_tuning_optimizer(
            model
        )
    )

    total_steps = (
        len(
            train_loader
        )
        * FINE_TUNING.epochs
    )

    warmup_steps = int(
        total_steps
        * FINE_TUNING.warmup_ratio
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=(
                warmup_steps
            ),
            num_training_steps=(
                total_steps
            ),
        )
    )

    epoch_losses = []

    for epoch in range(
        1,
        FINE_TUNING.epochs + 1,
    ):
        model.train()

        cumulative_loss = 0.0
        observed_samples = 0

        for batch in train_loader:
            (
                input_ids,
                attention_mask,
                token_type_ids,
                labels,
            ) = batch

            input_ids = input_ids.to(
                device,
                non_blocking=True,
            )

            attention_mask = (
                attention_mask.to(
                    device,
                    non_blocking=True,
                )
            )

            token_type_ids = (
                token_type_ids.to(
                    device,
                    non_blocking=True,
                )
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )

            loss = outputs.loss

            if (
                loss is None
                or not torch.isfinite(
                    loss
                )
            ):
                raise RuntimeError(
                    "Non-finite fine-tuning loss"
                )

            loss.backward()

            gradient_norm = (
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    FINE_TUNING.gradient_clip_norm,
                )
            )

            if not torch.isfinite(
                gradient_norm
            ):
                raise RuntimeError(
                    "Non-finite fine-tuning "
                    "gradient norm"
                )

            optimizer.step()
            scheduler.step()

            batch_size = labels.shape[
                0
            ]

            cumulative_loss += (
                float(
                    loss.detach().cpu()
                )
                * batch_size
            )

            observed_samples += (
                batch_size
            )

        epoch_loss = (
            cumulative_loss
            / observed_samples
        )

        epoch_losses.append(
            epoch_loss
        )

        print(
            f"epoch={epoch}, "
            f"train_loss="
            f"{epoch_loss:.6f}"
        )

    return {
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "epoch_train_loss": (
            epoch_losses
        ),
    }


def predict_fine_tuning_probabilities(
    model,
    evaluation_loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()

    probability_batches = []

    with torch.inference_mode():
        for batch in evaluation_loader:
            (
                input_ids,
                attention_mask,
                token_type_ids,
                _,
            ) = batch

            outputs = model(
                input_ids=input_ids.to(
                    device,
                    non_blocking=True,
                ),
                attention_mask=(
                    attention_mask.to(
                        device,
                        non_blocking=True,
                    )
                ),
                token_type_ids=(
                    token_type_ids.to(
                        device,
                        non_blocking=True,
                    )
                ),
            )

            probabilities = (
                torch.softmax(
                    outputs.logits,
                    dim=-1,
                )[
                    :,
                    1,
                ]
            )

            probability_batches.append(
                probabilities.detach()
                .cpu()
                .numpy()
            )

    if not probability_batches:
        raise ValueError(
            "Fine-tuned evaluation produced "
            "no predictions"
        )

    result = np.concatenate(
        probability_batches
    ).astype(
        np.float64,
        copy=False,
    )

    if not np.isfinite(
        result
    ).all():
        raise ValueError(
            "Fine-tuned probabilities "
            "contain non-finite values"
        )

    if (
        np.any(
            result < 0.0
        )
        or np.any(
            result > 1.0
        )
    ):
        raise ValueError(
            "Fine-tuned probabilities "
            "are outside [0, 1]"
        )

    return result