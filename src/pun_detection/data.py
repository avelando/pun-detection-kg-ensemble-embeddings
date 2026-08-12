import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pun_detection.config import DATA, PATHS


PAIR_ID_PATTERN = re.compile(r"^(?P<pair_id>.+)\.(?P<variant>[HN])$")


@dataclass(frozen=True)
class DevelopmentSplits:
    train: pd.DataFrame
    validation: pd.DataFrame


@dataclass(frozen=True)
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from error

    if not rows:
        raise ValueError(f"Empty dataset: {path}")

    return pd.DataFrame(rows)


def extract_pair_information(instance_id: str) -> tuple[str, str]:
    match = PAIR_ID_PATTERN.match(instance_id)

    if match is None:
        raise ValueError(f"Invalid instance ID: {instance_id}")

    return match.group("pair_id"), match.group("variant")


def validate_split(
    dataframe: pd.DataFrame,
    split_name: str,
    expected_size: int,
) -> pd.DataFrame:
    required_columns = {
        DATA.id_column,
        DATA.text_column,
        DATA.label_column,
        DATA.tokens_column,
        DATA.token_labels_column,
    }

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"{split_name} is missing columns: {sorted(missing_columns)}"
        )

    if len(dataframe) != expected_size:
        raise ValueError(
            f"{split_name} has {len(dataframe)} rows, expected {expected_size}"
        )

    if dataframe[DATA.id_column].duplicated().any():
        duplicated_ids = dataframe.loc[
            dataframe[DATA.id_column].duplicated(),
            DATA.id_column,
        ].tolist()

        raise ValueError(
            f"{split_name} contains duplicated IDs: {duplicated_ids[:10]}"
        )

    labels = set(dataframe[DATA.label_column].astype(int).unique())

    if not labels.issubset({0, 1}):
        raise ValueError(
            f"{split_name} contains invalid labels: {sorted(labels)}"
        )

    if dataframe[DATA.text_column].astype(str).str.strip().eq("").any():
        raise ValueError(f"{split_name} contains empty texts")

    pair_ids = []
    variants = []

    for row in dataframe.itertuples(index=False):
        instance_id = str(getattr(row, DATA.id_column))
        label = int(getattr(row, DATA.label_column))
        tokens = getattr(row, DATA.tokens_column)
        token_labels = getattr(row, DATA.token_labels_column)

        pair_id, variant = extract_pair_information(instance_id)

        expected_label = 1 if variant == "H" else 0

        if label != expected_label:
            raise ValueError(
                f"Label mismatch for {instance_id}: "
                f"label={label}, expected={expected_label}"
            )

        if len(tokens) != len(token_labels):
            raise ValueError(
                f"Token annotation mismatch for {instance_id}"
            )

        pair_ids.append(pair_id)
        variants.append(variant)

    validated = dataframe.copy()
    validated["pair_id"] = pair_ids
    validated["variant"] = variants

    return validated


def validate_instance_boundaries(
    named_splits: dict[str, pd.DataFrame],
) -> None:
    split_names = list(named_splits)

    for left_index, left_name in enumerate(split_names):
        left_ids = set(
            named_splits[left_name][DATA.id_column]
        )

        for right_name in split_names[left_index + 1 :]:
            right_ids = set(
                named_splits[right_name][DATA.id_column]
            )

            if left_ids.intersection(right_ids):
                raise ValueError(
                    f"{left_name} and {right_name} "
                    "contain repeated instance IDs"
                )


def test_access_is_unlocked() -> bool:
    freeze_path = PATHS.experiment_freeze_file

    if not freeze_path.is_file():
        return False

    try:
        with freeze_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            state = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False

    return state.get("status") == "frozen"


def require_test_access() -> None:
    if test_access_is_unlocked():
        return

    raise RuntimeError(
        "Test split is locked until the experimental "
        "configuration is frozen"
    )


def load_train_split() -> pd.DataFrame:
    return validate_split(
        load_jsonl(PATHS.train_file),
        "train",
        DATA.expected_train_size,
    )


def load_validation_split() -> pd.DataFrame:
    return validate_split(
        load_jsonl(PATHS.validation_file),
        "validation",
        DATA.expected_validation_size,
    )


def load_development_splits() -> DevelopmentSplits:
    splits = DevelopmentSplits(
        train=load_train_split(),
        validation=load_validation_split(),
    )

    validate_instance_boundaries(
        {
            "train": splits.train,
            "validation": splits.validation,
        }
    )

    return splits


def load_test_split() -> pd.DataFrame:
    require_test_access()

    return validate_split(
        load_jsonl(PATHS.test_file),
        "test",
        DATA.expected_test_size,
    )


def load_dataset_splits() -> DatasetSplits:
    splits = DatasetSplits(
        train=load_train_split(),
        validation=load_validation_split(),
        test=load_test_split(),
    )

    validate_instance_boundaries(
        {
            "train": splits.train,
            "validation": splits.validation,
            "test": splits.test,
        }
    )

    return splits