import hashlib
import json

import pandas as pd

from pun_detection.config import DATA


def text_dataset_fingerprint(
    dataframe: pd.DataFrame,
) -> str:
    digest = hashlib.sha256()

    for instance_id, text in dataframe[
        [
            DATA.id_column,
            DATA.text_column,
        ]
    ].itertuples(
        index=False,
        name=None,
    ):
        payload = json.dumps(
            [
                str(instance_id),
                str(text),
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )

        digest.update(
            payload.encode("utf-8")
        )

        digest.update(b"\n")

    return digest.hexdigest()


def supervised_dataset_fingerprint(
    dataframe: pd.DataFrame,
) -> str:
    digest = hashlib.sha256()

    for instance_id, text, label in dataframe[
        [
            DATA.id_column,
            DATA.text_column,
            DATA.label_column,
        ]
    ].itertuples(
        index=False,
        name=None,
    ):
        payload = json.dumps(
            [
                str(instance_id),
                str(text),
                int(label),
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )

        digest.update(
            payload.encode("utf-8")
        )

        digest.update(b"\n")

    return digest.hexdigest()