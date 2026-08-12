import hashlib
import json

import pandas as pd
import numpy as np

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


def array_fingerprint(
    array: np.ndarray,
) -> str:
    contiguous = np.ascontiguousarray(
        array
    )

    digest = hashlib.sha256()

    digest.update(
        str(
            contiguous.dtype
        ).encode("utf-8")
    )

    digest.update(b"\n")

    shape_payload = json.dumps(
        list(
            contiguous.shape
        ),
        separators=(",", ":"),
    )

    digest.update(
        shape_payload.encode("utf-8")
    )

    digest.update(b"\n")

    digest.update(
        contiguous.tobytes()
    )

    return digest.hexdigest()


def json_fingerprint(
    value,
) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()