import numpy as np
import pandas as pd

from pun_detection.config import DATA
from pun_detection.data import extract_pair_information


def counterpart_id(
    instance_id: str,
) -> str:
    pair_id, variant = extract_pair_information(
        instance_id
    )

    counterpart_variant = (
        "N"
        if variant == "H"
        else "H"
    )

    return (
        f"{pair_id}."
        f"{counterpart_variant}"
    )


def twin_in_reference_mask(
    dataframe: pd.DataFrame,
    reference: pd.DataFrame,
) -> np.ndarray:
    reference_ids = set(
        reference[
            DATA.id_column
        ].astype(str)
    )

    return np.array(
        [
            counterpart_id(
                str(instance_id)
            )
            in reference_ids
            for instance_id
            in dataframe[
                DATA.id_column
            ]
        ],
        dtype=bool,
    )