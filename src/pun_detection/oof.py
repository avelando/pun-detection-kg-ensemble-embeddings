from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from pun_detection.config import DATA, EXPERIMENT


@dataclass(frozen=True)
class OOFSplit:
    fold: int
    train_indices: np.ndarray
    holdout_indices: np.ndarray


def create_oof_splits(
    dataframe: pd.DataFrame,
    folds: int = EXPERIMENT.oof_folds,
    seed: int = EXPERIMENT.oof_split_seed,
) -> list[OOFSplit]:
    y = dataframe[DATA.label_column].astype(int).to_numpy()
    groups = dataframe["pair_id"].astype(str).to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )

    splits = []

    for fold, (train_indices, holdout_indices) in enumerate(
        splitter.split(
            X=np.zeros(len(dataframe)),
            y=y,
            groups=groups,
        )
    ):
        splits.append(
            OOFSplit(
                fold=fold,
                train_indices=train_indices,
                holdout_indices=holdout_indices,
            )
        )

    return splits