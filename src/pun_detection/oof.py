from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from pun_detection.config import DATA, EXPERIMENT


@dataclass(frozen=True)
class OOFSplit:
    fold: int
    train_indices: np.ndarray
    holdout_indices: np.ndarray


def create_oof_splits(
    dataframe: pd.DataFrame,
    folds: int = EXPERIMENT.oof_folds,
    seed: int = EXPERIMENT.primary_seed,
) -> list[OOFSplit]:
    y = dataframe[DATA.label_column].astype(int).to_numpy()

    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )

    splits = []

    for fold, (train_indices, holdout_indices) in enumerate(
        splitter.split(
            X=np.zeros(len(dataframe)),
            y=y,
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