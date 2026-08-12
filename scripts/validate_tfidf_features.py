from pun_detection.config import DATA
from pun_detection.data import load_train_split
from pun_detection.oof import create_oof_splits
from pun_detection.text.features import make_tfidf_vectorizer


def main():
    train = load_train_split()
    splits = create_oof_splits(train)

    for split in splits:
        fold_train = train.iloc[
            split.train_indices
        ]

        fold_holdout = train.iloc[
            split.holdout_indices
        ]

        vectorizer = make_tfidf_vectorizer()

        X_train = vectorizer.fit_transform(
            fold_train[
                DATA.text_column
            ].astype(str)
        )

        X_holdout = vectorizer.transform(
            fold_holdout[
                DATA.text_column
            ].astype(str)
        )

        if X_train.shape[1] != X_holdout.shape[1]:
            raise ValueError(
                f"Fold {split.fold} feature dimensions differ"
            )

        if X_train.shape[1] == 0:
            raise ValueError(
                f"Fold {split.fold} has empty vocabulary"
            )

        print(
            f"fold={split.fold}, "
            f"train={X_train.shape}, "
            f"holdout={X_holdout.shape}, "
            f"vocabulary={len(vectorizer.vocabulary_)}"
        )

    print(
        "TF-IDF feature extraction is valid"
    )


if __name__ == "__main__":
    main()