from sklearn.linear_model import LogisticRegression

from pun_detection.config import BASE_MODELS


def make_logistic_classifier(
    seed: int,
) -> LogisticRegression:
    return LogisticRegression(
        C=BASE_MODELS.logistic_c,
        max_iter=BASE_MODELS.logistic_max_iter,
        solver=BASE_MODELS.logistic_solver,
        random_state=seed,
    )