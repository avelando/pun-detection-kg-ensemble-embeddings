from pun_detection.base_views import (
    validate_base_view_names,
)
from pun_detection.config import (
    STACKING,
)


STACKING_ABLATIONS = {
    "selected_embedding_only": (
        "selected_embedding",
    ),
    "cooccurrence_only": (
        "cooccurrence",
    ),
    "ppmi_only": (
        "ppmi",
    ),
    "pun_context_only": (
        "pun_context",
    ),
    "all_graphs": (
        "cooccurrence",
        "ppmi",
        "pun_context",
    ),
    "selected_embedding_cooccurrence": (
        "selected_embedding",
        "cooccurrence",
    ),
    "selected_embedding_ppmi": (
        "selected_embedding",
        "ppmi",
    ),
    "selected_embedding_pun_context": (
        "selected_embedding",
        "pun_context",
    ),
    "without_cooccurrence": (
        "selected_embedding",
        "ppmi",
        "pun_context",
    ),
    "without_ppmi": (
        "selected_embedding",
        "cooccurrence",
        "pun_context",
    ),
    "without_pun_context": (
        "selected_embedding",
        "cooccurrence",
        "ppmi",
    ),
    "all_primary_views": (
        STACKING.primary_views
    ),
}


def validate_stacking_ablations() -> None:
    if not STACKING_ABLATIONS:
        raise ValueError(
            "Stacking ablations cannot be empty"
        )

    primary_view_set = set(
        STACKING.primary_views
    )

    for configuration_name, view_names in (
        STACKING_ABLATIONS.items()
    ):
        if not configuration_name:
            raise ValueError(
                "Stacking ablation name cannot be empty"
            )

        normalized = validate_base_view_names(
            view_names
        )

        if not set(
            normalized
        ).issubset(
            primary_view_set
        ):
            raise ValueError(
                "Stacking ablation contains "
                "a non-primary view"
            )

    if STACKING_ABLATIONS[
        "all_primary_views"
    ] != STACKING.primary_views:
        raise ValueError(
            "All-primary-views ablation is invalid"
        )