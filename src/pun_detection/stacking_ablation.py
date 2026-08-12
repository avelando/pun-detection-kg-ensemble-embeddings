from pun_detection.base_views import (
    BASE_VIEW_NAMES,
    validate_base_view_names,
)


STACKING_ABLATIONS = {
    "tfidf_only": (
        "tfidf",
    ),
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
    "tfidf_selected_embedding": (
        "tfidf",
        "selected_embedding",
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
    "selected_embedding_all_graphs": (
        "selected_embedding",
        "cooccurrence",
        "ppmi",
        "pun_context",
    ),
    "tfidf_all_graphs": (
        "tfidf",
        "cooccurrence",
        "ppmi",
        "pun_context",
    ),
    "all_views": BASE_VIEW_NAMES,
}


def validate_stacking_ablations() -> None:
    if not STACKING_ABLATIONS:
        raise ValueError(
            "Stacking ablations cannot be empty"
        )

    for configuration_name, view_names in (
        STACKING_ABLATIONS.items()
    ):
        if not configuration_name:
            raise ValueError(
                "Stacking ablation name cannot be empty"
            )

        validate_base_view_names(
            view_names
        )

    if STACKING_ABLATIONS[
        "all_views"
    ] != BASE_VIEW_NAMES:
        raise ValueError(
            "All-views ablation must contain "
            "all base views"
        )