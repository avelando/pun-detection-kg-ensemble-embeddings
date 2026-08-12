from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PathConfig:
    project_root: Path = PROJECT_ROOT
    corpus_dir: Path = PROJECT_ROOT / "corpus"
    train_file: Path = PROJECT_ROOT / "corpus" / "train.jsonl"
    validation_file: Path = PROJECT_ROOT / "corpus" / "validation.jsonl"
    test_file: Path = PROJECT_ROOT / "corpus" / "test.jsonl"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    base_views_dir: Path = PROJECT_ROOT / "artifacts" / "base_views"
    experiment_state_dir: Path = PROJECT_ROOT / "artifacts" / "experiment"
    experiment_freeze_file: Path = (
        PROJECT_ROOT / "artifacts" / "experiment" / "freeze.json"
    )
    graphs_dir: Path = PROJECT_ROOT / "artifacts" / "graphs"
    oof_graphs_dir: Path = PROJECT_ROOT / "artifacts" / "graphs" / "oof"
    full_train_graphs_dir: Path = PROJECT_ROOT / "artifacts" / "graphs" / "full_train"
    embeddings_dir: Path = PROJECT_ROOT / "artifacts" / "embeddings"
    models_dir: Path = PROJECT_ROOT / "artifacts" / "models"
    results_dir: Path = PROJECT_ROOT / "results"
    validation_results_dir: Path = PROJECT_ROOT / "results" / "validation"
    embedding_selection_file: Path = (
        PROJECT_ROOT / "results" / "validation" / "embedding_selection.json"
    )
    stacking_selection_file: Path = (
        PROJECT_ROOT / "results" / "validation" / "stacking_selection.json"
    )
    test_results_dir: Path = PROJECT_ROOT / "results" / "test"
    predictions_dir: Path = PROJECT_ROOT / "results" / "predictions"
    statistics_dir: Path = PROJECT_ROOT / "results" / "statistics"
    fine_tuned_models_dir: Path = (
        PROJECT_ROOT
        / "artifacts"
        / "models"
        / "bertimbau_finetuned"
    )


@dataclass(frozen=True)
class DataConfig:
    id_column: str = "id"
    text_column: str = "text"
    label_column: str = "label"
    tokens_column: str = "tokens"
    token_labels_column: str = "labels"
    expected_train_size: int = 3990
    expected_validation_size: int = 570
    expected_test_size: int = 1140


@dataclass(frozen=True)
class GraphConfig:
    cooccurrence_window: int = 5
    cooccurrence_min_frequency: int = 5
    cooccurrence_top_k: int = 30
    ppmi_window: int = 5
    ppmi_min_frequency: int = 5
    ppmi_top_k: int = 30
    pun_context_min_frequency: int = 3
    pun_context_top_k: int = 40
    svd_dimensions: int = 4
    coverage_features: int = 2


@dataclass(frozen=True)
class ExperimentConfig:
    oof_folds: int = 5
    oof_split_seed: int = 40
    primary_seed: int = 40
    seeds: tuple[int, ...] = (13, 21, 40, 42, 73)
    primary_metric: str = "macro_f1"


@dataclass(frozen=True)
class TfidfConfig:
    ngram_min: int = 1
    ngram_max: int = 2
    lowercase: bool = True
    strip_accents: str | None = "unicode"
    use_portuguese_stopwords: bool = True


@dataclass(frozen=True)
class BaseModelConfig:
    logistic_c: float = 1.0
    logistic_max_iter: int = 2000
    logistic_solver: str = "lbfgs"


@dataclass(frozen=True)
class ReferenceBaselineConfig:
    rf_estimators: int = 100
    rf_criterion: str = "entropy"
    rf_max_depth: int = 15
    lr_max_iter: int = 2000
    svm_c: float = 1.0
    svm_kernel: str = "rbf"
    voting: str = "soft"


@dataclass(frozen=True)
class StackingConfig:
    candidates: tuple[str, ...] = (
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    )
    random_forest_estimators: int = 100
    random_forest_criterion: str = "gini"
    random_forest_max_depth: int | None = None
    random_forest_max_features: str | None = "sqrt"
    gradient_boosting_estimators: int = 100
    gradient_boosting_learning_rate: float = 0.1
    gradient_boosting_max_depth: int = 3
    gradient_boosting_subsample: float = 1.0


@dataclass(frozen=True)
class FineTuningConfig:
    model_id: str = (
        "neuralmind/bert-base-portuguese-cased"
    )
    revision: str = (
        "26a1ec84a2da79680610baa1a143f341be7beeb8"
    )
    tokenizer_vocab_filename: str = "vocab.txt"
    max_length: int = 128
    train_batch_size: int = 8
    evaluation_batch_size: int = 32
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    num_labels: int = 2


@dataclass(frozen=True)
class EmbeddingModelConfig:
    model_id: str
    revision: str
    expected_dimension: int
    prompt: str | None
    normalize_embeddings: bool
    batch_size: int
    requires_auth: bool


@dataclass(frozen=True)
class EmbeddingRuntimeConfig:
    device: str = "cuda"
    precision: str = "float32"
    allowed_splits: tuple[str, ...] = (
        "train",
        "validation",
    )


PATHS = PathConfig()
DATA = DataConfig()
GRAPHS = GraphConfig()
EXPERIMENT = ExperimentConfig()
TFIDF = TfidfConfig()
BASE_MODELS = BaseModelConfig()
REFERENCE_BASELINE = ReferenceBaselineConfig()
STACKING = StackingConfig()
FINE_TUNING = FineTuningConfig()
EMBEDDINGS = EmbeddingRuntimeConfig()

EMBEDDING_MODELS = {
    "embeddinggemma": EmbeddingModelConfig(
        model_id="google/embeddinggemma-300m",
        revision="57c266a740f537b4dc058e1b0cda161fd15afa75",
        expected_dimension=768,
        prompt="task: classification | query: ",
        normalize_embeddings=True,
        batch_size=16,
        requires_auth=True,
    ),
    "e5": EmbeddingModelConfig(
        model_id="intfloat/multilingual-e5-large",
        revision="3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",
        expected_dimension=1024,
        prompt="query: ",
        normalize_embeddings=True,
        batch_size=16,
        requires_auth=False,
    ),
    "qwen": EmbeddingModelConfig(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        expected_dimension=1024,
        prompt=(
            "Instruct: Represent a Portuguese text for binary "
            "classification of whether it contains a pun.\n"
            "Query:"
        ),
        normalize_embeddings=True,
        batch_size=8,
        requires_auth=False,
    ),
    "paraphrase": EmbeddingModelConfig(
        model_id=(
            "sentence-transformers/"
            "paraphrase-multilingual-mpnet-base-v2"
        ),
        revision="4328cf26390c98c5e3c738b4460a05b95f4911f5",
        expected_dimension=768,
        prompt=None,
        normalize_embeddings=True,
        batch_size=32,
        requires_auth=False,
    ),
}