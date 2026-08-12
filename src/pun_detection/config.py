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
    graphs_dir: Path = PROJECT_ROOT / "artifacts" / "graphs"
    oof_graphs_dir: Path = PROJECT_ROOT / "artifacts" / "graphs" / "oof"
    full_train_graphs_dir: Path = PROJECT_ROOT / "artifacts" / "graphs" / "full_train"
    embeddings_dir: Path = PROJECT_ROOT / "artifacts" / "embeddings"
    models_dir: Path = PROJECT_ROOT / "artifacts" / "models"
    results_dir: Path = PROJECT_ROOT / "results"
    validation_results_dir: Path = PROJECT_ROOT / "results" / "validation"
    test_results_dir: Path = PROJECT_ROOT / "results" / "test"
    predictions_dir: Path = PROJECT_ROOT / "results" / "predictions"
    statistics_dir: Path = PROJECT_ROOT / "results" / "statistics"


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


PATHS = PathConfig()
DATA = DataConfig()
GRAPHS = GraphConfig()
EXPERIMENT = ExperimentConfig()
TFIDF = TfidfConfig()
BASE_MODELS = BaseModelConfig()
REFERENCE_BASELINE = ReferenceBaselineConfig()