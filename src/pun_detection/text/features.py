from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from pun_detection.config import TFIDF


def load_portuguese_stopwords() -> list[str]:
    return stopwords.words("portuguese")


def make_tfidf_vectorizer() -> TfidfVectorizer:
    stop_words = (
        load_portuguese_stopwords()
        if TFIDF.use_portuguese_stopwords
        else None
    )

    return TfidfVectorizer(
        ngram_range=(
            TFIDF.ngram_min,
            TFIDF.ngram_max,
        ),
        lowercase=TFIDF.lowercase,
        stop_words=stop_words,
    )