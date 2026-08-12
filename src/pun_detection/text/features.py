from nltk.corpus import stopwords
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    strip_accents_unicode,
)

from pun_detection.config import TFIDF


def load_portuguese_stopwords() -> list[str]:
    words = stopwords.words("portuguese")

    normalized = []

    for word in words:
        word = str(word).lower()

        if TFIDF.strip_accents == "unicode":
            word = strip_accents_unicode(word)

        normalized.append(word)

    return sorted(set(normalized))


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
        strip_accents=TFIDF.strip_accents,
        stop_words=stop_words,
    )