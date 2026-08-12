import nltk


def main():
    success = nltk.download(
        "stopwords",
        quiet=False,
        raise_on_error=True,
    )

    if not success:
        raise RuntimeError(
            "Failed to install NLTK stopwords"
        )


if __name__ == "__main__":
    main()