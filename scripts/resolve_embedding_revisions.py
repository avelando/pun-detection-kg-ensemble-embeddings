from huggingface_hub import HfApi


MODEL_IDS = (
    "google/embeddinggemma-300m",
    "intfloat/multilingual-e5-large",
    "Qwen/Qwen3-Embedding-0.6B",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)


def main():
    api = HfApi()

    for model_id in MODEL_IDS:
        try:
            info = api.model_info(
                model_id,
                expand=[
                    "sha",
                    "gated",
                    "lastModified",
                ],
            )

            print(
                f"model={model_id}, "
                f"sha={info.sha}, "
                f"gated={info.gated}, "
                f"last_modified={info.last_modified}"
            )
        except Exception as error:
            print(
                f"model={model_id}, "
                f"error={type(error).__name__}: "
                f"{error}"
            )


if __name__ == "__main__":
    main()