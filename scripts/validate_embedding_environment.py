import platform
import sys

import sentence_transformers
import torch
import transformers


def main():
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")
    print(f"torch={torch.__version__}")
    print(f"transformers={transformers.__version__}")
    print(
        "sentence_transformers="
        f"{sentence_transformers.__version__}"
    )
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"torch_cuda={torch.version.cuda}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available"
        )

    device = torch.cuda.current_device()

    print(f"cuda_device={device}")
    print(
        "cuda_device_name="
        f"{torch.cuda.get_device_name(device)}"
    )

    properties = torch.cuda.get_device_properties(
        device
    )

    print(
        "cuda_memory_gb="
        f"{properties.total_memory / 1024**3:.2f}"
    )

    left = torch.randn(
        1024,
        1024,
        device="cuda",
    )

    right = torch.randn(
        1024,
        1024,
        device="cuda",
    )

    result = left @ right

    if not torch.isfinite(result).all():
        raise RuntimeError(
            "CUDA computation produced invalid values"
        )

    del left
    del right
    del result

    torch.cuda.empty_cache()

    print(
        "Embedding environment is valid"
    )


if __name__ == "__main__":
    main()