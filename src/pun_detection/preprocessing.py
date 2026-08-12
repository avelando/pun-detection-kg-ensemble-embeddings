import re


GRAPH_TOKEN_PATTERN = re.compile(
    r"[^a-záàâãéèêíïóôõöúçñ]+",
    flags=re.IGNORECASE,
)


def normalize_graph_token(token: str) -> str:
    normalized = str(token).lower().strip()
    return GRAPH_TOKEN_PATTERN.sub("", normalized)


def is_valid_graph_token(token: str) -> bool:
    return bool(token) and len(token) > 1


def normalize_graph_tokens(tokens) -> list[str]:
    normalized = [normalize_graph_token(token) for token in tokens]
    return [token for token in normalized if is_valid_graph_token(token)]


def extract_pun_tokens(tokens, token_labels) -> list[str]:
    pun_tokens = []

    for token, label in zip(tokens, token_labels):
        if int(label) != 1:
            continue

        normalized = normalize_graph_token(token)

        if is_valid_graph_token(normalized):
            pun_tokens.append(normalized)

    return pun_tokens
