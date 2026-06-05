# Pun Detection in Portuguese with Stacking

This repository contains the code used in the experiments from the paper on **automatic pun detection in Portuguese** using **TF-IDF**, **lexical graphs**, and **stacking**.

## Structure

```text
.
├── corpus/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
├── outputs/
│   ├── KG_cooc.gpickle
│   ├── KG_ppmi.gpickle
│   └── KG_puncontext.gpickle
├── ensemble_stacking.py
├── ensemble_stacking_e5large.py
├── ensemble_stacking_gemma.py
├── ensemble_stacking_paraphrase.py
├── ensemble_stacking_qwen.py
└── README.md
```

## Folders

### `corpus/`
Contains the training, validation, and test files from the **Puntuguese** corpus.

Source:
- https://github.com/Superar/Puntuguese

### `outputs/`
Contains the graphs used in the experiments:
- `KG_cooc.gpickle` – co-occurrence graph
- `KG_ppmi.gpickle` – graph weighted with PPMI
- `KG_puncontext.gpickle` – pun-context graph

Source:
- https://github.com/liara-ifpi/pun-detection-kg-ensemble

## Code files

### `ensemble_stacking.py`
Base version of the experiment with:
- TF-IDF
- graphs (`cooc`, `ppmi`, `pun`, `all`)
- stacking with a classical ensemble

### `ensemble_stacking_gemma.py`
Version using embeddings from:
- `google/embeddinggemma-300m`

### `ensemble_stacking_paraphrase.py`
Version using embeddings from:
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

### `ensemble_stacking_e5large.py`
Version using embeddings from:
- `intfloat/multilingual-e5-large`

### `ensemble_stacking_qwen.py`
Version using embeddings from:
- `Qwen/Qwen3-Embedding-0.6B`

## Summary

All scripts:
- load the data from `corpus/`
- load the graphs from `outputs/`
- use `TF-IDF`
- train base classifiers with:
  - Random Forest
  - Logistic Regression
  - SVM
- combine predictions with **stacking**

The `gemma`, `paraphrase`, `e5large`, and `qwen` versions mainly differ in the **embedding model** they use.

## Main dependencies

- numpy
- pandas
- networkx
- nltk
- scipy
- scikit-learn
- sentence-transformers
- torch

## Running the code

### Base experiment
```bash
python ensemble_stacking.py
```

### Experiments with embeddings
```bash
python ensemble_stacking_gemma.py
python ensemble_stacking_paraphrase.py
python ensemble_stacking_e5large.py
python ensemble_stacking_qwen.py
```

## Note

The corpus and the graphs are not generated in this repository: they must be manually placed in the `corpus/` and `outputs/` folders, respectively.
