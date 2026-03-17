# MLOps Unit 1 – Git & Version Control

A practical introduction to Git and version control workflows tailored for Machine Learning Operations (MLOps).

---

## Table of Contents

1. [Why Version Control Matters in MLOps](#1-why-version-control-matters-in-mlops)
2. [Core Git Concepts](#2-core-git-concepts)
3. [Recommended Repository Structure](#3-recommended-repository-structure)
4. [MLOps Git Workflow](#4-mlops-git-workflow)
5. [Versioning Models and Experiments](#5-versioning-models-and-experiments)
6. [Data Version Control (DVC)](#6-data-version-control-dvc)
7. [Best Practices](#7-best-practices)
8. [Quick-Start Example](#8-quick-start-example)

---

## 1. Why Version Control Matters in MLOps

In traditional software development, version control tracks *code*. In MLOps, you also need to track:

| Artifact | Why it matters |
|---|---|
| **Code** | Reproducible training pipelines |
| **Data** | Identify which dataset produced a model |
| **Models** | Roll back to a previous model version |
| **Experiments** | Compare hyperparameters and metrics |
| **Configurations** | Reproduce the exact environment |

Without version control you cannot answer: *"Which code, data, and config produced this model?"*

---

## 2. Core Git Concepts

### Initialise a repository
```bash
git init my-mlops-project
cd my-mlops-project
```

### Stage and commit changes
```bash
git add .
git commit -m "feat: add baseline logistic regression model"
```

### Branches for experiments
```bash
# Create an experiment branch
git checkout -b experiment/add-feature-scaling

# After evaluation, merge back to main
git checkout main
git merge experiment/add-feature-scaling
```

### Tags for model releases
```bash
# Tag a production-ready model version
git tag -a v1.0.0 -m "Release: accuracy=0.94, f1=0.92"
git push origin v1.0.0
```

### Viewing history
```bash
git log --oneline --graph --decorate
git diff HEAD~1 HEAD -- src/train.py
```

---

## 3. Recommended Repository Structure

```
my-mlops-project/
├── data/
│   ├── raw/           # Original, immutable data (tracked with DVC)
│   └── processed/     # Transformed features (tracked with DVC)
├── models/            # Serialised model artefacts (tracked with DVC)
├── notebooks/         # Exploratory analysis (*.ipynb)
├── src/
│   ├── data_loader.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
│   └── test_train.py
├── configs/
│   └── params.yaml    # Hyperparameters & settings
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md
```

---

## 4. MLOps Git Workflow

A simple, effective branching strategy for ML projects:

```
main          ──●──────────────────●── (production-ready models)
                 \                /
feature/*    ─────●──●──●──●──●──    (new features / data prep)
                        \
experiment/* ────────────●──●──●     (model experiments)
```

### Commit message conventions

Follow [Conventional Commits](https://www.conventionalcommits.org/) to make history easy to scan:

| Prefix | Use case |
|---|---|
| `feat:` | New model architecture or pipeline step |
| `fix:` | Bug fix in data processing or training |
| `data:` | Dataset updates or new data sources |
| `exp:` | Experiment results / hyperparameter changes |
| `refactor:` | Code clean-up with no behaviour change |
| `ci:` | CI/CD pipeline changes |

---

## 5. Versioning Models and Experiments

### Use Git tags as experiment bookmarks

```bash
# After training, record results in a tag
git tag -a exp/v1-lr-baseline \
  -m "model=LogisticRegression lr=0.01 acc=0.87 f1=0.85"

# List all experiment tags
git tag -l "exp/*"

# Compare two experiments
git show exp/v1-lr-baseline
git show exp/v2-rf-tuned
```

### Store hyperparameters in a config file

Track `configs/params.yaml` in Git so every commit captures the exact settings used:

```yaml
# configs/params.yaml
model:
  type: RandomForestClassifier
  n_estimators: 200
  max_depth: 10
training:
  test_size: 0.2
  random_seed: 42
```

---

## 6. Data Version Control (DVC)

Large files (datasets, model binaries) should **not** be stored in Git. Use [DVC](https://dvc.org) instead.

```bash
# Install DVC
pip install dvc

# Initialise DVC alongside Git
dvc init

# Add a dataset to DVC tracking
dvc add data/raw/dataset.csv

# This creates data/raw/dataset.csv.dvc – commit this pointer to Git
git add data/raw/dataset.csv.dvc .gitignore
git commit -m "data: add raw dataset v1"

# Push data to remote storage (S3, GCS, Azure, SSH, …)
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc push
```

To reproduce the exact dataset used in a past experiment:
```bash
git checkout exp/v1-lr-baseline
dvc pull
```

---

## 7. Best Practices

1. **Never commit secrets** – use `.env` files (listed in `.gitignore`) or a secrets manager.
2. **Never commit large binaries** – use DVC or Git LFS for datasets and model files.
3. **Always tag releases** – tag every model that goes to staging or production.
4. **Write meaningful commit messages** – future-you will thank present-you.
5. **Use `.gitattributes`** – enforce consistent line endings and mark binary files.
6. **Pin dependencies** – commit `requirements.txt` or `pyproject.toml` with exact versions.
7. **Automate with CI/CD** – run tests and linting on every push (see `.github/workflows/`).

---

## 8. Quick-Start Example

Clone this repo and run the example training script:

```bash
git clone https://github.com/sajidtecho/MLops-unit-1.git
cd MLops-unit-1

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train.py

# Evaluate the model
python src/evaluate.py
```

After training, inspect what changed:
```bash
git status
git diff configs/params.yaml
git log --oneline
```

---

## License

This repository is released under the [MIT License](LICENSE).
