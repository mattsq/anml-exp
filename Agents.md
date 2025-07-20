Agents.md   ― spec _version: 0.2.0 (2025-07-20)

Purpose & Non-Goals
This repository is a sandbox for rapid prototyping and benchmarking of anomaly-scoring / detection algorithms in Python.
Out of scope: streaming / online detection, long-horizon model‐monitoring, deployment pipelines, adversarial-attack tooling, and drift-dashboard UIs.

⸻

1 · Big Picture

LLM-powered agents collaborate with human maintainers to
	1.	Implement diverse anomaly-detection models (classical, deep, self-supervised).
	2.	Expose a uniform API so models can be hot-swapped in benchmarks.
	3.	Automate dataset ingestion, metric computation, and result logging.
	4.	Guard code quality with lint, typing, tests, and docs.

Everything else—including production serving, online radiators, or attack toolkits—is outside this spec.

⸻

2 · Canonical Folder Layout

.
├── anomaly_models/          # Source code (importable)
│   ├── __init__.py
│   ├── base.py              # Abstract base (see §3.1)
│   └── isolation_forest.py  # Example reference model
├── datasets/                # Registry & loaders
│   └── registry.py
├── .agents/                 # LLM helper scripts (hidden from casual users)
│   ├── builder.py
│   ├── evaluator.py
│   └── reviewer.py
├── experiments/             # YAML experiment configs
├── results/                 # Auto-generated benchmark outputs (⚠ git-ignored)
├── tests/                   # Pytest & Hypothesis suites
├── docs/                    # Sphinx / MkDocs source
└── results/results-schema.json

Note: non-source artefacts (.agents/, results/) are git-ignored on fresh clones to keep the repo lean.

⸻

3 · Common Schema & Interfaces

3.1 Base Model API

All models must subclass anomaly_models.base.BaseAnomalyModel and implement:

Method/Property	Signature	Requirements
fit	`def fit(self, X: ArrayLike, y: ArrayLike	None = None) -> Self`
score_samples	def score_samples(self, X: ArrayLike) -> NDArray[float]	Higher score ⇒ more anomalous (document if inverted).
predict	`def predict(self, X: ArrayLike, *, threshold: float	None = None) -> NDArray[int]`
decision_threshold	@property def decision_threshold(self) -> float	May raise NotFittedError until fit succeeds.
save / load	optional	Persist model via joblib or similar.

All public APIs require PEP 604 typing and NumPy-style docstrings.

3.2 Dataset Registry

datasets.registry.load_dataset(name: str, split: str = "train") -> Dataset

# in datasets/registry.py
from typing import Tuple, Optional
import numpy as np

NDArray = np.ndarray
Dataset = Tuple[NDArray, Optional[NDArray]]  # (X, y), y=None for unsupervised train

Agents adding datasets must include:
	•	Plain-English description and citation in the docstring.
	•	Download / synth-generation code.
	•	Deterministic split logic (seed = 42 default).

3.3 Metrics & Benchmarks

The evaluator computes:
	•	ROC-AUC
	•	PR-AUC / Average Precision
	•	F1 @ best Youden threshold
	•	Mean wall-time per 1 000 samples (declare hardware string: e.g. "CPU-i7-1185G7" or "GPU-A6000")

Each run is stored at
results/{exp_name}/{model_name}.json and MUST validate against results/results-schema.json.
Minimal schema:

{
  "$schema": "./results-schema.json",
  "dataset": "kddcup99",
  "model": "isolation_forest",
  "n_samples": 145_586,
  "seed": 42,
  "hardware": "CPU-i7-1185G7",
  "roc_auc": 0.921,
  "pr_auc": 0.604,
  "f1": 0.432,
  "threshold": 0.79,
  "fit_time": 1.23,
  "score_time": 0.02,
  "params": {"n_estimators": 100, "max_samples": "auto"}
}


⸻

4 · Agent Roles

Agent	Intent	Typical Prompt Shape	Success Criteria (checked by Reviewer)
Builder	Generate or extend code (models, loaders, utils).	“Implement XYZ model using this paper…”	Implements required API, passes unit tests provided by Builder.
Evaluator	Run benchmarks & aggregate metrics—read-only on source.	“Benchmark all tree-based models on NAB”	Writes valid JSON + Markdown summary; no source diffs.
Reviewer	Gatekeeper: static analysis, typing, docs, tests, perf smoke-run.	Triggered automatically on every PR and after merge into main.	PR passes ruff, mypy --strict, pytest, doc build, and schema validation.


⸻

5 · Contribution Workflow

%%{init: {'theme': 'base'}}%%
flowchart TD
    subgraph Agent PR
        draft["Builder → Draft PR"]
        review["Reviewer → CI checks"]
        maintainer["Human → Merge / Request changes"]
    end
    draft --> review --> maintainer

Only humans may merge. Reviewer also re-runs on main nightly for regression safety.

⸻

6 · Coding Standards
	•	PEP 8 via ruff; project config lives in pyproject.toml.
	•	Type-hints everywhere; mypy --strict required.
	•	NumPy-style docstrings; include math where relevant.
	•	Dependencies

[project]
requires-python = ">=3.11"
dependencies = [
  "numpy",
  "scipy",
  "scikit-learn"
]

[project.optional-dependencies]
torch = ["torch>=2.0"]
pandas = ["pandas>=2.0"]
plot = ["matplotlib>=3.9"]


⸻

7 · Testing Strategy
	•	Unit tests for every public method (fit, score_samples, predict).
	•	Property-based tests (Hypothesis) for ranking invariance on synthetic blobs.
	•	Mutation-testing (mutmut or cosmic-ray) encouraged; if mutation suite passes, strict coverage gates may be waived.
	•	Performance smoke-test (tests/perf/) skipped in CI by default; Reviewer may run locally.

⸻

8 · Quick-Start Example

from anomaly_models.isolation_forest import IsolationForestModel
from datasets.registry import load_dataset
from sklearn.metrics import roc_auc_score

# Load synthetic toy data
X_train, _ = load_dataset("toy-blobs", split="train")
X_test, y_test = load_dataset("toy-blobs", split="test")

# Fit and score
model = IsolationForestModel(n_estimators=200).fit(X_train)
scores = model.score_samples(X_test)

print("ROC-AUC:", roc_auc_score(y_test, scores))

Copy-paste should run as-is with pip install .[torch] skipped.

⸻

9 · Road-Map (Initial)

Milestone	Owner (Agent)	Exit Criteria
M0 – Skeleton	Builder	Base class, dataset registry, CI pipeline, IsolationForest & LOF reference models.
M1 – Classical Benchmark	Evaluator	Run on 3 tabular datasets; JSON + Markdown leaderboard.
M2 – Deep Models	Builder	Implement AutoEncoder, DeepSVDD, minimalist USAD.
M3 – Time-Series Support	Builder + Evaluator	Sliding-window loader + Matrix-Profile (STOMP) + point-wise z-score detector baseline.


⸻

10 · Open Questions
	1.	Unified config system now (omegaconf) or later?
	2.	Preferred experiment tracker for agents (mlflow, wandb, plain JSON)?
	3.	CPU vs GPU determinism in CI—pin BLAS libraries or skip perf tests?
	4.	Sandboxing policy for code-gen agents: Docker-only, --no-network, and syscall allow-list?

⸻

11 · Meta
	•	results-schema.json provides a machine-readable contract for benchmark outputs.
	•	See CONTRIBUTING.md for a human-targeted recap of workflow & coding style.
	•	Spec updates must bump spec_version and include a changelog entry.

⸻

Last updated – 2025-07-20 @ 11:00 AEST.
