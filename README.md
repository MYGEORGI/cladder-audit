# Does the text of a task determine its gold answer? Audit of CLadder v1.5

Artifact of the paper: Georgi M.Yu. Does the Task Text Determine Its Gold Answer? Auditing CLadder Through Exact Re-computation. Trudy ISP RAN/Proc. ISP RAS, 2026 (Open Conference of ISP RAS 2026). Cite the paper together with release v1.0 of this repository (`CITATION.cff`).

It contains the verdict for every one of the 10,112 tasks of CLadder v1.5 under two readings of the printed numbers, the solver that computes them, and the scripts that reproduce every result file from the pinned public data.

## Results (`results/`)

| File | Contents |
|---|---|
| `label_readings.csv` | One row per task: `reading_E` (printed numbers exact) and `reading_R` (each printed probability within ±0.005), each `agree`, `contradict` or `undecidable` (the text does not determine the answer); `reason` (`rounding band`, `missing P(V1\|X=1)`, `zero denominator`, `denominator may vanish`); value range of the estimand under each reading (`E_low`..`E_high`, `R_low`..`R_high`; empty where the task has no numeric target or the range is unbounded) |
| `label_readings.json` | Counts of the verdicts, overall and by question type |
| `solvable_subset.txt` | Ids of the 9,580 tasks whose answer the text determines under reading R; score models on these to avoid rounding artefacts |
| `cladder_v15_defects.csv` | Text–structure consistency defects: `source_equation_mismatch` (81 tasks) and `source_alias_collision` (482 tasks of two stories), with details |
| `match_check.json` | Matching of every prompt sentence to a premise: 10,031 tasks pass, 81 are refused for an equation contradiction |
| `solver_check.json` | Solver against the label for every task, with the generator's printed formula where it can be read |
| `band_and_spikein.json` | Census of the rounding band and the spike-in test (labels of agreeing tasks flipped) |
| `ett_meta_check.json` | The 158 ETT tasks on the `confounding` graph against the generator's SCMs |
| `examples.txt` | Worked examples and witnesses (two text-compatible value sets with opposite answers) |
| `repairs.csv` | Every rule-based repair of the `reasoning` field (1,422 `P(X=1=1)` typos, 316 equation operands), as printed and as read, with the result of the prompt check; the prompt and the label are never changed |
| `perturb_check.json` | Prompt corruption test: one deleted probability sentence, one changed percentage, one reversed edge per aligned task; every corruption is rejected |
| `single_model.json` | The 493 rounding-band tasks under a single-model requirement: for each, two parameterizations of one Bayesian network on the task's graph whose printed probabilities lie in the rounding box and whose answers differ, verified in exact arithmetic (420 tasks), or the reason there is none (41 one-answer, 32 boundary) |
| `model_eval/*.jsonl` | Answers of three language models (OpenRouter, temperature 0) to every task, with tokens and cost |
| `model_eval_summary.json` | Their accuracy on all, solvable, undetermined and contradicting tasks, with 95% Wilson intervals |

## Reproduction

Python 3.11 or later. `numpy` and `scipy` are needed only by `single_model.py`; `model_eval.py` calls a paid API.

```
pip install -r requirements.txt
python scripts/fetch_cladder.py            # CLadder v1.5 and the generator files, checked by SHA-256 (inputs.json)
python -B scripts/solver_check.py --out results/solver_check.json
python -B scripts/match_check.py
python -B scripts/readings.py . results
python -B scripts/band_and_spikein.py . results/band_and_spikein.json
python -B scripts/ett_meta_check.py data/raw/cladder/full_v1.5_default.csv data/raw/cladder_meta/data/cladder-v1-meta-models.json results/solver_check.json results/ett_meta_check.json
python -B scripts/examples.py . results > results/examples.txt
python -B scripts/repairs.py
python -B scripts/perturb_check.py
python -B scripts/single_model.py                # needs numpy and scipy (the search only; certificates are exact)
python -B scripts/model_eval.py --model <model>  # paid; needs OPENROUTER_API_KEY; resumes from results/model_eval
python -B scripts/model_eval_summary.py
```

Every audit run is deterministic: two runs give byte-identical files. The exception is `model_eval.py`: a paid API may answer slightly differently when asked again, even at temperature 0, so the model answers in `results/model_eval/` are the recorded ones, and `model_eval_summary.py` reproduces the statistics from them byte for byte. SHA-256 of `results/solver_check.json`: `7ae39c5248e936825fc1d5b44fe7f42ea72912685525bd1ae94406b87d5feef6`. `cladder_v15_defects.csv` is shipped as computed; `examples.py` reads it.

Pinned inputs (`inputs.json`):

- CLadder v1.5, `full_v1.5_default.csv` from `huggingface.co/datasets/causal-nlp/CLadder` at revision `18d684e4`, SHA-256 `d9f74e43…0676` (MIT);
- the CLadder generator files (stories, variable mappings, v1 meta-models) from `github.com/causalNLP/cladder` at commit `4e54416` (MIT).

## Code (`cladder_audit/`)

- `cladder.py` — a CLadder record to atomic premises (graph edges, unobserved variables, probabilities, structural equations) and a question, read from the structured fields; generator typos are repaired by explicit rules;
- `solver.py` — the estimand recipes, evaluated in exact fractions; the rounding interval by enumeration of the box vertices;
- `parsers/`, `transfer.py` — matching of prompt sentences to premises by the generator's verbal forms, and the rename-and-read-back check used by `match_check.py`.

## Versions

- `v0.1-review` (commit cc531bb): the files behind the submitted paper, shared anonymously for double-blind review.
- `v1.0`: adds the revision for the reviewers (repairs list, prompt corruption test, single-model check, model evaluation); every result file of `v0.1-review` is unchanged.

## Licence

Code: MIT (`LICENSE`). Result files in `results/`: CC BY 4.0 (`LICENSE-DATA`). CLadder data and generator files are not redistributed; `fetch_cladder.py` downloads them under their own MIT licence.
