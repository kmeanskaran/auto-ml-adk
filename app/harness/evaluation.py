"""Evaluation and serving bundles, done by the harness rather than the engineer.

The engineer chooses and tunes each model; the harness fits and scores it, so the
numbers the human decides on are never self-reported. One simple protocol:

  fit a fresh copy on train → score test → the plan's metric picks the winner

The plan's metric is the business one by default: the expected loss per 1,000,
which weighs every mistake by how sure the model was, so two models practically never
tie. The cut-off comes from the costs, not the data: a record is flagged when its
score reaches false_alarm / (false_alarm + missed), e.g. 2 / (2 + 1) = 0.67. The model
that was scored is the model that is served.
Candidates are fitted in parallel, one CPU each.

It also reports calibration (mean score vs outcome rate on test) and, when the split
follows time, a drift report: the features whose test values move the scores most
(without a time order it is sampling noise, so it is skipped).
"""

from __future__ import annotations

import json
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any

from app import settings
from app.harness import catalog, project
from app.harness.environment import ProjectEnvironment, cpus
from app.harness.project import PLAIN_DTYPES, write_json

MODELS_DIR = "artifacts/models"  # the engineer's tuned models
FINAL_DIR = "artifacts/final"  # the harness's fits on train: what gets served
SMOKE_ROWS = 50

EVALUATE = r"""
import glob, json, os, time
import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn import metrics as M
from sklearn.base import clone
__PLAIN__
fd = os.environ["FEATURE_DIR"]
definition = json.load(open(os.path.join(fd, "definition.json")))
outcome, features = definition["outcome_column"], list(definition["features"])
cost_fn, cost_fp = float(os.environ["COST_FN"]), float(os.environ["COST_FP"])
tables = {s: plain(pd.read_parquet(os.path.join(fd, "offline", f"{s}.parquet"))) for s in ("train", "test")}
y = {s: t[outcome].to_numpy().astype(int) for s, t in tables.items()}
wanted = set(json.loads(os.environ.get("MODELS", "[]")))
ordered = bool((definition.get("split") or {}).get("ordered_by"))
# Flag when the expected cost of flagging is the lower one: score >= fp / (fp + fn).
T = round(cost_fp / (cost_fp + cost_fn), 4)

# Expected loss: each record costs its mistake's price times how sure the model was of
# the wrong answer, so a confident mistake costs more than a hesitant one.
def cost(yy, p):
    return 1000 * (cost_fn * np.sum((yy == 1) * (1 - p)) + cost_fp * np.sum((yy == 0) * p)) / len(yy)

def metrics(yy, p, t):
    p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    flag = (p >= t).astype(int)
    ranked = len(np.unique(yy)) > 1
    out = {
        "roc_auc": M.roc_auc_score(yy, p) if ranked else None,
        "pr_auc": M.average_precision_score(yy, p) if ranked else None,
        "f1": M.f1_score(yy, flag, zero_division=0),
        "precision": M.precision_score(yy, flag, zero_division=0),
        "recall": M.recall_score(yy, flag, zero_division=0),
        "accuracy": M.accuracy_score(yy, flag),
        "log_loss": M.log_loss(yy, p, labels=[0, 1]),
        "brier": M.brier_score_loss(yy, p),
        "cost_per_1000": cost(yy, p),
    }
    return {k: (None if v is None else round(float(v), 4)) for k, v in out.items()}

# No model: every record gets the training rate as its score.
flat = np.full(len(y["test"]), float(y["train"].mean()))
baseline = {"threshold": T, "test": metrics(y["test"], flat, T)}

def fitted(model):  # a fresh copy of the engineer's model, fitted on train only
    return clone(model).fit(tables["train"][features], tables["train"][outcome])

def calibration(yy, p):
    return {"mean_score": round(float(p.mean()), 4), "outcome_rate": round(float(yy.mean()), 4)}

# Features whose test values move the scores most: give the test rows this feature's
# values from the training rows and see how far the mean score moves.
def drift(model, top=5):
    if not ordered:  # same period: the shifts would be sampling noise
        return []
    test, ref = tables["test"][features], tables["train"][features]
    base = float(model.predict_proba(test)[:, 1].mean())
    moves = []
    for f in features:
        swapped = test.copy()
        swapped[f] = ref[f].sample(len(test), replace=True, random_state=0).values
        moves.append((f, float(model.predict_proba(swapped)[:, 1].mean()) - base))
    moves.sort(key=lambda m: -abs(m[1]))
    return [{"feature": f, "score_shift": round(d, 4)} for f, d in moves[:top]]

# Fair lending: the raw test rows (artifacts/test.parquet, the rows the feature view's
# test table was built from, in the same order) give each row's protected groups.
protected = json.loads(os.environ.get("PROTECTED", "[]"))
raw_test = None
if protected and os.path.exists("artifacts/test.parquet"):
    raw_test = pd.read_parquet("artifacts/test.parquet").reset_index(drop=True)
    if len(raw_test) != len(tables["test"]):
        raw_test = None

def fairness(yy, p, t):  # per group: rows, share flagged, share of positives flagged
    if raw_test is None:
        return {}
    flag, out = p >= t, {}
    for col in protected:
        if col not in raw_test.columns:
            continue
        groups = raw_test[col].astype("string").fillna("(missing)").to_numpy()
        out[col] = []
        for value in sorted(set(groups)):
            m = groups == value
            pos = m & (yy == 1)
            out[col].append({"group": str(value), "rows": int(m.sum()),
                             "flag_rate": round(float(flag[m].mean()), 4),
                             "recall": round(float(flag[pos].mean()), 4) if pos.any() else None})
    return out

def one_cpu(model):  # candidates run side by side, so each keeps to one CPU
    jobs = {k: 1 for k in model.get_params() if k == "n_jobs" or k.endswith("__n_jobs")}
    return model.set_params(**jobs) if jobs else model

# Honest protocol, enforced here whatever train.py did: fit on train, score test.
def evaluated(path):
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        final = fitted(one_cpu(joblib.load(path)))
        started = time.time()
        pt = final.predict_proba(tables["test"][features])[:, 1]
        ms = 1000 * (time.time() - started) / len(pt) * 1000
        joblib.dump(final, f"artifacts/final/{name}.joblib")
        return name, {"threshold": T, "test": metrics(y["test"], pt, T), "ms_per_1000_rows": round(ms, 2),
                      "calibration": {"test": calibration(y["test"], pt)},
                      "drift": drift(final), "fairness": fairness(y["test"], pt, T)}, None
    except Exception as exc:
        return name, None, (f"cannot be refit by the harness (sklearn.base.clone(model).fit(X, y)): "
                            f"{type(exc).__name__}: {exc}")[:400]

os.makedirs("artifacts/final", exist_ok=True)
paths = [p for p in sorted(glob.glob("artifacts/models/*.joblib"))
         if not wanted or os.path.splitext(os.path.basename(p))[0] in wanted]
done = Parallel(n_jobs=max(1, min(int(os.environ["CPUS"]), len(paths))), prefer="threads")(
    delayed(evaluated)(p) for p in paths)
candidates = {name: result for name, result, _ in done if result is not None}
errors = {name: error for name, _, error in done if error is not None}
print(json.dumps({"candidates": candidates, "errors": errors, "baseline": baseline, "ordered": ordered}))
""".replace("__PLAIN__", PLAIN_DTYPES)

PREDICT = '''"""Serving code, the same for every version: the feature view's build(), then the model.

Generated by the harness. Loads only this version's own files, so several
versions can be served side by side.
"""

import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
__PLAIN__
HOME = Path(__file__).resolve().parent.parent
META = json.loads((HOME / "artifacts" / "model.json").read_text(encoding="utf-8"))
MODEL = joblib.load(HOME / "artifacts" / "model.joblib")
_spec = importlib.util.spec_from_file_location(
    f"features_{META['bundle_id']}", HOME / "src" / "features.py"
)
_features = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_features)


def predict(rows: list[dict]) -> list[dict]:
    raw = pd.DataFrame(rows)
    scores = MODEL.predict_proba(plain(_features.build(raw.copy())[META["features"]]))[:, 1]
    key = META.get("entity_key")
    ids = raw[key].tolist() if key and key in raw.columns else list(range(len(rows)))
    return [
        {
            "id": i.item() if hasattr(i, "item") else i,
            "score": round(float(s), 6),
            "flag": int(s >= META["threshold"]),
        }
        for i, s in zip(ids, scores)
    ]
'''.replace("__PLAIN__", PLAIN_DTYPES)

SMOKE = """
import json, math, os, sys
sys.path.insert(0, os.environ["PREDICT_DIR"])
import pandas as pd
sample = pd.read_csv(os.environ["SMOKE_SAMPLE"], nrows=__ROWS__)
rows = sample[[c for c in sample.columns if not str(c).startswith("Unnamed")]].to_dict("records")
rows = [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()} for r in rows]
try:
    from predict import predict
    out = predict(rows)
    assert isinstance(out, list) and len(out) == len(rows), "predict must return one result per record"
    for item in out:
        assert {"id", "score", "flag"} <= set(item), "each result needs id, score and flag"
        assert 0.0 <= float(item["score"]) <= 1.0, "score must be a probability"
    json.dumps(out)
    print(json.dumps({"ok": True, "scored": len(out)}))
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
""".replace("__ROWS__", str(SMOKE_ROWS))


async def evaluate(
    run: Path, feature_dir: Path, plan: dict[str, Any], costs: tuple[float, float]
) -> tuple[dict[str, Any] | None, list[str]]:
    """Score the plan's candidates; write reports/evaluation.json. Returns (report, problems)."""
    (run / ".home").mkdir(exist_ok=True)
    (run / ".home" / "evaluate.py").write_text(EVALUATE, encoding="utf-8")
    env = {
        "FEATURE_DIR": str(feature_dir),
        "COST_FN": str(costs[0]),
        "COST_FP": str(costs[1]),
        "MODELS": json.dumps(plan["models"]),
        "CPUS": str(cpus()),
        "PROTECTED": json.dumps(list(settings.load().protected)),
    }
    result = await ProjectEnvironment(run, env).execute(
        f"{shlex.quote(sys.executable)} .home/evaluate.py", timeout=1800
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    if not lines:
        return None, [f"evaluation crashed: {(result.stderr or 'no output')[-600:]}"]
    scored = json.loads(lines[-1])
    problems = [f"{name}: {error}" for name, error in scored["errors"].items()]
    problems += [
        f"no model file {MODELS_DIR}/{name}.joblib for planned model {name!r}"
        for name in plan["models"]
        if name not in scored["candidates"] and name not in scored["errors"]
    ]
    metric = plan["metric"]
    best = None
    for name, result_ in scored["candidates"].items():
        if best is None or catalog.better(
            metric,
            result_["test"].get(metric),
            scored["candidates"][best]["test"].get(metric),
        ):
            best = name
    report = {
        "metric": metric,
        "protocol": PROTOCOL,
        "warnings": warnings(scored["candidates"].get(best), scored["ordered"])
        if best
        else [],
        "costs": {"false_negative": costs[0], "false_positive": costs[1]},
        "best": best,
        "baseline": scored["baseline"],
        "candidates": scored["candidates"],
    }
    write_json(run / "reports" / "evaluation.json", report)
    return report, problems


PROTOCOL = (
    "Each model learns from the training rows and is scored once on test rows it never "
    "saw. The best score on the plan's metric wins."
)


CALIBRATION_GAP = 0.2  # mean score off the outcome rate by more than 20% of it
DRIFT_SHIFT = 0.03  # a feature's newer values move the mean score this much
MIN_GROUP_ROWS = 20  # smaller groups are too few to compare
RECALL_GAP = 0.1  # eligible applicants approved: groups differ by more than this


def show(metric: str, value: float) -> str:
    """A metric value the way the console shows it: 41%, 0.864 or 260."""
    kind = catalog.METRICS[metric].kind
    return (
        f"{value:.0%}"
        if kind == "rate"
        else f"{value:.0f}"
        if kind == "cost"
        else f"{value:.3f}"
    )


def warnings(scores: dict[str, Any], ordered: bool = True) -> list[str]:
    """Plain-language checks on the best candidate, for the skeptic and the human.

    Drift is only a warning when the split follows time: on a random split the test
    rows come from the same period, and the drift numbers are sampling noise."""
    w = catalog.words()
    found = []
    test = scores["calibration"]["test"]
    rate, mean = test["outcome_rate"], test["mean_score"]
    if rate and abs(mean - rate) > CALIBRATION_GAP * rate:
        way, flags = ("under", "few") if mean < rate else ("over", "many")
        found.append(
            f"On the final-test {w['record']}s it predicts a {mean:.0%} {w['positive']} "
            f"rate, but the real rate was {rate:.0%}. It {way}estimates the risk, so it "
            f"flags too {flags}."
        )
    found += fairness_flags(scores)
    moved = [
        d
        for d in scores.get("drift", [])
        if ordered and abs(d["score_shift"]) >= DRIFT_SHIFT
    ]
    if moved:
        found.append(
            f"These features behave differently in the final-test {w['record']}s, so the "
            "model may rely on patterns that do not hold up: "
            + ", ".join(d["feature"] for d in moved)
            + "."
        )
    return found


def fairness_flags(scores: dict[str, Any]) -> list[str]:
    """Plain-language fair-lending warnings on one candidate: a group flagged far less
    often than another (below min_ratio, the four-fifths rule), or a group whose
    eligible records are flagged far less often (equal opportunity)."""
    w, ratio_floor = catalog.words(), settings.load().min_group_ratio
    found = []
    for attribute, groups in (scores.get("fairness") or {}).items():
        groups = [g for g in groups if g["rows"] >= MIN_GROUP_ROWS]
        if len(groups) < 2:
            continue
        low = min(groups, key=lambda g: g["flag_rate"])
        high = max(groups, key=lambda g: g["flag_rate"])
        if high["flag_rate"] and low["flag_rate"] / high["flag_rate"] < ratio_floor:
            found.append(
                f"Fair lending, `{attribute}`: {w['positive']} rate {low['flag_rate']:.0%} "
                f"for {low['group']}, {high['flag_rate']:.0%} for {high['group']} "
                f"(ratio {low['flag_rate'] / high['flag_rate']:.2f}, below {ratio_floor})."
            )
        rated = [g for g in groups if g["recall"] is not None]
        if len(rated) >= 2:
            low = min(rated, key=lambda g: g["recall"])
            high = max(rated, key=lambda g: g["recall"])
            if high["recall"] - low["recall"] > RECALL_GAP:
                found.append(
                    f"Fair lending, `{attribute}`: of the {w['record']}s that should get "
                    f"a yes, {low['group']} got one {low['recall']:.0%} of the time, "
                    f"{high['group']} {high['recall']:.0%}."
                )
    return found


def bundle(
    run: Path,
    model: str,
    evaluation: dict[str, Any],
    feature_dir: Path,
    target: Path,
    bundle_id: str,
) -> None:
    """Lay out a self-contained serving bundle: src/ (code) and artifacts/ (model)."""
    definition = json.loads(
        (feature_dir / "definition.json").read_text(encoding="utf-8")
    )
    (target / "src").mkdir(parents=True, exist_ok=True)
    (target / "artifacts").mkdir(exist_ok=True)
    shutil.copyfile(feature_dir / "features.py", target / "src" / "features.py")
    for name in ("data.py", "train.py"):  # kept for lineage; serving never runs them
        if (run / "src" / name).is_file():
            shutil.copy2(run / "src" / name, target / "src" / name)
    (target / "src" / "predict.py").write_text(PREDICT, encoding="utf-8")
    shutil.copy2(
        run / FINAL_DIR / f"{model}.joblib", target / "artifacts" / "model.joblib"
    )
    meta = {
        "bundle_id": bundle_id,
        "model": model,
        "threshold": evaluation["candidates"][model]["threshold"],
        "features": list(definition["features"]),
        # the raw fields the features are built from: /predict names the rest as ignored
        "sources": sorted(
            {s for f in definition["features"].values() for s in f.get("source") or []}
        ),
        "entity_key": definition.get("entity_key"),
        "outcome_column": definition.get("outcome_column"),
    }
    (target / "artifacts" / "model.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


async def smoke(bundle_dir: Path) -> dict[str, Any]:
    """Score production-shaped records (no outcome columns) with a bundle's predict.py."""
    sample = project.traffic_file()
    if sample is None:
        return {"ok": False, "error": "no production sample available"}
    home = bundle_dir / ".home"
    home.mkdir(parents=True, exist_ok=True)
    (home / "smoke.py").write_text(SMOKE, encoding="utf-8")
    env = {"SMOKE_SAMPLE": str(sample), "PREDICT_DIR": str(bundle_dir / "src")}
    result = await ProjectEnvironment(bundle_dir, env).execute(
        f"{shlex.quote(sys.executable)} .home/smoke.py", timeout=300
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    return (
        json.loads(lines[-1])
        if lines
        else {"ok": False, "error": (result.stderr or "no output")[-600:]}
    )
