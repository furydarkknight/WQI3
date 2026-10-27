"""
Leak-free training script for the Water Potability app (drop-in replacement for backend/train_model.py).

What changed versus the original:
  1. Imputer + scaler are fit on the TRAINING split only (original fit them on all data).
  2. SMOTE / class weights are applied INSIDE each CV training fold, never before CV
     (original ran CV on SMOTE'd data, which inflated CV accuracy by ~10 points).
  3. Model AND resampling strategy are selected by repeated stratified CV on the training
     split only. The test set is used once, at the end (original picked the best model by test AUC).
  4. Reports balanced accuracy, MCC, and a bootstrap 95% CI, not just accuracy.
  5. Saves the same artifacts app.py expects (best_model.pkl, imputer.pkl, scaler.pkl,
     feature_names.json, model_results.json), so no app changes are required.

Run:  python train_model.py
"""
import os, json, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, brier_score_loss, confusion_matrix,
                             balanced_accuracy_score, matthews_corrcoef)

warnings.filterwarnings("ignore")
SEED = 42
STRATEGIES = ["none", "class_weight", "smote"]

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE as _ImbSMOTE
    HAS_IMB = True
except ImportError:
    HAS_IMB = False


def smote_resample(X, y, seed=SEED, k=5):
    """SMOTE via imbalanced-learn if installed, else a small equivalent implementation."""
    if HAS_IMB:
        return _ImbSMOTE(random_state=seed).fit_resample(X, y)
    rng = np.random.RandomState(seed)
    X, y = np.asarray(X), np.asarray(y)
    cls, cnt = np.unique(y, return_counts=True)
    minority, need = cls[cnt.argmin()], cnt.max() - cnt.min()
    Xm = X[y == minority]
    idx = NearestNeighbors(n_neighbors=k + 1).fit(Xm).kneighbors(Xm, return_distance=False)[:, 1:]
    i = rng.randint(0, len(Xm), need)
    j = idx[i, rng.randint(0, k, need)]
    synth = Xm[i] + rng.rand(need, 1) * (Xm[j] - Xm[i])
    return np.vstack([X, synth]), np.r_[y, np.full(need, minority)]


def make_model(name):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=SEED)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                          subsample=0.8, random_state=SEED)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="logloss", random_state=SEED, n_jobs=-1)
    raise ValueError(name)


def fit_with_strategy(model_name, strategy, Xa, ya):
    """Apply the imbalance strategy to (Xa, ya) -- always TRAINING data only -- and fit."""
    m = make_model(model_name)
    if strategy == "smote":
        Xa, ya = smote_resample(Xa, ya)
        m.fit(Xa, ya)
    elif strategy == "class_weight":
        if model_name in ("Logistic Regression", "Random Forest"):
            m.set_params(class_weight="balanced"); m.fit(Xa, ya)
        elif model_name == "XGBoost":
            m.set_params(scale_pos_weight=(ya == 0).sum() / (ya == 1).sum()); m.fit(Xa, ya)
        else:  # Gradient Boosting
            m.fit(Xa, ya, sample_weight=compute_sample_weight("balanced", ya))
    else:
        m.fit(Xa, ya)
    return m


def prep_fit(X_tr):
    imp = SimpleImputer(strategy="mean").fit(X_tr)
    sc = MinMaxScaler().fit(imp.transform(X_tr))
    return imp, sc


def prep_apply(imp, sc, X):
    return sc.transform(imp.transform(X))


def cv_score(model_name, strategy, X_tr, y_tr):
    """Repeated stratified CV; imputer/scaler/resampling are all fit inside each training fold."""
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    aucs, accs = [], []
    for a, b in rskf.split(X_tr, y_tr):
        imp, sc = prep_fit(X_tr[a])
        Xa, Xb = prep_apply(imp, sc, X_tr[a]), prep_apply(imp, sc, X_tr[b])
        m = fit_with_strategy(model_name, strategy, Xa, y_tr[a])
        p = m.predict_proba(Xb)[:, 1]
        aucs.append(roc_auc_score(y_tr[b], p)); accs.append(accuracy_score(y_tr[b], p > 0.5))
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(accs)), float(np.std(accs))


def test_metrics(y_true, p):
    pred = (p > 0.5).astype(int)
    return {
        "accuracy": round(accuracy_score(y_true, pred) * 100, 2),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, pred) * 100, 2),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, pred, zero_division=0), 4),
        "mcc": round(matthews_corrcoef(y_true, pred), 4),
        "auc_roc": round(roc_auc_score(y_true, p), 4),
        "brier_score": round(brier_score_loss(y_true, p), 4),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def bootstrap_ci(y_true, p, n=1000):
    rng = np.random.RandomState(SEED)
    accs, aucs = [], []
    for _ in range(n):
        i = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[i])) < 2:
            continue
        accs.append(accuracy_score(y_true[i], p[i] > 0.5)); aucs.append(roc_auc_score(y_true[i], p[i]))
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"accuracy_ci95": q(accs), "auc_ci95": q(aucs)}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(here, "data", "water_potability.csv"), encoding="utf-8-sig")
    y = df["Potability"].values.astype(int)
    X = df.drop(columns="Potability")
    feature_names = list(X.columns)
    X = X.values.astype(float)  # NaNs kept: imputation happens inside folds

    # Split BEFORE any preprocessing or resampling.
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"Train {len(y_tr)} | Test {len(y_te)} | class balance (train): {np.bincount(y_tr).tolist()}")

    model_names = ["Logistic Regression", "Random Forest", "Gradient Boosting"] + (["XGBoost"] if HAS_XGB else [])

    # 1) Model x strategy selection by CV on the training split only.
    cv_table, best_per_model = {}, {}
    for mn in model_names:
        best = None
        for st in STRATEGIES:
            auc, auc_sd, acc, acc_sd = cv_score(mn, st, X_tr, y_tr)
            cv_table[f"{mn} | {st}"] = {"cv_auc": round(auc, 4), "cv_auc_std": round(auc_sd, 4),
                                        "cv_acc": round(acc * 100, 2), "cv_acc_std": round(acc_sd * 100, 2)}
            print(f"  CV  {mn:20s} {st:12s} AUC {auc:.3f}±{auc_sd:.3f}  acc {acc*100:.1f}±{acc_sd*100:.1f}")
            if best is None or auc > best[1]:
                best = (st, auc, acc, acc_sd)
        best_per_model[mn] = best

    # 2) Final fit on the full training split; evaluate on the untouched test split ONCE.
    imp, sc = prep_fit(X_tr)
    Xtr_s, Xte_s = prep_apply(imp, sc, X_tr), prep_apply(imp, sc, X_te)
    results, fitted, probs = {}, {}, {}
    for mn in model_names:
        st = best_per_model[mn][0]
        m = fit_with_strategy(mn, st, Xtr_s, y_tr)
        p = m.predict_proba(Xte_s)[:, 1]
        results[mn] = {**test_metrics(y_te, p), "strategy": st}
        fitted[mn], probs[mn] = m, p
        print(f"  TEST {mn:20s} [{st}] acc {results[mn]['accuracy']}  AUC {results[mn]['auc_roc']}  "
              f"balAcc {results[mn]['balanced_accuracy']}")

    best_name = max(model_names, key=lambda n: best_per_model[n][1])  # chosen by CV AUC, not test
    best_model = fitted[best_name]
    results["best_model_name"] = best_name
    results["best_strategy"] = best_per_model[best_name][0]
    results["cv_mean"] = round(best_per_model[best_name][2] * 100, 2)   # CV accuracy (app expects this key)
    results["cv_std"] = round(best_per_model[best_name][3] * 100, 2)
    results["cv_auc_mean"] = round(best_per_model[best_name][1], 4)
    results["test_ci"] = bootstrap_ci(y_te, probs[best_name])
    results["cv_table"] = cv_table
    results["protocol"] = ("80/20 stratified split; imputer/scaler/resampling fit on training data only "
                           "(inside each fold for CV); 5x3 repeated stratified CV; selection by CV AUC; "
                           "test set evaluated once.")
    if hasattr(best_model, "feature_importances_"):
        fi = dict(zip(feature_names, [round(float(v), 4) for v in best_model.feature_importances_]))
        results["feature_importance"] = dict(sorted(fi.items(), key=lambda kv: kv[1], reverse=True))

    out = os.path.join(here, "models"); os.makedirs(out, exist_ok=True)
    for fname, obj in [("best_model.pkl", best_model), ("imputer.pkl", imp), ("scaler.pkl", sc)]:
        with open(os.path.join(out, fname), "wb") as f:
            pickle.dump(obj, f)
    with open(os.path.join(out, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)
    with open(os.path.join(out, "model_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBest (by CV AUC): {best_name} [{results['best_strategy']}] | "
          f"test acc {results[best_name]['accuracy']}% | AUC {results[best_name]['auc_roc']} | "
          f"95% CI {results['test_ci']}")


if __name__ == "__main__":
    main()
