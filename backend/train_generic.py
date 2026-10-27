"""
Generic leak-free trainer: one command per dataset.
Put next to train_model.py (it reuses its helpers). Examples:

  python train_generic.py --data data/water_potability.csv --target Potability --out models_potability
  python train_generic.py --data data/Drinking_water.csv --target Potability --drop "Unnamed: 0,Carcinogenics,medical_waste" --out models_drinking
  python train_generic.py --data data/kenya.csv --target Label --positive Safe --max-rows 20000 --models "Logistic Regression,Random Forest" --repeats 1 --out models_kenya

--drop   comma-separated columns to remove (IDs, dates, and any column computed from the target, e.g. WQI)
--positive  label value that counts as "potable/safe" (needed when the target is text)
Results go to <out>/results.json; nothing in models/ is overwritten.
"""
import argparse, json, os, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import train_model as T
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--target", required=True)
ap.add_argument("--positive", default=None); ap.add_argument("--drop", default="")
ap.add_argument("--max-rows", type=int, default=None); ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--models", default="Logistic Regression,Random Forest,Gradient Boosting,XGBoost")
ap.add_argument("--out", default="models_generic")
a = ap.parse_args()

df = pd.read_csv(a.data, encoding="utf-8-sig")
drop = [c.strip() for c in a.drop.split(",") if c.strip()]
df = df.drop(columns=[c for c in drop if c in df.columns])
df = df.dropna(subset=[a.target])
y = (df[a.target].astype(str) == a.positive).astype(int).values if a.positive else df[a.target].astype(int).values
X = df.drop(columns=a.target).select_dtypes("number")
print(f"{len(df)} rows | features: {list(X.columns)} | positive rate {y.mean():.3f}")
if a.max_rows and len(X) > a.max_rows:
    keep, _ = train_test_split(np.arange(len(X)), train_size=a.max_rows, stratify=y, random_state=T.SEED)
    X, y = X.iloc[keep], y[keep]; print(f"subsampled to {len(X)} rows")
X = X.values.astype(float)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=T.SEED)
names = [m for m in a.models.split(",") if m != "XGBoost" or T.HAS_XGB]

res, best = {}, None
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=a.repeats, random_state=T.SEED)
for mn in names:
    for st in T.STRATEGIES:
        aucs = []
        for tr, va in rskf.split(Xtr, ytr):
            imp, sc = T.prep_fit(Xtr[tr])
            m = T.fit_with_strategy(mn, st, T.prep_apply(imp, sc, Xtr[tr]), ytr[tr])
            aucs.append(roc_auc_score(ytr[va], m.predict_proba(T.prep_apply(imp, sc, Xtr[va]))[:, 1]))
        res[f"{mn}|{st}"] = {"cv_auc": round(float(np.mean(aucs)), 4), "cv_auc_std": round(float(np.std(aucs)), 4)}
        print(f"CV {mn:20s} {st:12s} AUC {np.mean(aucs):.4f}±{np.std(aucs):.4f}")
        if best is None or np.mean(aucs) > best[2]: best = (mn, st, np.mean(aucs))

imp, sc = T.prep_fit(Xtr)
m = T.fit_with_strategy(best[0], best[1], T.prep_apply(imp, sc, Xtr), ytr)
p = m.predict_proba(T.prep_apply(imp, sc, Xte))[:, 1]
out = {"data": os.path.basename(a.data), "n_rows": int(len(y)), "positive_rate": round(float(y.mean()), 3),
       "selected": {"model": best[0], "strategy": best[1]}, "test": T.test_metrics(yte, p),
       "test_ci": T.bootstrap_ci(yte, p), "cv": res}
os.makedirs(a.out, exist_ok=True); json.dump(out, open(os.path.join(a.out, "results.json"), "w"), indent=2)
print("\nSELECTED", best[:2], "| TEST", {k: out["test"][k] for k in ("accuracy", "balanced_accuracy", "auc_roc", "mcc")}, out["test_ci"])
