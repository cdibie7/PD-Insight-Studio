"""
modeling.py
Optional predictive modelling: classification and regression baselines.
Clearly labelled as exploratory.
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix, mean_absolute_error,
                              mean_squared_error, r2_score)
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
import pickle


def prepare_features(df: pd.DataFrame, feature_cols: List[str],
                     target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """One-hot encode categoricals, impute medians, return X, y."""
    df_work = df[feature_cols + [target_col]].copy()
    df_work = df_work.dropna(subset=[target_col])

    X = df_work[feature_cols].copy()
    y = df_work[target_col]

    # Encode categoricals
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Impute missing with median
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    return X, y


def time_split(df: pd.DataFrame, time_col: str,
               test_pct: float = 0.2) -> Tuple[pd.Index, pd.Index]:
    """Return train/test indices based on time ordering."""
    time_num = pd.to_numeric(df[time_col], errors='coerce')
    sorted_idx = time_num.sort_values().index
    cutoff = int(len(sorted_idx) * (1 - test_pct))
    return sorted_idx[:cutoff], sorted_idx[cutoff:]


def run_classification(df: pd.DataFrame, feature_cols: List[str],
                       target_col: str, time_col: str = None,
                       test_size: float = 0.2,
                       random_state: int = 42) -> Dict[str, Any]:
    """
    Run logistic regression + random forest classifiers.
    Returns result dict with metrics, importance, and confusion matrix.
    """
    X, y = prepare_features(df, feature_cols, target_col)

    # Encode target if categorical
    le = None
    if y.dtype == object or str(y.dtype) == 'category':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)

    n_classes = y.nunique()
    if n_classes < 2:
        return {'error': 'Target has fewer than 2 classes.'}

    # Split
    if time_col and time_col in df.columns:
        train_idx, test_idx = time_split(df.loc[X.index], time_col, test_size)
        train_idx = [i for i in train_idx if i in X.index]
        test_idx = [i for i in test_idx if i in X.index]
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if n_classes <= 10 else None)

    if len(X_train) < 10 or len(X_test) < 5:
        return {'error': 'Not enough data after split.'}

    results = {
        'task': 'classification',
        'n_train': len(X_train),
        'n_test': len(X_test),
        'classes': le.classes_.tolist() if le else list(y.unique()),
        'feature_names': list(X.columns),
        'split_type': 'time-based' if time_col else 'random',
        'models': {},
    }

    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=500, C=1.0, random_state=random_state))
        ]),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state,
                                                 n_jobs=-1, min_samples_leaf=3),
    }

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            model_res = {
                'accuracy': round(accuracy_score(y_test, y_pred), 4),
                'f1_weighted': round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'confusion_labels': list(y.unique()),
            }
            if n_classes == 2:
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_test)[:, 1]
                    else:
                        proba = model.decision_function(X_test)
                    model_res['auc'] = round(roc_auc_score(y_test, proba), 4)
                except Exception:
                    pass

            # Feature importance
            imp = _get_importance(model, X_test, y_test, name)
            if imp:
                model_res['feature_importance'] = imp

            results['models'][name] = model_res
        except Exception as e:
            results['models'][name] = {'error': str(e)}

    results['best_model_name'] = max(
        [(k, v.get('f1_weighted', 0)) for k, v in results['models'].items() if 'error' not in v],
        key=lambda x: x[1], default=('None', 0)
    )[0]

    return results


def run_regression(df: pd.DataFrame, feature_cols: List[str],
                   target_col: str, time_col: str = None,
                   test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """Run Ridge + Random Forest regression."""
    X, y = prepare_features(df, feature_cols, target_col)
    y = pd.to_numeric(y, errors='coerce')
    valid = y.notna()
    X, y = X[valid], y[valid]

    if time_col and time_col in df.columns:
        train_idx, test_idx = time_split(df.loc[X.index], time_col, test_size)
        train_idx = [i for i in train_idx if i in X.index]
        test_idx = [i for i in test_idx if i in X.index]
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

    if len(X_train) < 10 or len(X_test) < 5:
        return {'error': 'Not enough data after split.'}

    results = {
        'task': 'regression',
        'n_train': len(X_train),
        'n_test': len(X_test),
        'target_mean': round(float(y.mean()), 4),
        'feature_names': list(X.columns),
        'split_type': 'time-based' if time_col else 'random',
        'models': {},
    }

    models = {
        'Ridge Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('reg', Ridge(alpha=1.0))
        ]),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state,
                                                n_jobs=-1, min_samples_leaf=3),
    }

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            model_res = {
                'mae': round(mean_absolute_error(y_test, y_pred), 4),
                'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
                'r2': round(r2_score(y_test, y_pred), 4),
            }
            imp = _get_importance(model, X_test, y_test, name)
            if imp:
                model_res['feature_importance'] = imp
            results['models'][name] = model_res
        except Exception as e:
            results['models'][name] = {'error': str(e)}

    return results


def _get_importance(model, X_test, y_test, name: str) -> Optional[Dict]:
    """Extract feature importance or coefficients."""
    try:
        if 'Random Forest' in name:
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
            else:
                imp = model.named_steps.get('clf') or model.named_steps.get('reg')
                imp = imp.feature_importances_ if imp else None
            if imp is not None:
                return dict(sorted(zip(X_test.columns, imp), key=lambda x: -x[1])[:15])
        else:
            # Permutation importance
            pi = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
            return dict(sorted(zip(X_test.columns, pi.importances_mean), key=lambda x: -x[1])[:15])
    except Exception:
        return None


def save_model(model_result: Dict, path: str):
    """Save model result dict as pickle."""
    with open(path, 'wb') as f:
        pickle.dump(model_result, f)


def model_report_md(result: Dict, dataset_name: str = '') -> str:
    """Generate a Markdown report for modelling results."""
    lines = []
    lines.append("# PD Insight Studio — Predictive Model Report (Exploratory)\n")
    lines.append("> ⚠️  **These models are exploratory research tools only.**")
    lines.append("> They have no clinical validity and results may be affected by confounding, small sample sizes, and data quality issues.\n")
    lines.append(f"**Dataset:** {dataset_name}  ")
    lines.append(f"**Task:** {result.get('task', '?')}  ")
    lines.append(f"**Split type:** {result.get('split_type', '?')}  ")
    lines.append(f"**Train n:** {result.get('n_train', '?')} | **Test n:** {result.get('n_test', '?')}\n")

    for model_name, metrics in result.get('models', {}).items():
        lines.append(f"## {model_name}\n")
        if 'error' in metrics:
            lines.append(f"- ❌ Error: {metrics['error']}\n")
            continue
        for k, v in metrics.items():
            if k in ('confusion_matrix', 'feature_importance', 'confusion_labels'):
                continue
            lines.append(f"- **{k.replace('_', ' ').title()}:** {v}")
        if 'feature_importance' in metrics:
            lines.append("\n**Top Feature Importances:**")
            for feat, score in list(metrics['feature_importance'].items())[:10]:
                lines.append(f"  - `{feat}`: {score:.4f}")
        lines.append("")

    return '\n'.join(lines)
