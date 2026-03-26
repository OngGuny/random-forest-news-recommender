import json
import logging
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from .config import MODEL_DIR, OUTPUT_DIR, RF_PARAMS
from .features import build_feature_pipeline, extract_meta_features

logger = logging.getLogger(__name__)

MODEL_PATH = MODEL_DIR / "model.joblib"


def train(df: pd.DataFrame) -> dict:
    """전처리 완료된 DataFrame으로 모델을 학습한다.

    필수 컬럼: processed_body, body, title, source, label
    반환: 평가 지표 dict
    """
    df = extract_meta_features(df)

    pipeline = build_feature_pipeline()
    X = pipeline.fit_transform(df)
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)

    # 모델 + 파이프라인 저장
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "pipeline": pipeline}, MODEL_PATH)
    logger.info("모델 저장: %s", MODEL_PATH)

    return metrics


def evaluate(model: RandomForestClassifier, X_test, y_test) -> dict:
    """모델 성능을 평가한다."""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
    }
    logger.info(
        "평가 결과 — Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f",
        metrics["accuracy"], metrics["precision"],
        metrics["recall"], metrics["f1"],
    )
    return metrics


def predict(df: pd.DataFrame, output_format: str = "json") -> list[dict]:
    """전처리 완료된 DataFrame에 대해 예측을 수행한다.

    필수 컬럼: processed_body, body, title, source, url
    """
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    pipeline = saved["pipeline"]

    df = extract_meta_features(df)
    X = pipeline.transform(df)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    results = []
    for i, row in df.iterrows():
        results.append({
            "title": row["title"],
            "url": row["url"],
            "source": row["source"],
            "prediction": int(predictions[i]),
            "probability": round(float(probabilities[i]), 4),
            "label": "👍 추천" if predictions[i] == 1 else "👎 스킵",
        })

    # 결과 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    if output_format == "json":
        output_path = OUTPUT_DIR / f"predictions_{timestamp}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    else:
        output_path = OUTPUT_DIR / f"predictions_{timestamp}.csv"
        pd.DataFrame(results).to_csv(output_path, index=False)

    logger.info("예측 결과 저장: %s (%d건)", output_path, len(results))
    return results
