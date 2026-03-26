import joblib
import pytest

from src.recommender.model import train, predict, MODEL_PATH
from src.recommender.preprocess import preprocess_df


class TestTrain:
    def test_train_returns_metrics(self, sample_training_df):
        df = preprocess_df(sample_training_df)
        metrics = train(df)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    def test_metrics_in_range(self, sample_training_df):
        df = preprocess_df(sample_training_df)
        metrics = train(df)

        for key, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{key}={value} is out of range"

    def test_model_saved(self, sample_training_df):
        df = preprocess_df(sample_training_df)
        train(df)

        assert MODEL_PATH.exists()
        saved = joblib.load(MODEL_PATH)
        assert "model" in saved
        assert "pipeline" in saved


class TestPredict:
    def test_predict_output(self, sample_training_df, sample_prediction_df):
        # 먼저 학습
        train_df = preprocess_df(sample_training_df)
        train(train_df)

        # 예측
        pred_df = preprocess_df(sample_prediction_df)
        results = predict(pred_df, output_format="json")

        assert len(results) == 2
        for r in results:
            assert "title" in r
            assert "prediction" in r
            assert "probability" in r
            assert r["prediction"] in (0, 1)
            assert 0.0 <= r["probability"] <= 1.0

    def test_predict_csv_format(self, sample_training_df, sample_prediction_df):
        train_df = preprocess_df(sample_training_df)
        train(train_df)

        pred_df = preprocess_df(sample_prediction_df)
        results = predict(pred_df, output_format="csv")

        from src.recommender.config import OUTPUT_DIR
        assert (OUTPUT_DIR / "predictions.csv").exists()
