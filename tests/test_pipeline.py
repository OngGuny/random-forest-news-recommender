"""통합 스모크 테스트: 크롤링 제외, 전체 파이프라인 동작 확인."""
from src.recommender.features import extract_meta_features, build_feature_pipeline
from src.recommender.model import train, predict
from src.recommender.preprocess import preprocess_df


def test_full_pipeline_smoke(sample_training_df, sample_prediction_df):
    """학습 → 예측 전체 파이프라인이 에러 없이 완료되는지 확인."""
    # 학습
    train_df = preprocess_df(sample_training_df)
    metrics = train(train_df)

    assert metrics["accuracy"] > 0

    # 예측
    pred_df = preprocess_df(sample_prediction_df)
    results = predict(pred_df, output_format="json")

    assert len(results) == len(sample_prediction_df)
    assert all(r["prediction"] in (0, 1) for r in results)
