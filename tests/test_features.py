import pandas as pd

from src.recommender.features import extract_meta_features, build_feature_pipeline
from src.recommender.preprocess import preprocess_df


class TestExtractMetaFeatures:
    def test_adds_meta_columns(self):
        df = pd.DataFrame({
            "title": ["테스트 제목"],
            "body": ["본문 내용입니다. 두 번째 문장입니다."],
        })
        result = extract_meta_features(df)
        assert "body_len" in result.columns
        assert "sent_count" in result.columns
        assert "title_len" in result.columns

    def test_body_len_value(self):
        df = pd.DataFrame({"title": ["제목"], "body": ["12345"]})
        result = extract_meta_features(df)
        assert result["body_len"].iloc[0] == 5

    def test_title_len_value(self):
        df = pd.DataFrame({"title": ["안녕하세요"], "body": ["본문"]})
        result = extract_meta_features(df)
        assert result["title_len"].iloc[0] == 5


class TestBuildFeaturePipeline:
    def test_pipeline_fit_transform(self, sample_training_df):
        df = preprocess_df(sample_training_df)
        df = extract_meta_features(df)

        pipeline = build_feature_pipeline()
        X = pipeline.fit_transform(df)

        assert X.shape[0] == len(df)
        assert X.shape[1] > 0

    def test_pipeline_transform_consistency(self, sample_training_df):
        """fit_transform 후 transform이 동일 입력에 동일 출력."""
        df = preprocess_df(sample_training_df)
        df = extract_meta_features(df)

        pipeline = build_feature_pipeline()
        X1 = pipeline.fit_transform(df)
        X2 = pipeline.transform(df)

        assert X1.shape == X2.shape
