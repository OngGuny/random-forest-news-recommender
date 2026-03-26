import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder

from .config import TFIDF_MAX_FEATURES


def extract_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """본문/제목으로부터 메타 피처를 추출한다."""
    df = df.copy()
    df["body_len"] = df["body"].str.len().fillna(0).astype(int)
    df["sent_count"] = df["body"].str.count(r"[.!?。\n]+").fillna(0).astype(int)
    df["title_len"] = df["title"].str.len().fillna(0).astype(int)
    return df


def build_feature_pipeline() -> ColumnTransformer:
    """피처 추출 파이프라인을 구성한다."""
    return ColumnTransformer(
        transformers=[
            (
                "tfidf",
                TfidfVectorizer(max_features=TFIDF_MAX_FEATURES),
                "processed_body",
            ),
            (
                "source",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ["source"],
            ),
            (
                "meta",
                "passthrough",
                ["body_len", "sent_count", "title_len"],
            ),
        ],
        remainder="drop",
    )
