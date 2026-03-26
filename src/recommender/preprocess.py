import re

import pandas as pd
from kiwipiepy import Kiwi

from .config import STOPWORD_POS

_kiwi = Kiwi()


def clean_html(text: str) -> str:
    """HTML 태그, 특수문자, 불필요한 공백을 제거한다."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\w\s가-힣]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Kiwi 형태소 분석 후 불용어(조사/어미 등)를 제거한 토큰 리스트를 반환한다."""
    tokens = []
    for token in _kiwi.tokenize(text):
        if token.tag in STOPWORD_POS:
            continue
        if len(token.form) < 2:
            continue
        tokens.append(token.form)
    return tokens


def preprocess(text: str) -> str:
    """텍스트 전처리 파이프라인: HTML 제거 → 형태소 분석 → 공백 조인."""
    if not text or not isinstance(text, str):
        return ""
    cleaned = clean_html(text)
    tokens = tokenize(cleaned)
    return " ".join(tokens)


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame의 body 컬럼을 전처리하여 processed_body 컬럼을 추가한다."""
    df = df.copy()
    df["processed_body"] = df["body"].apply(preprocess)
    return df
