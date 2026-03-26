from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "artifacts"
OUTPUT_DIR = BASE_DIR / "output"

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 5,
    "random_state": 42,
}

TFIDF_MAX_FEATURES = 5000

# Kiwi 형태소 분석 후 제거할 품사 태그 (조사, 어미, 구두점 등)
STOPWORD_POS = {
    "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ",
    "JX", "EP", "EF", "EC", "ETN", "ETM", "SF",
}
