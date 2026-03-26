import argparse
import logging
import sys

import pandas as pd

from src.recommender.model import train
from src.recommender.preprocess import preprocess_df

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="뉴스 추천 모델 학습")
    parser.add_argument(
        "--data",
        default="data/crawled_labeled.json",
        help="크롤링 완료된 라벨링 데이터 파일 경로 (json 또는 csv)",
    )
    args = parser.parse_args()

    # 1. 크롤링 데이터 로드
    logger.info("데이터 로드: %s", args.data)
    if args.data.endswith(".json"):
        df = pd.read_json(args.data)
    else:
        df = pd.read_csv(args.data)

    like_count = (df["label"] == 1).sum()
    dislike_count = (df["label"] == 0).sum()
    logger.info("학습 데이터: %d건 (좋아요: %d, 싫어요: %d)", len(df), like_count, dislike_count)

    if len(df) < 10:
        logger.error("학습 가능한 데이터가 너무 적습니다 (%d건). 최소 10건 필요.", len(df))
        sys.exit(1)

    # 2. 전처리
    logger.info("텍스트 전처리 중...")
    df = preprocess_df(df)

    # 3. 학습
    logger.info("모델 학습 시작...")
    metrics = train(df)

    print("\n=== 학습 완료 ===")
    print(f"데이터: {len(df)}건")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1: {metrics['f1']}")


if __name__ == "__main__":
    main()
