import argparse
import logging

import pandas as pd

from src.recommender.model import predict
from src.recommender.preprocess import preprocess_df

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="뉴스 기사 추천 예측")
    parser.add_argument(
        "--input",
        default="data/crawled_unlabeled.json",
        help="크롤링 완료된 미평가 데이터 파일 경로 (json 또는 csv)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="출력 형식 (기본: json)",
    )
    args = parser.parse_args()

    # 1. 데이터 로드
    logger.info("데이터 로드: %s", args.input)
    if args.input.endswith(".json"):
        df = pd.read_json(args.input)
    else:
        df = pd.read_csv(args.input)
    logger.info("예측 대상: %d건", len(df))

    # 2. 전처리
    logger.info("텍스트 전처리 중...")
    df = preprocess_df(df)

    # 3. 예측
    logger.info("예측 수행 중...")
    results = predict(df, output_format=args.format)

    like_count = sum(1 for r in results if r["prediction"] == 1)
    print(f"\n=== 예측 완료 ({len(results)}건) ===")
    print(f"추천: {like_count}건 | 스킵: {len(results) - like_count}건")


if __name__ == "__main__":
    main()
