import argparse
import logging

from src.recommender.crawler import crawl_articles
from src.recommender.data_loader import load_training_excel, load_unlabeled_excel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _save(df, path, fmt):
    if fmt == "json":
        df.to_json(path, orient="records", force_ascii=False, indent=2)
    else:
        df.to_csv(path, index=False)
    logger.info("저장 완료: %s (%d건)", path, len(df))


def main():
    parser = argparse.ArgumentParser(description="뉴스 기사 크롤링")
    parser.add_argument(
        "--data",
        required=True,
        help="학습 데이터 엑셀 파일 경로",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="저장 형식 (기본: json)",
    )
    args = parser.parse_args()
    ext = "json" if args.format == "json" else "csv"

    # 1. 라벨링된 데이터 (학습용)
    logger.info("=== 라벨링 데이터 크롤링 ===")
    labeled = load_training_excel(args.data)
    logger.info("라벨링 %d건 (추천: %d, 스킵: %d)",
                len(labeled), (labeled["label"] == 1).sum(), (labeled["label"] == 0).sum())

    labeled = crawl_articles(labeled)
    logger.info("크롤링 완료: %d건", len(labeled))
    _save(labeled, f"data/crawled_labeled.{ext}", args.format)

    # 2. 미평가 데이터 (예측용)
    logger.info("=== 미평가 데이터 크롤링 ===")
    unlabeled = load_unlabeled_excel(args.data)
    logger.info("미평가 %d건", len(unlabeled))

    unlabeled = crawl_articles(unlabeled)
    logger.info("크롤링 완료: %d건", len(unlabeled))
    _save(unlabeled, f"data/crawled_unlabeled.{ext}", args.format)

    print(f"\n=== 크롤링 완료 ===")
    print(f"라벨링: {len(labeled)}건 → data/crawled_labeled.{ext}")
    print(f"미평가: {len(unlabeled)}건 → data/crawled_unlabeled.{ext}")


if __name__ == "__main__":
    main()
