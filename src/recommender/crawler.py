import logging
import re
import time
from urllib.parse import urlparse

import pandas as pd

_MOJIBAKE_RE = re.compile(r"[\x80-\x9f]|ã[\S]{0,2}[ã±-ãŽ]|ï¼|è²·|æ¥ç|é·|ç¥")
MOJIBAKE_THRESHOLD = 0.02
import requests
import trafilatura
from googlenewsdecoder import new_decoderv1

logger = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )
})


def resolve_google_news_url(url: str) -> str:
    """Google News RSS 리다이렉트 URL을 원본 URL로 변환한다."""
    try:
        result = new_decoderv1(url)
        if result.get("status") and result.get("decoded_url"):
            return result["decoded_url"]
        logger.warning("Google News URL 디코딩 실패, 원본 사용: %s", url)
        return url
    except Exception as e:
        logger.warning("리다이렉트 해소 실패: %s → %s", url, e)
        return url


def _extract_body(html: str) -> str:
    """HTML에서 기사 본문을 추출한다."""
    body = trafilatura.extract(html)
    return body or ""


def crawl_article(url: str) -> dict | None:
    """단일 기사 URL에서 본문과 출처를 크롤링한다."""
    try:
        real_url = resolve_google_news_url(url)
        resp = _SESSION.get(real_url, timeout=15)
        resp.raise_for_status()

        body = _extract_body(resp.text)
        source = urlparse(real_url).netloc.replace("www.", "")

        if len(body.strip()) < 50:
            logger.warning("본문이 너무 짧음 (스킵): %s", real_url)
            return None

        mojibake_ratio = len(_MOJIBAKE_RE.findall(body)) / len(body)
        if mojibake_ratio > MOJIBAKE_THRESHOLD:
            logger.warning("인코딩 깨짐 (스킵, %.1f%%): %s", mojibake_ratio * 100, real_url)
            return None

        return {
            "original_url": url,
            "real_url": real_url,
            "body": body,
            "source": source,
        }
    except requests.RequestException as e:
        logger.warning("크롤링 실패: %s → %s", url, e)
        return None


def crawl_articles(df: pd.DataFrame, sleep_sec: float = 1.0) -> pd.DataFrame:
    """DataFrame의 url 컬럼을 기반으로 기사를 일괄 크롤링한다.

    크롤링 실패 행은 제외된다.
    """
    rows = []
    for idx, row in df.iterrows():
        url = row["url"]
        logger.info("[%d/%d] 크롤링: %s", idx + 1, len(df), url)
        result = crawl_article(url)
        if result:
            rows.append(result)

        time.sleep(sleep_sec)

    crawled = pd.DataFrame(rows)

    # df에 크롤링 결과 병합
    merged = df.merge(
        crawled[["original_url", "real_url", "body", "source"]],
        left_on="url",
        right_on="original_url",
        how="left",
    )
    merged["url"] = merged["real_url"].fillna(merged["url"])
    merged = merged.drop(columns=["original_url", "real_url"])

    before = len(merged)
    merged = merged.dropna(subset=["body"])
    after = len(merged)
    if before != after:
        logger.warning("본문 없는 %d건 제외됨", before - after)

    return merged.reset_index(drop=True)
