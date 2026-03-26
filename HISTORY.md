# 작업 히스토리

## 2026-03-26 — 크롤링 오류 수정 및 파이프라인 개선

### 1. Google News RSS URL 디코딩 오류 수정

**문제**: `crawler.py`의 `resolve_google_news_url()`이 `requests` 리다이렉트 방식으로 Google News RSS URL을 원본 기사 URL로 변환하고 있었으나, Google이 2024년부터 HTTP 리다이렉트를 JS 기반으로 변경하여 전혀 동작하지 않음. 151건 전부 "본문이 너무 짧음"으로 스킵되어 크롤링 성공률 0%.

**조치**: `googlenewsdecoder` 라이브러리를 도입하여 protobuf 인코딩된 URL을 직접 디코딩하도록 변경.

**결과**: 151건 중 147건 크롤링 성공.

**변경 파일**: `src/recommender/crawler.py`, `pyproject.toml`

---

### 2. 크롤링 URL을 원본 기사 URL로 교체

**문제**: 크롤링 결과의 `url` 컬럼이 Google News RSS URL 그대로 저장되어, 향후 중복 처리 등에 활용 불가.

**조치**: `crawl_articles()`에서 merge 시 `real_url`을 `url`로 교체하도록 수정.

**변경 파일**: `src/recommender/crawler.py`

---

### 3. 본문 추출 품질 개선 (trafilatura 도입)

**문제**: `_extract_body()`가 `<article>` 태그 없는 사이트에서 `<body>` 전체를 가져와, 네비게이션 메뉴/관련 기사 목록/푸터/댓글 등 UI 텍스트가 본문에 대량 포함됨. hellot.net, weekly.donga.com, youthdaily.co.kr 등 다수 사이트에서 발생. 이로 인해 TF-IDF 피처와 메타 피처(body_len, sent_count)가 오염되어 모델 성능에 악영향.

**조치**: `BeautifulSoup` 기반 수동 추출을 `trafilatura` 라이브러리로 교체. 네비게이션, 메뉴, 푸터, 관련 기사 목록 등을 자동 제거하고 순수 본문만 추출.

**결과**:
| 지표 | BeautifulSoup | trafilatura |
|---|---|---|
| Accuracy | 84.38% | **87.50%** |
| Precision | 85.19% | **88.46%** |
| Recall | 95.83% | 95.83% |
| F1 | 90.20% | **92.00%** |

**변경 파일**: `src/recommender/crawler.py`, `pyproject.toml`

---

### 4. 크롤링 파이프라인 분리 (라벨링/미평가 데이터)

**문제**: `crawl.py`가 라벨링된 데이터(학습용)만 크롤링. 미평가 데이터(예측용)는 `predict.py` 실행 시마다 매번 크롤링해야 하는 비효율.

**조치**:
- `crawl.py`: 라벨링 + 미평가 데이터를 한 번에 크롤링하여 각각 `data/crawled_labeled.json`, `data/crawled_unlabeled.json`으로 저장.
- `data_loader.py`: `load_unlabeled_excel()` 함수 추가 (quality_score가 None인 미평가 기사 추출).
- `train.py`: 기본 입력 경로를 `data/crawled_labeled.json`으로 변경.
- `predict.py`: 크롤링 완료된 `data/crawled_unlabeled.json`만 입력받도록 단순화.

**변경 파일**: `crawl.py`, `predict.py`, `train.py`, `src/recommender/data_loader.py`

---

### 5. 예측 결과 타임스탬프 버전 관리

**문제**: 예측 결과가 `output/predictions.json` 고정 경로에 덮어쓰기되어 이전 버전 결과 추적 불가.

**조치**: 출력 파일명에 타임스탬프 추가 — `output/predictions_YYYYMMDD_HHMM.json`.

**변경 파일**: `src/recommender/model.py`

---

### 6. 모델 저장 디렉토리 이름 변경

**문제**: `models/` 디렉토리명이 pydantic/SQLAlchemy 등의 데이터 모델 디렉토리와 혼동 가능.

**조치**: `models/` → `artifacts/`로 변경.

**변경 파일**: `src/recommender/config.py`, `.gitignore`

---

### 현재 파이프라인

```bash
# 1. 크롤링 (라벨링 + 미평가 동시)
uv run python crawl.py --data data/data_for_training.xlsx

# 2. 학습
uv run python train.py

# 3. 예측
uv run python predict.py
```
