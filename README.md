# random-forest-news-recommender

RandomForest 기반 뉴스 기사 추천 이진 분류기

## 개요

뉴스 기사를 입력받아 **추천(1)** 또는 **스킵(0)** 으로 분류하는 RandomForest 모델.
사용자가 평가한 quality_score(1~3점)를 기반으로 학습하여 새로운 기사의 추천 여부를 예측한다.

## 모델 성능

| 지표 | 값 |
|---|---|
| **Accuracy** | 87.50% |
| **Precision** | 88.46% |
| **Recall** | 95.83% |
| **F1 Score** | 92.00% |

- 학습 데이터: 157건 (추천 120 / 스킵 37)
- Train/Test 분할: 8:2 (stratified)

## 파이프라인

```
학습 데이터 (xlsx)
  → [crawl.py]    엑셀 파싱 + Google News URL 디코딩 + 기사 본문 크롤링
  → [train.py]    전처리 + 피처 추출 + RandomForest 학습
  → [predict.py]  미평가 기사에 대한 추천 예측
```

### 라벨 매핑

| quality_score | label | 의미 |
|---|---|---|
| None | 제외 | 미평가 (예측 대상) |
| 1 | 0 | 스킵 |
| 2, 3 | 1 | 추천 |

### 피처 구성

| 피처 | 설명 |
|---|---|
| TF-IDF (max 5,000) | 본문 단어 빈도-역문서 빈도 |
| body_len | 본문 글자 수 |
| sent_count | 본문 문장 수 |
| title_len | 제목 글자 수 |
| source | 언론사 (OrdinalEncoder) |

### 전처리

- 본문 추출: `trafilatura` (네비게이션/메뉴/푸터 자동 제거)
- HTML 태그, 특수문자 제거
- 형태소 분석: `Kiwi`
- 불용어 제거 (조사, 어미, 구두점)

## 실행

```bash
# 1. 크롤링 — 라벨링 + 미평가 데이터 동시 크롤링
uv run python crawl.py --data data/data_for_training.xlsx

# 2. 학습
uv run python train.py

# 3. 예측
uv run python predict.py
```

### 입출력

| 단계 | 입력 | 출력 |
|---|---|---|
| 크롤링 | `data/data_for_training.xlsx` | `data/crawled_labeled.json`, `data/crawled_unlabeled.json` |
| 학습 | `data/crawled_labeled.json` | `artifacts/model.joblib` |
| 예측 | `data/crawled_unlabeled.json` | `output/predictions_YYYYMMDD_HHMM.json` |

### 예측 결과 형식

```json
{
  "title": "기사 제목",
  "url": "https://example.com/article/123",
  "source": "example.com",
  "prediction": 1,
  "probability": 0.83,
  "label": "👍 추천"
}
```

## 프로젝트 구조

```
random-forest-news-recommender/
├── crawl.py                    # 크롤링 CLI
├── train.py                    # 학습 CLI
├── predict.py                  # 예측 CLI
├── src/recommender/
│   ├── config.py               # 설정 (경로, RF 파라미터, 불용어)
│   ├── data_loader.py          # 엑셀 파싱, 라벨 변환
│   ├── crawler.py              # Google News URL 디코딩 + 본문 크롤링
│   ├── preprocess.py           # 텍스트 정제, 형태소 분석
│   ├── features.py             # TF-IDF + 메타 피처 파이프라인
│   └── model.py                # 학습, 평가, 예측
├── data/
│   ├── data_for_training.xlsx  # 원본 학습 데이터
│   ├── crawled_labeled.json    # 크롤링된 라벨링 데이터
│   └── crawled_unlabeled.json  # 크롤링된 미평가 데이터
├── artifacts/
│   └── model.joblib            # 학습된 모델 + 파이프라인
├── output/
│   └── predictions_*.json      # 예측 결과 (타임스탬프별)
└── tests/
```
