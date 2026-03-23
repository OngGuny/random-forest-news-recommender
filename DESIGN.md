# 설계 문서: RandomForest 뉴스 추천기

## 1. 데이터 흐름

```
학습데이터.xlsx (URL + quality_score)
  → [data_loader] 엑셀 파싱, 미평가 제외, 라벨 변환 (분류 축 무시, 점수만 사용)
  → [crawler] Google News RSS 리다이렉트 → 원본 URL → 기사 본문 크롤링
  → data/crawled_articles.csv (중간 산출물, 캐싱용)
  → [preprocess] HTML 제거, 형태소 분석(Kiwi), 불용어 제거
  → [features] TF-IDF + 메타 피처 추출
  → [model] RandomForest 학습 → models/model.joblib
```

```
예측:
  new_articles.xlsx (URL 목록)
  → [crawler] 기사 본문 크롤링
  → [preprocess + features] 동일 파이프라인 (transform)
  → [model] 예측
  → output/predictions.json 또는 output/predictions.csv
```

## 2. 라벨 매핑

분류 축(시트명)은 무시하고 quality_score만으로 판단한다.

| quality_score | label | 의미 |
|---|---|---|
| None | 제외 | 미평가 → 학습 대상 아님 (추후 예측 테스트용) |
| 1 | 0 | 싫어요 (스킵) |
| 2 | 1 | 좋아요 (추천) |
| 3 | 1 | 좋아요 (추천) |

## 3. 모듈 설계

### `src/recommender/config.py`

```python
DATA_DIR = "data"
MODEL_DIR = "models"
OUTPUT_DIR = "output"
CRAWLED_CACHE = "data/crawled_articles.csv"

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 5,
    "random_state": 42,
}

TFIDF_MAX_FEATURES = 5000

# Kiwi 형태소 분석 후 제거할 품사 태그
STOPWORD_POS = {"JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ",
                "JX", "EP", "EF", "EC", "ETN", "ETM", "SF"}
```

### `src/recommender/data_loader.py`

엑셀 파싱 및 라벨 변환 담당.

```
load_training_excel(path: str) -> DataFrame
```

- 3개 시트 합치기 (분류 축 정의 시트는 건너뜀)
- quality_score가 None인 행 제거
- 라벨 변환: 1 → 0, 2/3 → 1
- 반환 컬럼: title, url, label

```
load_prediction_excel(path: str) -> DataFrame
```

- URL 컬럼이 있는 xlsx 파싱
- 반환 컬럼: title, url

### `src/recommender/crawler.py`

Google News RSS URL → 원본 기사 본문 크롤링.

```
resolve_google_news_url(url: str) -> str        # 리다이렉트 따라가기
crawl_article(url: str) -> dict                  # 본문, 출처 추출
crawl_articles(df: DataFrame) -> DataFrame       # 일괄 크롤링 + 캐싱
```

- `requests` + `BeautifulSoup`로 본문 추출
- 이미 크롤링된 URL은 `data/crawled_articles.csv`에서 캐시 히트
- 크롤링 실패 시 해당 행 제외 + 경고 로그
- 요청 간 1초 sleep (예의)
- 반환 컬럼 추가: body, source(도메인)

### `src/recommender/preprocess.py`

텍스트 정제 및 형태소 분석.

```
clean_html(text: str) -> str          # HTML 태그, 특수문자 제거
tokenize(text: str) -> list[str]      # Kiwi 형태소 + 불용어 제거
preprocess(text: str) -> str          # clean → tokenize → 공백 조인
preprocess_df(df: DataFrame) -> DataFrame  # body 컬럼 일괄 전처리
```

### `src/recommender/features.py`

scikit-learn Pipeline 기반 피처 추출.

```
build_feature_pipeline() -> ColumnTransformer
extract_meta_features(df: DataFrame) -> DataFrame
```

**ColumnTransformer 구성:**

```
├─ tfidf: TfidfVectorizer(max_features=5000)  ← processed_body
├─ source: OrdinalEncoder(handle_unknown=use_encoded_value) ← source
└─ meta: passthrough  ← [body_len, sent_count, title_len]
```

- category 피처 제거 (분류 축은 학습 기준이 아님)

### `src/recommender/model.py`

모델 학습/예측/평가.

```
train(df: DataFrame) -> None
  1. extract_meta_features
  2. build_feature_pipeline → fit_transform
  3. train_test_split (8:2, stratified)
  4. RandomForestClassifier fit
  5. evaluate (accuracy, precision, recall, f1)
  6. save: model + pipeline → models/model.joblib

predict(df: DataFrame, output_format: str = "json") -> None
  1. load model + pipeline
  2. extract_meta_features
  3. pipeline.transform
  4. model.predict + predict_proba
  5. 결과 저장 → output/predictions.{json|csv}

evaluate(model, X_test, y_test) -> dict
  accuracy, precision, recall, f1 출력
```

## 4. CLI 진입점

### `train.py`

```bash
uv run python train.py --data "뉴스기사 라벨링데이터/학습데이터.xlsx"
```

흐름: load_training_excel → crawl_articles → preprocess_df → train

### `predict.py`

```bash
uv run python predict.py --input data/new_articles.xlsx --format json
```

흐름: load_prediction_excel → crawl_articles → preprocess_df → predict

## 5. 출력 형식

### predictions.json

```json
[
  {
    "title": "기사 제목",
    "url": "https://...",
    "source": "매일경제",
    "prediction": 1,
    "probability": 0.83,
    "label": "👍 추천"
  }
]
```

### predictions.csv

```csv
title,url,source,prediction,probability,label
기사 제목,https://...,매일경제,1,0.83,👍 추천
```

## 6. 프로젝트 구조

```
random-forest-news-recommender/
├── pyproject.toml
├── train.py
├── predict.py
├── src/recommender/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── crawler.py
│   ├── preprocess.py
│   ├── features.py
│   └── model.py
├── data/
│   ├── crawled_articles.csv   # 크롤링 캐시 (자동 생성)
│   └── new_articles.xlsx      # 예측 입력
├── models/
│   └── model.joblib           # 학습 후 생성
├── output/
│   └── predictions.json       # 예측 후 생성
└── tests/
    ├── test_data_loader.py
    ├── test_preprocess.py
    ├── test_features.py
    └── test_model.py
```

## 7. 의존성

```
scikit-learn, pandas, openpyxl, kiwipiepy, joblib,
requests, beautifulsoup4, lxml
```

## 8. 테스트 전략

### 8.1 단위 테스트 (pytest)

#### `test_data_loader.py`
- **엑셀 파싱 검증**: 테스트용 미니 xlsx(5행) 생성 → 로드 → 컬럼/타입 확인
- **라벨 변환 검증**: quality_score 1→0, 2→1, 3→1 매핑 정확성
- **미평가 제외 검증**: None 행이 결과에 포함되지 않는지
- **분류 축 정의 시트 무시 검증**: 첫 번째 시트가 결과에 포함되지 않는지

#### `test_preprocess.py`
- **HTML 제거**: `<p>본문</p>` → `본문`
- **특수문자 제거**: 이메일, URL, 특수기호 처리
- **형태소 분석**: 알려진 문장 → 기대 토큰 리스트 비교
- **불용어 제거**: 조사/어미 등 STOPWORD_POS 태그 토큰이 결과에 없는지
- **빈 문자열/None 입력**: 에러 없이 빈 결과 반환

#### `test_features.py`
- **메타 피처 추출**: body_len, sent_count, title_len 계산 정확성
- **파이프라인 구성**: build_feature_pipeline() 반환 타입 및 컬럼 매핑 확인
- **fit_transform → transform 일관성**: 동일 입력에 동일 출력

#### `test_model.py`
- **학습 → 저장 → 로드 round-trip**: joblib 저장 후 로드한 모델이 동일 예측 반환
- **예측 출력 형식**: predict 결과에 prediction, probability 키 존재
- **evaluate 지표 범위**: accuracy/precision/recall/f1이 0~1 사이

### 8.2 통합 테스트

#### `test_pipeline.py`
- **전체 파이프라인 스모크 테스트**: 미니 데이터셋(10건)으로 load → preprocess → features → train → predict 전 과정이 에러 없이 완료되는지
- **크롤링 제외 테스트**: body가 이미 있는 DataFrame을 직접 주입하여 크롤링 없이 파이프라인 동작 확인

### 8.3 테스트 원칙

- 크롤링(`crawler.py`)은 외부 의존성이므로 단위 테스트에서 mock 처리
- 통합 테스트는 크롤링 완료된 캐시 데이터를 사용
- 테스트용 fixture는 `tests/fixtures/`에 미니 xlsx, csv 배치
- CI 없이 로컬 실행: `uv run pytest tests/`

## 9. 주요 결정 사항

- **DB 없음**: 모든 I/O는 CSV/JSON 파일 기반
- **크롤링 캐시**: 한번 크롤링한 기사는 CSV에 저장, 재크롤링 방지
- **미평가 데이터 제외**: quality_score가 None인 행은 학습에서 제외, 추후 예측 테스트용으로 활용
- **파이프라인 일체 저장**: TF-IDF 등 전처리 파이프라인을 모델과 함께 joblib 저장 → 예측 시 동일 변환 보장
- **분류 축 무시**: 시트명(카테고리)은 피처로 사용하지 않음. 순수 quality_score만으로 좋아요/싫어요 판단
- **입력 포맷 통일**: 학습/예측 모두 xlsx 입력
