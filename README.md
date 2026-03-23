# random-forest-news-recommender

RandomForest 기반 뉴스 기사 추천 이진 분류기

## 개요

뉴스 기사를 입력받아 **읽을 만함(👍)** 또는 **스킵(👎)** 으로 분류하는 RandomForest 모델.
사용자의 좋아요/싫어요 피드백 데이터를 학습하여 새로운 기사의 추천 여부를 예측한다.

## 분류 파이프라인

```
원본 기사 → 전처리 → 피처 추출 → RandomForest → 👍 / 👎
```

### 1. 전처리

- HTML 태그, 특수문자 제거
- 형태소 분석 (Kiwi)
- 불용어 제거

### 2. 피처 추출

| 피처 | 설명 |
|---|---|
| TF-IDF 벡터 | 기사 본문의 단어 빈도-역문서 빈도 |
| 기사 길이 | 총 글자 수 / 문장 수 |
| 카테고리 | 정치, 경제, IT, 스포츠 등 (원핫 인코딩) |
| 제목 길이 | 제목 글자 수 |
| 출처 | 언론사 (라벨 인코딩) |

### 3. 모델

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    random_state=42
)

# 라벨: 1 = 읽을 만함, 0 = 스킵
model.fit(X_train, y_train)
```

### 4. 예측

```python
prediction = model.predict(features)  # [1] 또는 [0]
probability = model.predict_proba(features)  # [[스킵 확률, 추천 확률]]
```

- `predict_proba`의 추천 확률이 **0.5 이상**이면 👍, 미만이면 👎

## 학습 데이터

```
data/
├── liked.csv      # 좋아요 누른 기사
└── disliked.csv   # 싫어요 누른 기사
```

각 CSV 컬럼: `title`, `body`, `category`, `source`, `label`

## 실행

```bash
pip install -r requirements.txt

# 학습
python train.py

# 예측
python predict.py --url "https://example.com/news/12345"
```
