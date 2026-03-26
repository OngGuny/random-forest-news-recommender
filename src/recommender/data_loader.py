import pandas as pd


SKIP_SHEETS = {"분류 축 정의"}

LABEL_MAP = {
    1: 0,  # 싫어요
    2: 1,  # 좋아요
    3: 1,  # 좋아요
}


def load_training_excel(path: str) -> pd.DataFrame:
    """엑셀 파일에서 학습 데이터를 로드한다.

    - 분류 축 정의 시트는 건너뜀
    - quality_score가 None인 행 제거 (미평가)
    - 라벨 변환: 1 → 0(싫어요), 2/3 → 1(좋아요)
    """
    xls = pd.ExcelFile(path)
    frames = []

    for sheet_name in xls.sheet_names:
        if sheet_name in SKIP_SHEETS:
            continue

        df = xls.parse(sheet_name)
        col_map = {
            "Title": "title",
            "URL": "url",
            "quality_score (1~3)": "quality_score",
        }
        df = df.rename(columns=col_map)
        df = df[["title", "url", "quality_score"]]
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # 미평가 제외
    combined = combined.dropna(subset=["quality_score"])
    combined["quality_score"] = combined["quality_score"].astype(int)

    # 라벨 변환
    combined["label"] = combined["quality_score"].map(LABEL_MAP)
    combined = combined.drop(columns=["quality_score"])

    return combined.reset_index(drop=True)


def load_unlabeled_excel(path: str) -> pd.DataFrame:
    """학습 데이터 엑셀에서 미평가(quality_score가 None) 기사만 추출한다."""
    xls = pd.ExcelFile(path)
    frames = []

    for sheet_name in xls.sheet_names:
        if sheet_name in SKIP_SHEETS:
            continue

        df = xls.parse(sheet_name)
        unlabeled = df[df["quality_score (1~3)"].isna()]
        col_map = {"Title": "title", "URL": "url"}
        unlabeled = unlabeled.rename(columns=col_map)
        frames.append(unlabeled[["title", "url"]])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["url"])
    return combined.reset_index(drop=True)


def load_prediction_excel(path: str) -> pd.DataFrame:
    """예측용 엑셀 파일에서 URL 목록을 로드한다."""
    xls = pd.ExcelFile(path)
    frames = []

    for sheet_name in xls.sheet_names:
        if sheet_name in SKIP_SHEETS:
            continue

        df = xls.parse(sheet_name)
        col_map = {"Title": "title", "URL": "url"}
        df = df.rename(columns=col_map)

        cols = [c for c in ["title", "url"] if c in df.columns]
        frames.append(df[cols])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["url"])
    return combined.reset_index(drop=True)
