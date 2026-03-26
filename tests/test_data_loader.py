import tempfile

import openpyxl
import pytest

from src.recommender.data_loader import load_training_excel, load_prediction_excel


def _create_test_excel(path, sheets_data):
    """테스트용 엑셀 파일을 생성한다."""
    wb = openpyxl.Workbook()
    # 기본 시트 제거
    wb.remove(wb.active)

    for sheet_name, rows in sheets_data.items():
        ws = wb.create_sheet(title=sheet_name)
        for row in rows:
            ws.append(row)

    wb.save(path)


class TestLoadTrainingExcel:
    def test_label_mapping(self, tmp_path):
        """quality_score 1→0, 2→1, 3→1 매핑 확인."""
        xlsx_path = tmp_path / "test.xlsx"
        _create_test_excel(xlsx_path, {
            "테스트": [
                ["Title", "URL", "axis_name", "is_recommended (Y/N)", "quality_score (1~3)", "comment"],
                ["기사1", "https://a.com/1", "-", "N", 1, ""],
                ["기사2", "https://a.com/2", "-", "Y", 2, ""],
                ["기사3", "https://a.com/3", "-", "Y", 3, ""],
            ]
        })

        df = load_training_excel(str(xlsx_path))
        assert list(df["label"]) == [0, 1, 1]

    def test_none_excluded(self, tmp_path):
        """quality_score가 None인 행은 제외."""
        xlsx_path = tmp_path / "test.xlsx"
        _create_test_excel(xlsx_path, {
            "테스트": [
                ["Title", "URL", "axis_name", "is_recommended (Y/N)", "quality_score (1~3)", "comment"],
                ["기사1", "https://a.com/1", "-", "N", 1, ""],
                ["기사2", "https://a.com/2", "-", None, None, ""],
                ["기사3", "https://a.com/3", "-", "Y", 3, ""],
            ]
        })

        df = load_training_excel(str(xlsx_path))
        assert len(df) == 2
        assert "https://a.com/2" not in df["url"].values

    def test_skip_definition_sheet(self, tmp_path):
        """분류 축 정의 시트는 건너뜀."""
        xlsx_path = tmp_path / "test.xlsx"
        _create_test_excel(xlsx_path, {
            "분류 축 정의": [
                ["설명", "이 시트는 무시"],
            ],
            "테스트": [
                ["Title", "URL", "axis_name", "is_recommended (Y/N)", "quality_score (1~3)", "comment"],
                ["기사1", "https://a.com/1", "-", "Y", 3, ""],
            ]
        })

        df = load_training_excel(str(xlsx_path))
        assert len(df) == 1

    def test_multiple_sheets_combined(self, tmp_path):
        """여러 시트 데이터가 합쳐짐."""
        xlsx_path = tmp_path / "test.xlsx"
        _create_test_excel(xlsx_path, {
            "시트A": [
                ["Title", "URL", "axis_name", "is_recommended (Y/N)", "quality_score (1~3)", "comment"],
                ["기사1", "https://a.com/1", "-", "Y", 3, ""],
            ],
            "시트B": [
                ["Title", "URL", "axis_name", "is_recommended (Y/N)", "quality_score (1~3)", "comment"],
                ["기사2", "https://a.com/2", "-", "N", 1, ""],
            ],
        })

        df = load_training_excel(str(xlsx_path))
        assert len(df) == 2

    def test_columns(self, tmp_path):
        """반환 DataFrame에 title, url, label 컬럼만 존재."""
        xlsx_path = tmp_path / "test.xlsx"
        _create_test_excel(xlsx_path, {
            "테스트": [
                ["Title", "URL", "axis_name", "is_recommended (Y/N)", "quality_score (1~3)", "comment"],
                ["기사1", "https://a.com/1", "-", "Y", 3, "좋음"],
            ]
        })

        df = load_training_excel(str(xlsx_path))
        assert set(df.columns) == {"title", "url", "label"}


class TestLoadPredictionExcel:
    def test_basic_load(self, tmp_path):
        """예측용 로더 기본 동작."""
        xlsx_path = tmp_path / "test.xlsx"
        _create_test_excel(xlsx_path, {
            "테스트": [
                ["Title", "URL"],
                ["기사1", "https://a.com/1"],
                ["기사2", "https://a.com/2"],
            ]
        })

        df = load_prediction_excel(str(xlsx_path))
        assert len(df) == 2
        assert set(df.columns) == {"title", "url"}

    def test_null_url_excluded(self, tmp_path):
        """URL이 None인 행은 제외."""
        xlsx_path = tmp_path / "test.xlsx"
        _create_test_excel(xlsx_path, {
            "테스트": [
                ["Title", "URL"],
                ["기사1", "https://a.com/1"],
                ["기사2", None],
            ]
        })

        df = load_prediction_excel(str(xlsx_path))
        assert len(df) == 1
