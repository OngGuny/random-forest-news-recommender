import pandas as pd

from src.recommender.preprocess import clean_html, tokenize, preprocess, preprocess_df


class TestCleanHtml:
    def test_removes_html_tags(self):
        assert "본문" in clean_html("<p>본문</p>")
        assert "<" not in clean_html("<div class='a'>텍스트</div>")

    def test_removes_special_chars(self):
        result = clean_html("가격은 $100 입니다!!")
        assert "$" not in result

    def test_collapses_whitespace(self):
        result = clean_html("여러   공백이    있다")
        assert "  " not in result

    def test_empty_string(self):
        assert clean_html("") == ""


class TestTokenize:
    def test_basic_tokenization(self):
        tokens = tokenize("삼성전자가 반도체를 생산한다")
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        # 조사 "가", "를"은 제거되어야 함
        assert "가" not in tokens
        assert "를" not in tokens

    def test_short_tokens_removed(self):
        """1글자 토큰은 제거."""
        tokens = tokenize("나는 AI를 좋아한다")
        for token in tokens:
            assert len(token) >= 2


class TestPreprocess:
    def test_full_pipeline(self):
        result = preprocess("<p>삼성전자가 HBM4 양산을 시작했다</p>")
        assert isinstance(result, str)
        assert "<p>" not in result
        assert len(result) > 0

    def test_none_input(self):
        assert preprocess(None) == ""

    def test_empty_input(self):
        assert preprocess("") == ""


class TestPreprocessDf:
    def test_adds_processed_body_column(self):
        df = pd.DataFrame({"body": ["철강 산업이 발전하고 있다"]})
        result = preprocess_df(df)
        assert "processed_body" in result.columns
        assert len(result["processed_body"].iloc[0]) > 0

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"body": ["테스트 문장"]})
        _ = preprocess_df(df)
        assert "processed_body" not in df.columns
