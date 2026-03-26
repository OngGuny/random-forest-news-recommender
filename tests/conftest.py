import pandas as pd
import pytest


@pytest.fixture
def sample_training_df():
    """크롤링 완료된 학습용 샘플 DataFrame."""
    return pd.DataFrame({
        "title": [
            "AI 반도체 전쟁 격화",
            "철강업계 피지컬 AI 가속",
            "포스코 1조 투자 미국 진출",
            "제철소 AI 도입 사례",
            "철강협회 AI 인재 양성",
            "현대제철 스마트팩토리 구축",
            "단순 보도자료 기사",
            "날씨 관련 기사",
            "연예 뉴스 기사",
            "광고성 기사 내용",
        ],
        "url": [f"https://example.com/news/{i}" for i in range(10)],
        "body": [
            "삼성전자가 HBM4 양산을 본격화하면서 AI 반도체 시장에서의 경쟁이 치열해지고 있다. 엔비디아와의 협력도 강화되는 추세다.",
            "제철소 위험 공정에 휴머노이드 로봇을 투입하는 방안이 추진되고 있다. 철강업계가 피지컬 AI 도입에 속도를 내고 있다.",
            "포스코가 미국 철강사 지분을 인수하기 위해 1조원대 투자를 결정했다. 글로벌 철강 시장 재편에 나선다.",
            "AI를 활용한 품질 검사 시스템이 제철소에 도입되어 불량률을 크게 줄이고 있다. 데이터 기반 의사결정이 확산되고 있다.",
            "철강협회와 AI 소프트웨어 협회가 손잡고 철강 특화 AI 인재를 양성하기로 했다. 자율제조 생태계 조성이 목표다.",
            "현대제철이 스마트팩토리를 구축하여 생산성을 20% 향상시켰다. IoT 센서와 AI 분석 시스템이 핵심이다.",
            "이것은 단순한 보도자료를 그대로 옮긴 기사입니다.",
            "오늘 날씨가 맑겠습니다. 기온은 영상 10도입니다.",
            "유명 연예인이 새 드라마에 출연합니다.",
            "이 제품을 사용하면 놀라운 효과를 볼 수 있습니다. 지금 바로 구매하세요.",
        ],
        "source": [
            "news.example.com", "daily.example.com", "economy.example.com",
            "tech.example.com", "industry.example.com", "steel.example.com",
            "pr.example.com", "weather.example.com", "ent.example.com",
            "ad.example.com",
        ],
        "label": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    })


@pytest.fixture
def sample_prediction_df():
    """크롤링 완료된 예측용 샘플 DataFrame."""
    return pd.DataFrame({
        "title": ["새로운 AI 기술 기사", "일반 뉴스 기사"],
        "url": ["https://example.com/new/1", "https://example.com/new/2"],
        "body": [
            "철강 산업에서 AI 기술이 빠르게 도입되고 있다. 생산 효율성과 안전성이 크게 향상되었다.",
            "오늘의 날씨와 주요 사건을 정리한 기사입니다.",
        ],
        "source": ["tech.example.com", "news.example.com"],
    })
