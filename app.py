import streamlit as st

st.set_page_config(
    page_title="아파트 실거래가 분석",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏠 한국 아파트 실거래가 분석 · 예측")
st.caption("국토교통부 MOLIT API + Kakao 지오코딩 + ML | 수도권 2020–2026")

col1, col2 = st.columns(2)
with col1:
    st.info(
        "### 🔮 가격 예측\n"
        "지역 · 전용면적 · 층 · 건축년도를 입력하면\n"
        "LightGBM / XGBoost / RandomForest 중 선택한 모델이 예상 매매가를 계산합니다.\n\n"
        "👈 왼쪽 사이드바에서 **가격예측** 페이지를 선택하세요"
    )
with col2:
    st.info(
        "### 📊 데이터 탐색\n"
        "월별 가격 추이 · 지역별 비교 ·\n"
        "전용면적 vs 가격 분포 · 시군구 가격 지도\n\n"
        "👈 왼쪽 사이드바에서 **데이터탐색** 페이지를 선택하세요"
    )

st.divider()

# 데이터 현황
c1, c2, c3 = st.columns(3)
c1.metric("수집 기간", "2020.01 – 2026.04")
c2.metric("수집 지역", "수도권 56개 구")
c3.metric("총 거래 건수", "1,129,994 건")

st.divider()

# 모델 성능 비교
st.subheader("모델 성능 비교")

import pandas as _pd
cmp_data = _pd.DataFrame([
    {"모델": "RandomForest", "시계열 R²": 0.8870, "단지 R²": 0.8592, "시계열 MAE": "1.23억", "단지 MAE": "1.17억"},
    {"모델": "LightGBM",     "시계열 R²": 0.8802, "단지 R²": 0.8833, "시계열 MAE": "1.28억", "단지 MAE": "1.07억"},
    {"모델": "XGBoost",      "시계열 R²": 0.8573, "단지 R²": 0.8781, "시계열 MAE": "1.42억", "단지 MAE": "1.12억"},
])
st.dataframe(cmp_data, hide_index=True, use_container_width=True)
st.caption("시계열 분리: cutoff 2025-01-01 | 단지 분리: unseen 단지 20% holdout")
