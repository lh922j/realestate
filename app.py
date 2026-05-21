import streamlit as st

st.set_page_config(
    page_title="아파트 실거래가 분석",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏠 한국 아파트 실거래가 분석 · 예측")
st.caption("국토교통부 MOLIT API + Kakao 지오코딩 + LightGBM | 수도권 2020–2025")

col1, col2 = st.columns(2)
with col1:
    st.info(
        "### 🔮 가격 예측\n"
        "지역 · 전용면적 · 층 · 건축년도를 입력하면\n"
        "LightGBM 모델이 예상 매매가를 계산합니다.\n\n"
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

c1, c2, c3, c4 = st.columns(4)
c1.metric("수집 기간", "2020.01 – 2025.02")
c2.metric("수집 지역", "수도권 56개 구")
c3.metric("총 거래 건수", "약 24만 건")
c4.metric("예측 모델", "LightGBM")

st.caption(
    "모델 성능: MAE 1.47억 · R² 0.90 (시계열 분리) / MAE 1.54억 · R² 0.85 (단지 분리)"
)
