import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.realestate.config import REGION_CODES

st.set_page_config(page_title="이상 거래 탐지", page_icon="🔍", layout="wide")
st.title("🔍 이상 거래 탐지")
st.caption("Isolation Forest | 구 단위 분리 탐지 | contamination 2%")

DB_PATH = Path("data/processed/realestate.db")

# lawd_cd → 구 이름 역매핑
LAWD_TO_NAME = {v: k.split("_", 1)[1] for k, v in REGION_CODES.items()}


# ── 데이터 로드 ────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """전체 + 이상치 데이터 로드"""
    if not DB_PATH.exists():
        return pd.DataFrame(), pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)

    # apt_anomaly 테이블 존재 확인
    tbl = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='apt_anomaly'"
    ).fetchone()
    if not tbl:
        conn.close()
        return pd.DataFrame(), pd.DataFrame()

    df_all = pd.read_sql_query(
        """
        SELECT
            t.id, t.apt_name, t.lawd_cd, t.dong_name,
            t.deal_date, t.deal_year, t.deal_month,
            t.deal_amount, t.area_exclusive, t.floor,
            t.build_year, t.dealing_type,
            a.anomaly_score, a.is_anomaly, a.price_per_sqm
        FROM apt_trade t
        JOIN apt_anomaly a ON t.id = a.id
        ORDER BY t.deal_date
        """,
        conn,
        parse_dates=["deal_date"],
    )
    conn.close()

    df_all["gu_name"]   = df_all["lawd_cd"].map(LAWD_TO_NAME).fillna(df_all["lawd_cd"])
    df_all["is_anomaly"] = df_all["is_anomaly"].astype(bool)

    df_anomaly = df_all[df_all["is_anomaly"]].copy()
    return df_all, df_anomaly


df_all, df_anomaly = load_data()

if df_all.empty:
    st.warning("이상탐지 결과가 없습니다. 먼저 아래 명령을 실행하세요.")
    st.code("python -m src.realestate.main anomaly --region 수도권 --contamination 0.02")
    st.stop()

df_normal = df_all[~df_all["is_anomaly"]]

# ── KPI ───────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("전체 거래",   f"{len(df_all):,}건")
k2.metric("이상치 건수", f"{len(df_anomaly):,}건",
          f"{len(df_anomaly)/len(df_all)*100:.1f}%")
k3.metric("이상치 중앙가",
          f"{df_anomaly['deal_amount'].median()/10000:.2f}억",
          f"정상 {df_normal['deal_amount'].median()/10000:.2f}억 대비",
          delta_color="off")

direct_normal  = (df_normal["dealing_type"] == "직거래").mean() * 100
direct_anomaly = (df_anomaly["dealing_type"] == "직거래").mean() * 100
k4.metric("이상치 직거래 비율",
          f"{direct_anomaly:.1f}%",
          f"정상 {direct_normal:.1f}% 대비 +{direct_anomaly - direct_normal:.1f}%p")

st.divider()

# ── 비즈니스 활용 ───────────────────────────────────────────────────
with st.expander("💡 이상 거래 탐지 — 실무 활용 방안", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**🏦 시세 산정 신뢰도 향상**
이상 거래를 제외한 정상 거래만으로 시세를 산정하면 시장 왜곡 없는
정확한 시세를 제공할 수 있습니다.
특히 직거래 급매나 특수관계인 거래가 많은 단지는
시세 신뢰도 점수(confidence score)를 낮게 부여하는 데 활용됩니다.

**🔍 담보대출 심사 플래그**
이상 거래 비율이 높은 단지는 담보가치 산정 시
주의 플래그를 표시하여 심사관의 추가 검토를 유도합니다.
이상치 중앙가가 9.60억인 점은 단순 고가가 아닌
지역 시세 이탈 거래임을 의미합니다.
        """)
    with c2:
        st.markdown("""
**👨‍👩‍👧 특수관계인 거래 모니터링**
직거래 비율이 정상 대비 2.2배 높은 점은
가족·법인 간 저가 거래 가능성을 시사합니다.
국세청·지자체 부동산 탈루 조사의 1차 필터로
활용될 수 있습니다.

**📈 급매 투자 기회 신호**
같은 단지·면적 내에서 시세보다 유의미하게 낮은
이상 거래는 급매 매물 탐지에 활용 가능합니다.
anomaly_score가 낮을수록(더 이상한 거래일수록)
급매 가능성이 높은 매물로 해석할 수 있습니다.
        """)

st.divider()

# ── 탭 ────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 거래유형 분석", "🗺 지역별 현황", "🏢 단지별 집중도"])


# ── Tab 1: 거래유형 분석 ──────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("거래유형 비율 비교")
        type_data = pd.DataFrame({
            "구분":     ["정상", "정상", "이상치", "이상치"],
            "거래유형": ["중개거래", "직거래", "중개거래", "직거래"],
            "비율":     [
                100 - direct_normal, direct_normal,
                100 - direct_anomaly, direct_anomaly,
            ],
        })
        fig = px.bar(
            type_data, x="구분", y="비율", color="거래유형",
            barmode="stack",
            color_discrete_map={"중개거래": "#4C9BE8", "직거래": "#E84C4C"},
            text_auto=".1f",
        )
        fig.update_layout(height=350, yaxis_title="비율 (%)", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("연도별 거래유형별 이상치 비율 추이")
        yearly = (
            df_all.groupby(["deal_year", "dealing_type"])
            .apply(lambda g: g["is_anomaly"].mean() * 100)
            .reset_index(name="이상치비율")
        )
        fig2 = px.line(
            yearly, x="deal_year", y="이상치비율", color="dealing_type",
            markers=True,
            color_discrete_map={"중개거래": "#4C9BE8", "직거래": "#E84C4C"},
            labels={"dealing_type": "거래유형"},
        )
        fig2.update_layout(height=350, xaxis_title="연도", yaxis_title="이상치 비율 (%)")
        st.plotly_chart(fig2, use_container_width=True)

    # 거래유형 선택
    deal_type = st.radio(
        "이상치 목록 보기", ["전체", "직거래", "중개거래"], horizontal=True
    )

    df_filtered = df_anomaly.copy()
    if deal_type != "전체":
        df_filtered = df_filtered[df_filtered["dealing_type"] == deal_type]

    df_filtered["가격(억)"] = (df_filtered["deal_amount"] / 10000).round(2)
    show_cols = ["deal_date", "apt_name", "gu_name", "dong_name",
                 "area_exclusive", "floor", "가격(억)", "price_per_sqm",
                 "dealing_type", "anomaly_score"]

    st.caption(f"{deal_type} 이상치: {len(df_filtered):,}건")
    st.dataframe(
        df_filtered[show_cols].sort_values("anomaly_score").head(100),
        use_container_width=True,
    )

    # 거래유형별 가격 분포 비교
    st.subheader("거래유형별 이상치 가격 분포")
    df_anomaly["가격(억)"] = df_anomaly["deal_amount"] / 10000
    fig3 = px.histogram(
        df_anomaly, x="가격(억)", color="dealing_type",
        nbins=50, barmode="overlay", opacity=0.7,
        color_discrete_map={"중개거래": "#4C9BE8", "직거래": "#E84C4C"},
        labels={"dealing_type": "거래유형"},
    )
    fig3.update_layout(height=350, yaxis_title="건수")
    st.plotly_chart(fig3, use_container_width=True)


# ── Tab 2: 지역별 현황 ─────────────────────────────────────────────
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("구별 이상치 건수 (상위 20)")
        cnt_by_gu = (
            df_anomaly.groupby("gu_name").size()
            .reset_index(name="이상치건수")
            .sort_values("이상치건수", ascending=False)
            .head(20)
        )
        fig3 = px.bar(
            cnt_by_gu, x="이상치건수", y="gu_name", orientation="h",
            color="이상치건수", color_continuous_scale="Reds",
        )
        fig3.update_layout(height=500, yaxis={"categoryorder": "total ascending"},
                           coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.subheader("구별 이상치 비율 (상위 20)")
        total_by_gu   = df_all.groupby("gu_name").size().reset_index(name="전체")
        anomaly_by_gu = df_anomaly.groupby("gu_name").size().reset_index(name="이상치")
        rate_by_gu = total_by_gu.merge(anomaly_by_gu, on="gu_name", how="left").fillna(0)
        rate_by_gu["이상치비율"] = (rate_by_gu["이상치"] / rate_by_gu["전체"] * 100).round(2)
        rate_by_gu = rate_by_gu.sort_values("이상치비율", ascending=False).head(20)

        fig4 = px.bar(
            rate_by_gu, x="이상치비율", y="gu_name", orientation="h",
            color="이상치비율", color_continuous_scale="Oranges",
            text_auto=".1f",
        )
        fig4.update_layout(height=500, yaxis={"categoryorder": "total ascending"},
                           coloraxis_showscale=False, xaxis_title="이상치 비율 (%)")
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("구별 정상 vs 이상치 가격 분포")
    top10_gu = cnt_by_gu.head(10)["gu_name"].tolist()
    df_top = df_all[df_all["gu_name"].isin(top10_gu)].copy()
    df_top["구분"] = df_top["is_anomaly"].map({True: "이상치", False: "정상"})
    df_top["가격(억)"] = df_top["deal_amount"] / 10000

    fig5 = px.box(
        df_top, x="gu_name", y="가격(억)", color="구분",
        color_discrete_map={"정상": "#4C9BE8", "이상치": "#E84C4C"},
        points=False,
    )
    fig5.update_layout(height=400, xaxis_title="", yaxis_title="거래가 (억원)")
    st.plotly_chart(fig5, use_container_width=True)


# ── Tab 3: 단지별 집중도 ───────────────────────────────────────────
with tab3:
    st.subheader("단지별 이상 거래 집중도 (이상치 건수 기준 상위 30)")

    complex_total   = df_all.groupby(["apt_name", "gu_name"]).size().reset_index(name="전체거래")
    complex_anomaly = df_anomaly.groupby(["apt_name", "gu_name"]).agg(
        이상치건수=("id", "count"),
        이상치평균가=("deal_amount", lambda x: round(x.mean() / 10000, 2)),
        평균anomaly_score=("anomaly_score", "mean"),
    ).reset_index()

    complex_df = complex_total.merge(complex_anomaly, on=["apt_name", "gu_name"], how="inner")
    complex_df["이상치비율(%)"] = (
        complex_df["이상치건수"] / complex_df["전체거래"] * 100
    ).round(1)
    complex_df["평균anomaly_score"] = complex_df["평균anomaly_score"].round(4)
    complex_df = complex_df.sort_values("이상치건수", ascending=False).head(30)

    st.dataframe(
        complex_df[[
            "apt_name", "gu_name", "전체거래", "이상치건수",
            "이상치비율(%)", "이상치평균가", "평균anomaly_score",
        ]].rename(columns={
            "apt_name": "단지명", "gu_name": "구",
            "이상치평균가": "이상치평균가(억)",
        }),
        use_container_width=True,
        height=600,
    )

    st.subheader("이상치 가격 vs anomaly_score 분포")
    df_plot = df_anomaly.copy()
    df_plot["가격(억)"] = df_plot["deal_amount"] / 10000
    fig6 = px.scatter(
        df_plot.sample(min(5000, len(df_plot)), random_state=42),
        x="가격(억)", y="anomaly_score",
        color="gu_name",
        hover_data=["apt_name", "deal_date", "area_exclusive", "dealing_type"],
        opacity=0.6,
        title="이상치 거래 분포 (score 낮을수록 더 이상)",
    )
    fig6.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig6, use_container_width=True)
