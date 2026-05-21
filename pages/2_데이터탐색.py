import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="데이터 탐색", page_icon="📊", layout="wide")
st.title("📊 실거래가 데이터 탐색")

DB_PATH  = Path("data/processed/realestate.db")
GEO_PATH = Path("data/geo/skorea_municipalities.geojson")

# ── 지역 매핑 ──────────────────────────────────────────────────────
LAWD_TO_GU: dict[str, str] = {
    "11110": "종로구",    "11140": "중구",      "11170": "용산구",
    "11200": "성동구",    "11215": "광진구",    "11230": "동대문구",
    "11260": "중랑구",    "11290": "성북구",    "11305": "강북구",
    "11320": "도봉구",    "11350": "노원구",    "11380": "은평구",
    "11410": "서대문구",  "11440": "마포구",    "11470": "양천구",
    "11500": "강서구",    "11530": "구로구",    "11545": "금천구",
    "11560": "영등포구",  "11590": "동작구",    "11620": "관악구",
    "11650": "서초구",    "11680": "강남구",    "11710": "송파구",
    "11740": "강동구",
    "41111": "장안구(수원)",  "41113": "권선구(수원)",  "41115": "팔달구(수원)",
    "41117": "영통구(수원)",  "41131": "수정구(성남)",  "41133": "중원구(성남)",
    "41135": "분당구(성남)",  "41281": "덕양구(고양)",  "41285": "일산동구(고양)",
    "41287": "일산서구(고양)","41461": "처인구(용인)",  "41463": "기흥구(용인)",
    "41465": "수지구(용인)",  "41171": "만안구(안양)",  "41173": "동안구(안양)",
    "41192": "원미구(부천)",  "41194": "소사구(부천)",  "41196": "오정구(부천)",
    "41590": "화성시",        "41570": "김포시",        "41360": "남양주시",
    "41450": "하남시",        "41290": "과천시",
    "28110": "중구(인천)",    "28140": "동구(인천)",    "28177": "미추홀구",
    "28185": "연수구",        "28200": "남동구",        "28237": "부평구",
    "28245": "계양구",        "28260": "서구(인천)",
}

# lawd_cd → GeoJSON code 매핑 (choropleth key_on 용)
# GeoJSON은 서울 11xxx, 인천 23xxx, 경기 31xxx 코드 체계 사용
LAWD_TO_GEO_CODE: dict[str, str] = {
    # 서울
    "11110": "11010", "11140": "11020", "11170": "11030", "11200": "11040",
    "11215": "11050", "11230": "11060", "11260": "11070", "11290": "11080",
    "11305": "11090", "11320": "11100", "11350": "11110", "11380": "11120",
    "11410": "11130", "11440": "11140", "11470": "11150", "11500": "11160",
    "11530": "11170", "11545": "11180", "11560": "11190", "11590": "11200",
    "11620": "11210", "11650": "11220", "11680": "11230", "11710": "11240",
    "11740": "11250",
    # 인천
    "28110": "23010", "28140": "23020", "28177": "23030", "28185": "23040",
    "28200": "23050", "28237": "23060", "28245": "23070", "28260": "23080",
    # 경기
    "41111": "31011", "41113": "31012", "41115": "31013", "41117": "31014",
    "41131": "31021", "41133": "31022", "41135": "31023",
    "41171": "31041", "41173": "31042",
    "41192": "31050", "41194": "31050", "41196": "31050",  # 부천 3개 구 → 부천시
    "41281": "31101", "41285": "31103", "41287": "31104",
    "41290": "31110", "41360": "31130", "41450": "31180",
    "41461": "31191", "41463": "31192", "41465": "31193",
    "41570": "31230", "41590": "31240",
}

SIDO_MAP = {
    "서울": [k for k in LAWD_TO_GU if k.startswith("11")],
    "경기": [k for k in LAWD_TO_GU if k.startswith("41")],
    "인천": [k for k in LAWD_TO_GU if k.startswith("28")],
}


# ── 데이터 로드 ────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_trade(lawd_codes: tuple, year_start: int, year_end: int) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    ph = ",".join("?" * len(lawd_codes))
    df = pd.read_sql(
        f"""SELECT deal_year, deal_month, deal_amount, area_exclusive,
                   floor, building_age, lawd_cd, dealing_type
            FROM apt_trade
            WHERE lawd_cd IN ({ph})
              AND deal_year BETWEEN ? AND ?
              AND deal_amount > 0
              AND (cancel_type IS NULL OR cancel_type != 'Y')""",
        conn,
        params=list(lawd_codes) + [year_start, year_end],
    )
    conn.close()
    df["gu_name"] = df["lawd_cd"].map(LAWD_TO_GU).fillna(df["lawd_cd"])
    return df


# ── 사이드바 필터 ──────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 필터")

    sido_sel = st.selectbox("시도", ["전체"] + list(SIDO_MAP.keys()))
    pool = SIDO_MAP.get(sido_sel, list(LAWD_TO_GU.keys()))

    gu_opts = {LAWD_TO_GU.get(k, k): k for k in pool}
    sel_gu  = st.multiselect("구/시 (빈칸=전체)", list(gu_opts.keys()))
    codes   = tuple(gu_opts[g] for g in sel_gu) if sel_gu else tuple(pool)

    years   = st.slider("기간 (연도)", 2020, 2025, (2020, 2025))

if not codes:
    st.warning("지역을 선택해주세요.")
    st.stop()

df = load_trade(codes, years[0], years[1])

if df.empty:
    st.warning("데이터가 없습니다. `collect` 명령을 먼저 실행하세요.")
    st.stop()

# ── KPI ──────────────────────────────────────────────────────────────
st.subheader("📌 요약")
k1, k2, k3, k4 = st.columns(4)
k1.metric("총 거래건수",  f"{len(df):,}건")
k2.metric("평균 매매가",  f"{df['deal_amount'].mean() / 10000:.2f} 억원")
k3.metric("중간 매매가",  f"{df['deal_amount'].median() / 10000:.2f} 억원")
k4.metric("최고 매매가",  f"{df['deal_amount'].max() / 10000:.2f} 억원")

st.divider()

# ── 탭 ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 월별 추이", "🏙️ 지역 비교", "📐 면적 vs 가격", "🗺️ 가격 지도"]
)

# ── 탭 1: 월별 추이 ───────────────────────────────────────────────
with tab1:
    monthly = (
        df.groupby(["deal_year", "deal_month"])["deal_amount"]
        .mean()
        .reset_index()
    )
    monthly["date"] = pd.to_datetime(
        monthly["deal_year"].astype(str)
        + "-"
        + monthly["deal_month"].astype(str).str.zfill(2)
    )
    monthly = monthly.sort_values("date")
    monthly["avg_eok"] = (monthly["deal_amount"] / 10000).round(2)

    fig = px.line(
        monthly,
        x="date",
        y="avg_eok",
        title="월별 평균 매매가 추이",
        labels={"date": "거래월", "avg_eok": "평균 매매가 (억원)"},
        markers=True,
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ── 탭 2: 지역 비교 ───────────────────────────────────────────────
with tab2:
    reg = (
        df.groupby("gu_name")["deal_amount"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={"mean": "avg", "median": "med", "count": "cnt"})
        .sort_values("avg", ascending=True)
    )
    reg["avg_eok"] = (reg["avg"] / 10000).round(2)
    reg["med_eok"] = (reg["med"] / 10000).round(2)

    fig = px.bar(
        reg,
        x="avg_eok",
        y="gu_name",
        orientation="h",
        title="지역별 평균 매매가",
        labels={"avg_eok": "평균 매매가 (억원)", "gu_name": "지역"},
        color="avg_eok",
        color_continuous_scale="RdYlGn_r",
        text="avg_eok",
        hover_data={"med_eok": True, "cnt": True},
    )
    fig.update_traces(texttemplate="%{text:.2f}억", textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        height=max(400, len(reg) * 28),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── 탭 3: 면적 vs 가격 ────────────────────────────────────────────
with tab3:
    sample = df.sample(min(5000, len(df)), random_state=42)
    sample["deal_amount_eok"] = (sample["deal_amount"] / 10000).round(2)

    fig = px.scatter(
        sample,
        x="area_exclusive",
        y="deal_amount_eok",
        color="gu_name",
        opacity=0.45,
        title="전용면적 vs 매매가",
        labels={
            "area_exclusive":   "전용면적 (㎡)",
            "deal_amount_eok":  "매매가 (억원)",
            "gu_name":          "지역",
        },
        hover_data=["floor", "building_age"],
    )
    fig.update_layout(height=560)
    st.plotly_chart(fig, use_container_width=True)

# ── 탭 4: 가격 지도 ───────────────────────────────────────────────
with tab4:
    if not GEO_PATH.exists():
        st.info("GeoJSON 파일이 없습니다. `train` 명령 실행 시 자동으로 다운로드됩니다.")
    else:
        try:
            import folium
            from streamlit_folium import st_folium

            with open(GEO_PATH, encoding="utf-8") as f:
                geojson = json.load(f)

            # lawd_cd → GeoJSON code 매핑으로 집계 (이름 불일치 방지)
            avg_by_lawd = (
                df.groupby("lawd_cd")["deal_amount"]
                .mean()
                .div(10000)
                .round(2)
                .reset_index()
            )
            avg_by_lawd["geo_code"] = avg_by_lawd["lawd_cd"].map(LAWD_TO_GEO_CODE)
            avg_by_lawd["gu_name"]  = avg_by_lawd["lawd_cd"].map(LAWD_TO_GU)
            # 부천시처럼 여러 lawd_cd가 같은 geo_code일 경우 평균
            avg_by_code = (
                avg_by_lawd.dropna(subset=["geo_code"])
                .groupby(["geo_code", "gu_name"])["deal_amount"]
                .mean()
                .reset_index()
                .rename(columns={"deal_amount": "avg_eok"})
            )

            m = folium.Map(
                location=[37.5665, 127.0],
                zoom_start=10,
                tiles="CartoDB positron",
            )

            folium.Choropleth(
                geo_data=geojson,
                data=avg_by_code,
                columns=["geo_code", "avg_eok"],
                key_on="feature.properties.code",
                fill_color="YlOrRd",
                fill_opacity=0.7,
                line_opacity=0.3,
                legend_name="평균 매매가 (억원)",
                nan_fill_color="lightgray",
                highlight=True,
            ).add_to(m)

            # 클릭 가능한 GeoJson 레이어 (툴팁 + 하이라이트)
            price_by_code = avg_by_code.set_index("geo_code")["avg_eok"].to_dict()
            name_by_code  = avg_by_code.set_index("geo_code")["gu_name"].to_dict()
            for feature in geojson["features"]:
                code  = feature["properties"].get("code", "")
                price = price_by_code.get(code)
                gu    = name_by_code.get(code, feature["properties"].get("name", ""))
                label = f"{gu}: {price:.2f}억원" if price else gu
                folium.GeoJson(
                    feature,
                    style_function=lambda x: {
                        "fillOpacity": 0,
                        "weight": 0.5,
                        "color": "#555",
                    },
                    highlight_function=lambda x: {
                        "fillOpacity": 0.2,
                        "weight": 2.5,
                        "color": "#333",
                        "fillColor": "#fff",
                    },
                    tooltip=folium.Tooltip(label, sticky=True),
                ).add_to(m)

            st.caption("🖱️ 지역을 클릭하면 상세 정보가 아래에 표시됩니다.")
            map_data = st_folium(m, height=520, use_container_width=True)

            # ── 클릭 지역 상세 정보 ────────────────────────────────
            clicked_gu = None
            tooltip_text = (map_data or {}).get("last_object_clicked_tooltip", "")
            if tooltip_text and ":" in tooltip_text:
                clicked_gu = tooltip_text.split(":")[0].strip()

            if clicked_gu and clicked_gu in df["gu_name"].values:
                st.divider()
                st.subheader(f"📍 {clicked_gu} 상세 정보")

                gu_df = df[df["gu_name"] == clicked_gu]

                # KPI
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("총 거래건수",  f"{len(gu_df):,}건")
                c2.metric("평균 매매가",  f"{gu_df['deal_amount'].mean() / 10000:.2f} 억원")
                c3.metric("중간 매매가",  f"{gu_df['deal_amount'].median() / 10000:.2f} 억원")
                c4.metric("최고 매매가",  f"{gu_df['deal_amount'].max() / 10000:.2f} 억원")

                col_left, col_right = st.columns(2)

                # 연도별 평균가 추이
                with col_left:
                    yearly = (
                        gu_df.groupby("deal_year")["deal_amount"]
                        .mean()
                        .div(10000)
                        .round(2)
                        .reset_index()
                    )
                    yearly.columns = ["연도", "평균가(억원)"]
                    fig_y = px.bar(
                        yearly,
                        x="연도",
                        y="평균가(억원)",
                        title=f"{clicked_gu} 연도별 평균 매매가",
                        color="평균가(억원)",
                        color_continuous_scale="YlOrRd",
                        text="평균가(억원)",
                    )
                    fig_y.update_traces(texttemplate="%{text:.2f}억", textposition="outside")
                    fig_y.update_layout(coloraxis_showscale=False, showlegend=False)
                    st.plotly_chart(fig_y, use_container_width=True)

                # 전용면적별 가격 분포
                with col_right:
                    gu_df = gu_df.copy()
                    gu_df["deal_amount_eok"] = gu_df["deal_amount"] / 10000
                    fig_s = px.scatter(
                        gu_df.sample(min(2000, len(gu_df)), random_state=42),
                        x="area_exclusive",
                        y="deal_amount_eok",
                        color="deal_year",
                        opacity=0.5,
                        title=f"{clicked_gu} 면적 vs 가격",
                        labels={
                            "area_exclusive":   "전용면적 (㎡)",
                            "deal_amount_eok":  "매매가 (억원)",
                            "deal_year":        "거래연도",
                        },
                        color_continuous_scale="Blues",
                    )
                    st.plotly_chart(fig_s, use_container_width=True)

        except ImportError:
            st.warning(
                "folium 또는 streamlit-folium이 설치되지 않았습니다.\n"
                "`pip install folium streamlit-folium`"
            )
        except Exception as e:
            st.error(f"지도 오류: {e}")
            avg_df = df.groupby("gu_name")["deal_amount"].mean().div(10000).round(2).reset_index()
            avg_df.columns = ["지역", "평균가(억원)"]
            fig = px.bar(
                avg_df.sort_values("평균가(억원)", ascending=False),
                x="지역", y="평균가(억원)",
                title="지역별 평균 매매가 (지도 대체)",
                color="평균가(억원)",
                color_continuous_scale="YlOrRd",
            )
            st.plotly_chart(fig, use_container_width=True)
