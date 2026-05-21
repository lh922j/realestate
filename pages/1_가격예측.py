import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.realestate.spatial import SUBWAY_STATIONS, OFFICE_LOCATIONS

st.set_page_config(page_title="가격 예측", page_icon="🔮", layout="wide")
st.title("🔮 아파트 매매가 예측")
st.caption("LightGBM 모델 기반 | MAE ≈ 1.47억원, R² 0.90")

# ── 경로 ──────────────────────────────────────────────────────────
DB_PATH    = Path("data/processed/realestate.db")
MODEL_PATH = Path("data/models/price_model.pkl")
GEO_PATH   = Path("data/geo/skorea_municipalities.geojson")

GANGNAM_LAT, GANGNAM_LNG = 37.4979, 127.0276

# 지하철역·시청구청 좌표 배열 (캐시용)
_STA_NAMES  = list(SUBWAY_STATIONS.keys())
_STA_COORDS = np.array(list(SUBWAY_STATIONS.values()))
_OFF_NAMES  = list(OFFICE_LOCATIONS.keys())
_OFF_COORDS = np.array(list(OFFICE_LOCATIONS.values()))

# ── 지역 데이터 ────────────────────────────────────────────────────
REGIONS = {
    "서울": {
        "강남구": "11680", "서초구": "11650", "송파구": "11710",
        "강동구": "11740", "마포구": "11440", "용산구": "11170",
        "성동구": "11200", "광진구": "11215", "영등포구": "11560",
        "동작구": "11590", "관악구": "11620", "종로구": "11110",
        "중구":   "11140", "동대문구": "11230", "성북구": "11290",
        "강북구": "11305", "도봉구":  "11320", "노원구":  "11350",
        "은평구": "11380", "서대문구": "11410", "양천구": "11470",
        "강서구": "11500", "구로구":  "11530", "금천구":  "11545",
        "중랑구": "11260",
    },
    "경기": {
        "분당구(성남)": "41135", "수지구(용인)": "41465", "기흥구(용인)": "41463",
        "영통구(수원)": "41117", "팔달구(수원)": "41115", "화성시":      "41590",
        "하남시":      "41450", "과천시":       "41290", "김포시":       "41570",
        "남양주시":    "41360", "일산동구(고양)": "41285", "일산서구(고양)": "41287",
        "덕양구(고양)": "41281", "장안구(수원)": "41111", "권선구(수원)": "41113",
        "수정구(성남)": "41131", "중원구(성남)": "41133", "처인구(용인)": "41461",
        "만안구(안양)": "41171", "동안구(안양)": "41173",
        "원미구(부천)": "41192", "소사구(부천)": "41194", "오정구(부천)": "41196",
    },
    "인천": {
        "연수구": "28185", "남동구": "28200", "서구":   "28260",
        "부평구": "28237", "계양구": "28245", "미추홀구": "28177",
        "중구":   "28110", "동구":   "28140",
    },
}


# ── 헬퍼 ──────────────────────────────────────────────────────────
def _haversine_km(lat1, lng1, lat2, lng2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng / 2) ** 2)
    return float(R * 2 * np.arcsin(np.sqrt(a)))


def _nearest(lat: float, lng: float, coords: np.ndarray, names: list) -> tuple[str, float]:
    """coords 배열에서 (lat, lng)에 가장 가까운 지점의 (이름, 거리km) 반환"""
    R = 6371.0
    dlat = np.radians(coords[:, 0] - lat)
    dlng = np.radians(coords[:, 1] - lng)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat)) * np.cos(np.radians(coords[:, 0])) * np.sin(dlng / 2) ** 2)
    dists = R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    idx = int(np.argmin(dists))
    return names[idx], round(float(dists[idx]), 2)


def _season(month: int) -> int:
    return {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
            6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}[month]


# ── 캐시 로드 ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_geocode_avg() -> dict[str, tuple[float, float]]:
    if not DB_PATH.exists():
        return {}
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT lawd_cd, AVG(latitude) AS lat, AVG(longitude) AS lng "
        "FROM apt_geocode WHERE source != 'failed' AND latitude IS NOT NULL "
        "GROUP BY lawd_cd",
        conn,
    )
    conn.close()
    return {r.lawd_cd: (r.lat, r.lng) for r in df.itertuples()}


@st.cache_resource
def load_gdf():
    if not GEO_PATH.exists():
        return None
    try:
        import geopandas as gpd
        return gpd.read_file(GEO_PATH)
    except Exception:
        return None


@st.cache_data
def regional_stats(lawd_cd: str, area_min: float, area_max: float):
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(DB_PATH)
    row = pd.read_sql(
        "SELECT AVG(deal_amount) avg, MIN(deal_amount) min, "
        "MAX(deal_amount) max, COUNT(*) cnt "
        "FROM apt_trade "
        "WHERE lawd_cd=? AND area_exclusive BETWEEN ? AND ? AND deal_year>=2023",
        conn,
        params=(lawd_cd, area_min, area_max),
    ).iloc[0]
    conn.close()
    return None if pd.isna(row["avg"]) else row


def _sgg_code(lat: float, lng: float, gdf) -> str | None:
    try:
        from shapely.geometry import Point
        pt = Point(lng, lat)
        for _, row in gdf.iterrows():
            if row.geometry and row.geometry.contains(pt):
                return str(row["code"])
    except Exception:
        pass
    return None


def _predict(model_data: dict, features: dict) -> float:
    model          = model_data["model"]
    label_encoders = model_data["label_encoders"]
    feature_names  = model_data["feature_names"]

    X = pd.DataFrame([features])
    for col, le in label_encoders.items():
        if col in X.columns:
            val = str(X[col].iloc[0])
            if val not in set(le.classes_):
                val = "unknown" if "unknown" in set(le.classes_) else le.classes_[0]
            X[col] = le.transform([val])

    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_names].astype(float)
    return float(np.expm1(model.predict(X)[0]))


# ── 리소스 로드 ────────────────────────────────────────────────────
model_data   = load_model()
geocode_avg  = load_geocode_avg()
gdf          = load_gdf()

if model_data is None:
    st.warning("⚠️ 학습된 모델이 없습니다. 아래 명령을 먼저 실행하세요.")
    st.code("python -m src.realestate.main train --model lgbm --cutoff 2024-01-01")
    st.stop()

# ── 사이드바 입력 ──────────────────────────────────────────────────
with st.sidebar:
    st.header("📝 조건 입력")

    sido     = st.selectbox("시도", list(REGIONS.keys()))
    gu_map   = REGIONS[sido]
    gu_name  = st.selectbox("구 / 시", list(gu_map.keys()))
    lawd_cd  = gu_map[gu_name]

    area       = st.slider("전용면적 (㎡)", 20.0, 250.0, 84.0, step=1.0)
    floor      = st.slider("층", 1, 50, 10)
    build_year = st.slider("건축년도", 1970, 2024, 2010)

    deal_year  = st.selectbox("거래 연도", list(range(2020, 2027)), index=4)
    deal_month = st.selectbox("거래 월", list(range(1, 13)), index=0)
    dealing    = st.selectbox("거래 유형", ["중개거래", "직거래"])

    predict_btn = st.button("🔮 가격 예측", type="primary", use_container_width=True)

# ── 좌표 · 피처 계산 ───────────────────────────────────────────────
lat, lng = geocode_avg.get(lawd_cd, (37.5665, 126.9780))
sgg_code = (_sgg_code(lat, lng, gdf) if gdf is not None else None) or lawd_cd

dist_gangnam                = _haversine_km(lat, lng, GANGNAM_LAT, GANGNAM_LNG)
nearest_sta_name, dist_sta  = _nearest(lat, lng, _STA_COORDS, _STA_NAMES)
nearest_off_name, dist_off  = _nearest(lat, lng, _OFF_COORDS, _OFF_NAMES)

features = {
    "area_exclusive":       area,
    "floor":                float(floor),
    "building_age":         float(2025 - build_year),
    "deal_year":            float(deal_year),
    "deal_month":           float(deal_month),
    "deal_season":          float(_season(deal_month)),
    "latitude":             lat,
    "longitude":            lng,
    "dist_to_gangnam_km":   dist_gangnam,
    "dist_to_subway_km":    dist_sta,
    "dist_to_cityhall_km":  dist_off,
    "dealing_type":         dealing,
    "sgg_code":             sgg_code,
}

# ── 메인 영역 ──────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("예측 중..."):
        price = _predict(model_data, features)

    st.success(f"### 예상 매매가: **{price / 10000:.2f} 억원**  ({price:,.0f} 만원)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("예상 가격",   f"{price / 10000:.2f} 억원")
    c2.metric("건물 연령",   f"{2025 - build_year}년")
    c3.metric(f"최근접역 ({nearest_sta_name})", f"{dist_sta:.2f} km")
    c4.metric("강남역 거리", f"{dist_gangnam:.1f} km")

    # 지역 시세 비교
    stats = regional_stats(lawd_cd, max(10.0, area - 15), area + 15)
    if stats is not None:
        st.divider()
        st.subheader(f"📊 {gu_name} 실거래 시세 (2023년 이후, {area - 15:.0f}–{area + 15:.0f}㎡)")

        diff_pct = (price - stats["avg"]) / stats["avg"] * 100
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("지역 평균가", f"{stats['avg'] / 10000:.2f} 억원")
        d2.metric("지역 최저가", f"{stats['min'] / 10000:.2f} 억원")
        d3.metric("지역 최고가", f"{stats['max'] / 10000:.2f} 억원")
        d4.metric("예측 vs 평균", f"{diff_pct:+.1f}%", delta_color="off")
        st.caption(f"비교 샘플: {int(stats['cnt']):,}건")
else:
    st.info("👈 왼쪽에서 조건을 입력하고 **가격 예측** 버튼을 누르세요.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전용면적",   f"{area:.0f} ㎡")
    c2.metric("층",        f"{floor}층")
    c3.metric(f"최근접역 ({nearest_sta_name})", f"{dist_sta:.2f} km")
    c4.metric("강남역 거리", f"{dist_gangnam:.1f} km")
