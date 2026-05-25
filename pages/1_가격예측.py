import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.realestate.spatial import SUBWAY_STATIONS, OFFICE_LOCATIONS

st.set_page_config(page_title="가격 예측", page_icon="🔮", layout="wide")

# ── 경로 ──────────────────────────────────────────────────────────
DB_PATH   = Path("data/processed/realestate.db")
MODEL_DIR = Path("data/models")
GEO_PATH  = Path("data/geo/skorea_municipalities.geojson")

MODEL_TYPE_LABELS = {
    "lgbm":    "LightGBM",
    "xgboost": "XGBoost",
    "rf":      "RandomForest",
    "et":      "ExtraTrees",
    "gbm":     "GradientBoosting",
    "ridge":   "Ridge",
}

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


SPLIT_LABELS = {"time": "시계열", "complex": "단지"}

# ── 모델 스캔 ──────────────────────────────────────────────────────
@st.cache_data
def _scan_trade_models() -> dict[str, Path]:
    """data/models/price_model_trade_{model}_{split}.pkl 스캔 → {표시명: 경로}"""
    if not MODEL_DIR.exists():
        return {}
    result = {}
    for p in sorted(MODEL_DIR.glob("price_model_trade_*.pkl")):
        stem   = p.stem.replace("price_model_trade_", "")
        parts  = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in SPLIT_LABELS:
            mtype, split = parts
        else:
            mtype, split = stem, "time"
        model_label = MODEL_TYPE_LABELS.get(mtype, mtype.upper())
        split_label = SPLIT_LABELS.get(split, split)
        try:
            obj = joblib.load(str(p))
            r2  = obj.get("metrics", {}).get("r2", None)
            label = f"{model_label} / {split_label}  (R² {r2:.3f})" if r2 is not None else f"{model_label} / {split_label}"
        except Exception:
            label = f"{model_label} / {split_label}"
        result[label] = p
    return result


# ── 캐시 로드 ──────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path_str: str):
    p = Path(model_path_str)
    if not p.exists():
        return None
    return joblib.load(p)


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


# ── 모델 선택 (사이드바보다 먼저 스캔) ────────────────────────────
_available_models = _scan_trade_models()

# ── 리소스 로드 ────────────────────────────────────────────────────
geocode_avg = load_geocode_avg()
gdf         = load_gdf()

# ── 사이드바 입력 ──────────────────────────────────────────────────
with st.sidebar:
    st.header("📝 조건 입력")

    # 모델 선택
    if _available_models:
        selected_model_label = st.selectbox(
            "예측 모델",
            list(_available_models.keys()),
            help="학습된 모델 중 하나를 선택하세요",
        )
        selected_model_path = _available_models[selected_model_label]

        # 시계열 vs 단지 분리 비교표
        with st.expander("분리 방식별 성능 비교", expanded=False):
            rows = []
            for lbl, p in _available_models.items():
                try:
                    obj   = joblib.load(str(p))
                    m     = obj.get("metrics", {})
                    stem  = p.stem.replace("price_model_trade_", "")
                    parts = stem.rsplit("_", 1)
                    mtype = parts[0] if len(parts) == 2 else stem
                    split = parts[1] if len(parts) == 2 else "time"
                    rows.append({
                        "모델":     MODEL_TYPE_LABELS.get(mtype, mtype.upper()),
                        "분리 방식": SPLIT_LABELS.get(split, split),
                        "R²":       round(m.get("r2", float("nan")), 4),
                        "MAE (억원)": round(m.get("mae_eok", float("nan")), 2),
                    })
                except Exception:
                    pass
            if rows:
                import pandas as _pd
                cmp_df = (
                    _pd.DataFrame(rows)
                    .sort_values(["모델", "분리 방식"])
                    .reset_index(drop=True)
                )
                st.dataframe(cmp_df, hide_index=True, use_container_width=True)
    else:
        st.error("학습된 모델이 없습니다.")
        st.code("python -m src.realestate.main train --model lgbm --cutoff 2025-01-01")
        st.stop()

    st.divider()

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

# ── 선택된 모델 로드 ───────────────────────────────────────────────
model_data = load_model(str(selected_model_path))
if model_data is None:
    st.error(f"모델 파일을 읽을 수 없습니다: {selected_model_path}")
    st.stop()

# 타이틀 (모델 선택 후 표시)
_mtype   = model_data.get("model_type", selected_model_label)
_mae_eok = model_data.get("metrics", {}).get("mae_eok", None)
_r2      = model_data.get("metrics", {}).get("r2", None)
_caption = f"{selected_model_label} 모델"
if _mae_eok is not None and _r2 is not None:
    _caption += f" | MAE ≈ {_mae_eok:.2f}억원, R² {_r2:.2f}"
st.title("🔮 아파트 매매가 예측")
st.caption(_caption)

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


# ── SHAP 해석 ─────────────────────────────────────────────────────

FEATURE_LABELS_KO = {
    "area_exclusive":      "전용면적",
    "floor":               "층",
    "building_age":        "건물연령",
    "deal_year":           "거래연도",
    "deal_month":          "거래월",
    "deal_season":         "계절",
    "latitude":            "위도",
    "longitude":           "경도",
    "dist_to_gangnam_km":  "강남역 거리(km)",
    "dist_to_subway_km":   "지하철역 거리(km)",
    "dist_to_cityhall_km": "구청 거리(km)",
    "dealing_type":        "거래유형",
    "sgg_code":            "시군구",
}


@st.cache_resource(show_spinner="SHAP 분석 준비 중...")
def get_shap_explainer_and_background(model_path_str: str):
    """
    DB 샘플 2000건으로 전역 SHAP 값 사전 계산 (모델별 1회 캐시)

    Returns:
        (explainer, shap_values, X_sample, feature_names) 또는 None
    """
    try:
        import shap as shap_lib
    except ImportError:
        return None

    model_data = load_model(model_path_str)
    if model_data is None or not DB_PATH.exists():
        return None

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT t.deal_date, t.deal_amount, t.area_exclusive, t.floor,
               t.build_year, t.dealing_type, t.lawd_cd,
               g.latitude, g.longitude
        FROM apt_trade t
        JOIN apt_geocode g ON (t.apt_name || '_' || t.lawd_cd) = g.apt_key
        WHERE g.source != 'failed' AND t.deal_year < 2024 AND t.deal_amount > 0
        ORDER BY RANDOM() LIMIT 2000
        """,
        conn,
    )
    conn.close()

    if len(df) < 100:
        return None

    # 시간 피처
    dt = pd.to_datetime(df["deal_date"], errors="coerce")
    df["deal_year"]    = dt.dt.year.fillna(2023).astype(float)
    df["deal_month"]   = dt.dt.month.fillna(6).astype(float)
    df["deal_season"]  = df["deal_month"].map(
        lambda m: 0.0 if m in (12, 1, 2) else 1.0 if m in (3, 4, 5)
        else 2.0 if m in (6, 7, 8) else 3.0
    )
    df["building_age"] = (2025 - df["build_year"].fillna(2000)).astype(float)

    # 공간 피처
    lats = df["latitude"].values
    lngs = df["longitude"].values
    df["dist_to_gangnam_km"]  = [
        _haversine_km(la, lo, GANGNAM_LAT, GANGNAM_LNG)
        for la, lo in zip(lats, lngs)
    ]
    df["dist_to_subway_km"]   = [
        _nearest(la, lo, _STA_COORDS, _STA_NAMES)[1]
        for la, lo in zip(lats, lngs)
    ]
    df["dist_to_cityhall_km"] = [
        _nearest(la, lo, _OFF_COORDS, _OFF_NAMES)[1]
        for la, lo in zip(lats, lngs)
    ]
    df["sgg_code"] = df["lawd_cd"]

    feature_names  = model_data["feature_names"]
    label_encoders = model_data["label_encoders"]

    X = pd.DataFrame(index=df.index)
    for col in feature_names:
        X[col] = df.get(col, pd.Series(0.0, index=df.index))

    for col, le in label_encoders.items():
        if col in X.columns:
            X[col] = X[col].fillna("unknown").astype(str)
            known  = set(le.classes_)
            X[col] = X[col].apply(lambda v: v if v in known else le.classes_[0])
            X[col] = le.transform(X[col])

    X = X.fillna(X.median()).astype(float)

    try:
        explainer   = shap_lib.TreeExplainer(model_data["model"])
        shap_values = np.array(explainer.shap_values(X))
    except Exception:
        return None

    return explainer, shap_values, X, feature_names


st.divider()
st.subheader("🔬 SHAP 모델 해석")

shap_result = get_shap_explainer_and_background(str(selected_model_path))

if shap_result is None:
    _is_linear = model_data.get("model_type", "") in ("ridge", "elasticnet", "lasso")
    if _is_linear:
        st.info(f"SHAP TreeExplainer는 트리 기반 모델(LightGBM·XGBoost)만 지원합니다. {selected_model_label}은 SHAP 분석을 제공하지 않습니다.")
    else:
        st.info("SHAP 분석을 위한 데이터가 부족합니다. DB와 모델을 확인하세요.")
else:
    explainer_obj, bg_shap_vals, bg_X, feat_names = shap_result

    tab_global, tab_local = st.tabs(["📊 전체 피처 영향도", "🎯 이 예측 설명"])

    with tab_global:
        st.caption("2,000건 샘플 기준 — 각 피처가 가격 예측에 기여하는 평균 절댓값 (log 공간)")
        mean_abs = np.abs(bg_shap_vals).mean(axis=0)
        shap_df = pd.DataFrame({
            "피처":        [FEATURE_LABELS_KO.get(f, f) for f in feat_names],
            "평균 |SHAP|": mean_abs,
        }).sort_values("평균 |SHAP|", ascending=True)

        fig_shap = px.bar(
            shap_df,
            x="평균 |SHAP|",
            y="피처",
            orientation="h",
            color="평균 |SHAP|",
            color_continuous_scale="Blues",
            text_auto=".3f",
        )
        fig_shap.update_layout(
            height=420,
            coloraxis_showscale=False,
            xaxis_title="평균 |SHAP| (log1p 가격 기준)",
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        with st.expander("📌 해석 가이드"):
            st.markdown(
                "- **값이 클수록** 해당 피처가 예측에 더 큰 영향을 미침\n"
                "- **위도·경도·강남역 거리**가 상위권 → 한국 부동산에서 입지의 압도적 영향 확인\n"
                "- **건물연령·전용면적**은 입지 다음으로 가격을 결정하는 물리적 요소\n"
                "- **거래유형(직거래)**의 영향이 낮으면 시세와 거래방식은 큰 관련 없음을 의미"
            )

    with tab_local:
        st.caption("현재 입력 조건에 대한 개별 예측 설명")

        try:
            import shap as shap_lib  # noqa: F811

            # 현재 입력 features → X_single
            X_single = pd.DataFrame([features])
            label_encoders_curr = model_data["label_encoders"]
            for col, le in label_encoders_curr.items():
                if col in X_single.columns:
                    val = str(X_single[col].iloc[0])
                    if val not in set(le.classes_):
                        val = le.classes_[0]
                    X_single[col] = le.transform([val])
            for col in feat_names:
                if col not in X_single.columns:
                    X_single[col] = 0.0
            X_single = X_single[feat_names].astype(float)

            sv_single = np.array(explainer_obj.shap_values(X_single))[0]
            base_val  = float(explainer_obj.expected_value)

            # 기여도 DataFrame (log 공간 → 억원 근사: contribution ≈ sv × expm1(base+sv) − expm1(base))
            base_price   = float(np.expm1(base_val))
            contributions_eok = []
            for sv in sv_single:
                delta = float(np.expm1(base_val + sv) - base_price)
                contributions_eok.append(delta / 10000)

            contrib_df = pd.DataFrame({
                "피처":      [FEATURE_LABELS_KO.get(f, f) for f in feat_names],
                "기여(억원)": contributions_eok,
            }).sort_values("기여(억원)", key=abs, ascending=True)

            colors = ["#E84C4C" if v > 0 else "#4C9BE8" for v in contrib_df["기여(억원)"]]
            fig_local = px.bar(
                contrib_df,
                x="기여(억원)",
                y="피처",
                orientation="h",
                color="기여(억원)",
                color_continuous_scale=[[0, "#4C9BE8"], [0.5, "#f0f0f0"], [1, "#E84C4C"]],
                text_auto=".2f",
            )
            fig_local.update_layout(
                height=420,
                coloraxis_showscale=False,
                xaxis_title="가격 기여도 (억원, 근사값)",
            )
            fig_local.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_local, use_container_width=True)

            st.caption(
                f"기준값: {base_price/10000:.2f}억 (전체 평균) | "
                "빨강=가격 상승 요인, 파랑=가격 하락 요인"
            )

        except Exception as e:
            st.info(f"개별 SHAP 계산 중 오류: {e}")
