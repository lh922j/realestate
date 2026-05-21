"""
spatial.py - 좌표 기반 공간 피처 생성

위경도 좌표에서 다음 피처를 추가합니다:
  - dist_to_gangnam_km:  강남역까지 직선 거리 (Haversine)
  - dist_to_subway_km:   가장 가까운 지하철역까지 거리
  - dist_to_cityhall_km: 가장 가까운 시청/구청까지 거리
  - sgg_name:            시군구명 (GeoJSON 경계 기반 공간 조인)
  - sgg_code:            시군구 코드

GeoJSON 데이터:
    southkorea/southkorea-maps (GitHub) — 행정구역 시군구 경계
    최초 실행 시 자동 다운로드 → data/geo/ 에 저장
"""

from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger

from .config import GEO_DIR, GANGNAM_LAT, GANGNAM_LNG

# ─────────────────────────────────────────────────────────────────
# 수도권 지하철역 좌표 (주요 환승역 + 간선역 ~200개)
# ─────────────────────────────────────────────────────────────────
# fmt: off
SUBWAY_STATIONS: dict[str, tuple[float, float]] = {
    # 1호선
    "인천역":       (37.4739, 126.6217), "제물포":     (37.4636, 126.6503),
    "도화":         (37.4689, 126.6643), "주안":       (37.4749, 126.6888),
    "간석":         (37.4796, 126.7145), "동암":       (37.4892, 126.7223),
    "부평":         (37.5082, 126.7232), "부개":       (37.5122, 126.7508),
    "역곡":         (37.4914, 126.7997), "소사":       (37.4877, 126.8027),
    "부천":         (37.5034, 126.7660), "중동":       (37.5024, 126.7850),
    "송내":         (37.4982, 126.8090), "시흥":       (37.4795, 126.8030),
    "온수":         (37.4924, 126.8248), "오류동":     (37.4970, 126.8420),
    "구일":         (37.4964, 126.8570), "구로":       (37.4997, 126.8826),
    "신도림":       (37.5085, 126.8913), "영등포":     (37.5157, 126.9064),
    "신길":         (37.5171, 126.9202), "대방":       (37.5145, 126.9349),
    "노량진":       (37.5132, 126.9424), "용산":       (37.5299, 126.9654),
    "남영":         (37.5432, 126.9722), "서울역_1":   (37.5547, 126.9707),
    "시청_1":       (37.5642, 126.9750), "종각":       (37.5701, 126.9826),
    "종로3가_1":    (37.5717, 126.9919), "종로5가":    (37.5717, 127.0005),
    "동대문_1":     (37.5717, 127.0096), "청량리":     (37.5805, 127.0455),
    "회기":         (37.5894, 127.0576), "외대앞":     (37.5948, 127.0601),
    "망우":         (37.6005, 127.0883), "도봉":       (37.6786, 127.0453),
    "도봉산_1":     (37.6891, 127.0443), "수원":       (37.2665, 127.0001),
    "의왕":         (37.3565, 126.9841), "군포":       (37.3615, 126.9372),
    "안양":         (37.3957, 126.9556), "관악_1호":   (37.4121, 126.9376),
    "금천구청":     (37.4568, 126.8988), "석수":       (37.4664, 126.9095),
    # 2호선
    "시청_2":       (37.5642, 126.9750), "을지로입구": (37.5660, 126.9825),
    "을지로3가_2":  (37.5660, 126.9912), "을지로4가":  (37.5662, 127.0000),
    "동대문역사문화": (37.5650, 127.0085), "왕십리_2":  (37.5613, 127.0372),
    "한양대":       (37.5559, 127.0447), "뚝섬":       (37.5477, 127.0472),
    "성수":         (37.5445, 127.0562), "건대입구":   (37.5403, 127.0697),
    "구의":         (37.5384, 127.0848), "강변":       (37.5340, 127.0939),
    "잠실나루":     (37.5161, 127.0985), "잠실_2":     (37.5133, 127.1001),
    "잠실새내":     (37.5095, 127.1007), "종합운동장": (37.5099, 127.0736),
    "삼성_2":       (37.5088, 127.0627), "선릉":       (37.5044, 127.0492),
    "역삼":         (37.5004, 127.0363), "강남_2":     (37.4979, 127.0276),
    "교대_2":       (37.4934, 127.0141), "서초":       (37.4917, 127.0079),
    "방배":         (37.4818, 126.9977), "사당_2":     (37.4766, 126.9815),
    "낙성대":       (37.4781, 126.9627), "서울대입구": (37.4813, 126.9527),
    "신림":         (37.4843, 126.9298), "신대방":     (37.4876, 126.9163),
    "구로디지털단지": (37.4851, 126.9013), "대림_2":   (37.4921, 126.8951),
    "문래":         (37.5188, 126.8950), "영등포구청_2": (37.5261, 126.8952),
    "당산_2":       (37.5337, 126.9016), "합정_2":     (37.5499, 126.9139),
    "홍대입구":     (37.5572, 126.9249), "신촌_2":     (37.5553, 126.9369),
    "이대":         (37.5565, 126.9466), "아현":       (37.5559, 126.9565),
    "충정로_2":     (37.5592, 126.9645),
    # 3호선
    "연신내_3":     (37.6191, 126.9186), "불광_3":     (37.6099, 126.9298),
    "녹번":         (37.6010, 126.9369), "홍제":       (37.5952, 126.9473),
    "독립문":       (37.5855, 126.9610), "경복궁":     (37.5761, 126.9698),
    "안국":         (37.5782, 126.9855), "종로3가_3":  (37.5717, 126.9919),
    "을지로3가_3":  (37.5660, 126.9912), "충무로_3":   (37.5614, 126.9931),
    "약수_3":       (37.5530, 127.0116), "금호":       (37.5486, 127.0173),
    "옥수":         (37.5445, 127.0175), "압구정":     (37.5270, 127.0281),
    "신사":         (37.5199, 127.0202), "잠원":       (37.5119, 127.0121),
    "고속터미널_3": (37.5047, 127.0047), "교대_3":     (37.4934, 127.0141),
    "남부터미널":   (37.4847, 127.0160), "양재_3":     (37.4848, 127.0349),
    "도곡":         (37.4949, 127.0451), "대치":       (37.4950, 127.0635),
    "학여울":       (37.4904, 127.0800), "수서_3":     (37.4876, 127.1013),
    "가락시장_3":   (37.4966, 127.1176),
    # 4호선
    "당고개":       (37.6576, 127.0649), "노원_4":     (37.6556, 127.0637),
    "창동_4":       (37.6526, 127.0472), "수유":       (37.6375, 127.0252),
    "미아사거리":   (37.6119, 127.0311), "길음":       (37.6023, 127.0255),
    "성신여대입구": (37.5920, 127.0165), "한성대입구": (37.5883, 127.0069),
    "혜화":         (37.5824, 127.0015), "동대문_4":   (37.5717, 127.0096),
    "명동":         (37.5600, 126.9857), "회현":       (37.5574, 126.9783),
    "서울역_4":     (37.5547, 126.9707), "숙대입구":   (37.5438, 126.9717),
    "삼각지_4":     (37.5378, 126.9720), "이촌_4":     (37.5246, 126.9601),
    "동작_4":       (37.5020, 126.9800), "사당_4":     (37.4766, 126.9815),
    "남태령":       (37.4642, 126.9814),
    # 5호선
    "방화":         (37.5720, 126.8013), "김포공항_5": (37.5627, 126.8020),
    "화곡":         (37.5482, 126.8493), "까치산":     (37.5411, 126.8647),
    "목동":         (37.5256, 126.8745), "오목교":     (37.5266, 126.8836),
    "영등포구청_5": (37.5261, 126.8952), "여의도_5":   (37.5219, 126.9245),
    "마포_5":       (37.5409, 126.9407), "공덕_5":     (37.5446, 126.9523),
    "광화문":       (37.5757, 126.9769), "종로3가_5":  (37.5717, 126.9919),
    "왕십리_5":     (37.5613, 127.0372), "답십리":     (37.5706, 127.0553),
    "군자_5":       (37.5597, 127.0795), "천호_5":     (37.5385, 127.1237),
    "강동_5":       (37.5303, 127.1255), "고덕":       (37.5527, 127.1614),
    # 6호선
    "디지털미디어시티_6": (37.5746, 126.8914), "마포구청_6": (37.5648, 126.9089),
    "망원":         (37.5560, 126.9076), "합정_6":     (37.5499, 126.9139),
    "이태원":       (37.5345, 126.9943), "한강진":     (37.5303, 127.0044),
    "약수_6":       (37.5530, 127.0116), "안암":       (37.5892, 127.0264),
    "월곡":         (37.6016, 127.0367), "태릉입구_6": (37.6260, 127.0745),
    # 7호선
    "도봉산_7":     (37.6891, 127.0443), "노원_7":     (37.6556, 127.0637),
    "태릉입구_7":   (37.6260, 127.0745), "건대입구_7": (37.5403, 127.0697),
    "청담":         (37.5216, 127.0535), "강남구청_7": (37.5173, 127.0444),
    "논현":         (37.5113, 127.0228), "반포_7":     (37.5087, 127.0007),
    "고속터미널_7": (37.5047, 127.0047), "이수":       (37.4867, 126.9811),
    "대림_7":       (37.4921, 126.8951), "가산디지털단지": (37.4776, 126.8825),
    "철산":         (37.4767, 126.8633), "부천시청_7": (37.5034, 126.7660),
    "까치울":       (37.4972, 126.7898), "온수_7":     (37.4924, 126.8248),
    # 8호선
    "암사":         (37.5519, 127.1337), "천호_8":     (37.5385, 127.1237),
    "잠실_8":       (37.5133, 127.1001), "석촌":       (37.5029, 127.1043),
    "가락시장_8":   (37.4966, 127.1176), "복정":       (37.4736, 127.1468),
    # 9호선
    "김포공항_9":   (37.5627, 126.8020), "가양":       (37.5618, 126.8642),
    "당산_9":       (37.5337, 126.9016), "여의도_9":   (37.5219, 126.9245),
    "노량진_9":     (37.5132, 126.9424), "동작_9":     (37.5020, 126.9800),
    "고속터미널_9": (37.5047, 127.0047), "신논현":     (37.5047, 127.0253),
    "선정릉":       (37.5098, 127.0493), "봉은사":     (37.5154, 127.0653),
    "종합운동장_9": (37.5099, 127.0736), "석촌_9":     (37.5029, 127.1043),
    # 분당선
    "수원_분당":    (37.2665, 127.0001), "정자":       (37.3629, 127.1142),
    "미금":         (37.3531, 127.1169), "오리":       (37.3297, 127.1024),
    "죽전":         (37.3155, 127.1073), "기흥":       (37.2714, 127.1172),
    "야탑":         (37.4111, 127.1277), "이매":       (37.3943, 127.1294),
    "수내":         (37.3783, 127.1170), "서현":       (37.3697, 127.1170),
    # 신분당선
    "강남_신분당":  (37.4979, 127.0276), "양재_신분당": (37.4848, 127.0349),
    "청계산입구":   (37.4549, 127.0538), "판교":       (37.3945, 127.1108),
    "동천":         (37.3470, 127.1162),
    # 경의중앙선
    "공덕_경의":    (37.5446, 126.9523), "홍대입구_경의": (37.5572, 126.9249),
    "디지털미디어시티_경의": (37.5746, 126.8914),
    "수색":         (37.5843, 126.8735), "능곡":       (37.6300, 126.8268),
    "행신":         (37.6194, 126.8375),
    # 인천 1호선
    "계양_인천":    (37.5978, 126.7390), "임학":       (37.5610, 126.7187),
    "부평_인천":    (37.5082, 126.7232), "인천시청_인천": (37.4568, 126.7055),
    "인천터미널":   (37.4549, 126.6879), "원인재":     (37.4121, 126.6730),
    "인천대입구":   (37.4268, 126.6275), "센트럴파크": (37.4328, 126.6275),
    # 경강선
    "판교_경강":    (37.3945, 127.1108), "이매_경강":  (37.3943, 127.1294),
    "삼동":         (37.3792, 127.2135), "경기광주":   (37.3600, 127.2613),
}
# fmt: on

# ─────────────────────────────────────────────────────────────────
# 시청 / 구청 좌표
# ─────────────────────────────────────────────────────────────────
# fmt: off
OFFICE_LOCATIONS: dict[str, tuple[float, float]] = {
    # 서울특별시
    "서울시청":      (37.5665, 126.9780),
    "종로구청":      (37.5736, 126.9793), "중구청":       (37.5631, 126.9971),
    "용산구청":      (37.5324, 126.9899), "성동구청":     (37.5635, 127.0368),
    "광진구청":      (37.5378, 127.0822), "동대문구청":   (37.5747, 127.0432),
    "중랑구청":      (37.6064, 127.0924), "성북구청":     (37.5894, 127.0180),
    "강북구청":      (37.6399, 127.0253), "도봉구청":     (37.6688, 127.0472),
    "노원구청":      (37.6543, 127.0620), "은평구청":     (37.6026, 126.9289),
    "서대문구청":    (37.5791, 126.9368), "마포구청":     (37.5648, 126.9089),
    "양천구청":      (37.5169, 126.8665), "강서구청":     (37.5510, 126.8495),
    "구로구청":      (37.4955, 126.8876), "금천구청_청사": (37.4568, 126.8988),
    "영등포구청":    (37.5261, 126.8952), "동작구청":     (37.5133, 126.9395),
    "관악구청":      (37.4782, 126.9513), "서초구청":     (37.4836, 127.0327),
    "강남구청":      (37.5173, 127.0444), "송파구청":     (37.5146, 127.1060),
    "강동구청":      (37.5300, 127.1248),
    # 경기도 주요 시청
    "수원시청":      (37.2810, 127.0164), "성남시청":     (37.4201, 127.1265),
    "의정부시청":    (37.7382, 127.0425), "안양시청":     (37.3943, 126.9567),
    "부천시청":      (37.5034, 126.7660), "광명시청":     (37.4779, 126.8642),
    "동두천시청":    (37.9038, 127.0601), "고양시청":     (37.6582, 126.8320),
    "과천시청":      (37.4292, 126.9878), "구리시청":     (37.5966, 127.1296),
    "남양주시청":    (37.6360, 127.2164), "오산시청":     (37.1499, 127.0787),
    "시흥시청":      (37.3799, 126.8031), "군포시청":     (37.3614, 126.9358),
    "의왕시청":      (37.3449, 126.9683), "하남시청":     (37.5394, 127.2148),
    "용인시청":      (37.2411, 127.1774), "파주시청":     (37.7601, 126.7798),
    "김포시청":      (37.6154, 126.7152), "화성시청":     (37.1993, 126.8317),
    "양주시청":      (37.7854, 127.0458), "포천시청":     (37.8954, 127.2002),
    # 인천광역시
    "인천시청":      (37.4568, 126.7055), "중구청_인천":  (37.4742, 126.6196),
    "동구청_인천":   (37.4742, 126.6583), "미추홀구청":   (37.4568, 126.6756),
    "연수구청":      (37.4102, 126.6782), "남동구청":     (37.4448, 126.7312),
    "부평구청_인천": (37.5116, 126.7250), "계양구청":     (37.5399, 126.7383),
    "서구청_인천":   (37.5454, 126.7019), "강화군청":     (37.7447, 126.4882),
}
# fmt: on

_GEOJSON_URL = (
    "https://raw.githubusercontent.com/southkorea/southkorea-maps/"
    "master/kostat/2018/json/skorea-municipalities-2018-geo.json"
)
_GEOJSON_PATH = GEO_DIR / "skorea_municipalities.geojson"


def _haversine_km(
    lat1: np.ndarray | float,
    lng1: np.ndarray | float,
    lat2: float,
    lng2: float,
) -> np.ndarray:
    """두 좌표 간 Haversine 거리 (km), numpy 벡터화"""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng / 2) ** 2
    )
    return R * 2 * np.arcsin(np.sqrt(a))


def _haversine_km_matrix(
    lats: np.ndarray,        # shape (M,) — 아파트 위도
    lngs: np.ndarray,        # shape (M,) — 아파트 경도
    target_lats: np.ndarray, # shape (N,) — 목표지점 위도
    target_lngs: np.ndarray, # shape (N,) — 목표지점 경도
) -> np.ndarray:             # shape (M, N)
    """M개 아파트 × N개 목표지점 간 Haversine 거리 행렬 (km)"""
    R = 6371.0
    lats = lats[:, np.newaxis]   # (M, 1)
    lngs = lngs[:, np.newaxis]   # (M, 1)
    dlat = np.radians(target_lats - lats)    # (M, N)
    dlng = np.radians(target_lngs - lngs)   # (M, N)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lats)) * np.cos(np.radians(target_lats)) * np.sin(dlng / 2) ** 2
    )
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class SpatialFeatureBuilder:
    """
    좌표 → 공간 피처 생성기

    사용 예시:
        builder = SpatialFeatureBuilder()
        df = builder.build_features(df)   # latitude, longitude 컬럼 필요
        # df에 dist_to_gangnam_km, sgg_name, sgg_code 컬럼이 추가됨
    """

    def __init__(self, geo_dir: Path = GEO_DIR):
        self.geo_dir = geo_dir
        self._boundaries = None   # GeoDataFrame (lazy load)

    # ─────────────────────────────────────────────────────────────
    # GeoJSON 로드 / 다운로드
    # ─────────────────────────────────────────────────────────────

    def download_boundaries(self) -> Path:
        """
        시군구 경계 GeoJSON을 GitHub에서 다운로드

        Returns:
            저장된 파일 경로
        """
        logger.info(f"시군구 경계 GeoJSON 다운로드 중: {_GEOJSON_URL}")
        resp = requests.get(_GEOJSON_URL, timeout=30)
        resp.raise_for_status()

        _GEOJSON_PATH.write_bytes(resp.content)
        logger.info(f"GeoJSON 저장 완료: {_GEOJSON_PATH}")
        return _GEOJSON_PATH

    def load_boundaries(self):
        """
        시군구 경계 GeoDataFrame 로드 (없으면 자동 다운로드)

        Returns:
            geopandas.GeoDataFrame
        """
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "geopandas가 설치되지 않았습니다.\n"
                "pip install geopandas 또는 conda install geopandas"
            )

        if not _GEOJSON_PATH.exists():
            self.download_boundaries()

        gdf = gpd.read_file(_GEOJSON_PATH)
        logger.debug(f"경계 데이터 로드: {len(gdf)}개 시군구")
        return gdf

    def _get_boundaries(self):
        """경계 GeoDataFrame을 지연 로드 (한 번만 로드 후 재사용)"""
        if self._boundaries is None:
            self._boundaries = self.load_boundaries()
        return self._boundaries

    # ─────────────────────────────────────────────────────────────
    # 피처 생성
    # ─────────────────────────────────────────────────────────────

    def add_subway_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        """가장 가까운 지하철역까지 직선 거리(km) 컬럼 추가"""
        sta_coords = np.array(list(SUBWAY_STATIONS.values()))   # (N, 2)
        sta_lats   = sta_coords[:, 0]
        sta_lngs   = sta_coords[:, 1]

        df   = df.copy()
        mask = df["latitude"].notna() & df["longitude"].notna()
        df["dist_to_subway_km"] = np.nan

        if mask.any():
            lats  = df.loc[mask, "latitude"].to_numpy()
            lngs  = df.loc[mask, "longitude"].to_numpy()
            dists = _haversine_km_matrix(lats, lngs, sta_lats, sta_lngs)  # (M, N)
            df.loc[mask, "dist_to_subway_km"] = dists.min(axis=1).round(3)

        return df

    def add_office_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        """가장 가까운 시청/구청까지 직선 거리(km) 컬럼 추가"""
        off_coords = np.array(list(OFFICE_LOCATIONS.values()))  # (N, 2)
        off_lats   = off_coords[:, 0]
        off_lngs   = off_coords[:, 1]

        df   = df.copy()
        mask = df["latitude"].notna() & df["longitude"].notna()
        df["dist_to_cityhall_km"] = np.nan

        if mask.any():
            lats  = df.loc[mask, "latitude"].to_numpy()
            lngs  = df.loc[mask, "longitude"].to_numpy()
            dists = _haversine_km_matrix(lats, lngs, off_lats, off_lngs)  # (M, N)
            df.loc[mask, "dist_to_cityhall_km"] = dists.min(axis=1).round(3)

        return df

    def add_gangnam_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        강남역까지 직선 거리(km) 컬럼 추가

        Args:
            df: latitude, longitude 컬럼 포함

        Returns:
            dist_to_gangnam_km 컬럼이 추가된 DataFrame
        """
        mask = df["latitude"].notna() & df["longitude"].notna()
        df = df.copy()
        df["dist_to_gangnam_km"] = np.nan
        if mask.any():
            df.loc[mask, "dist_to_gangnam_km"] = _haversine_km(
                df.loc[mask, "latitude"].to_numpy(),
                df.loc[mask, "longitude"].to_numpy(),
                GANGNAM_LAT,
                GANGNAM_LNG,
            ).round(2)
        return df

    def add_sgg_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        시군구명/코드 컬럼 추가 (GeoJSON 경계와 공간 조인)

        좌표가 없는 행은 NaN으로 유지됩니다.

        Args:
            df: latitude, longitude 컬럼 포함

        Returns:
            sgg_name, sgg_code 컬럼이 추가된 DataFrame
        """
        try:
            import geopandas as gpd
            from shapely.geometry import Point
        except ImportError:
            logger.warning("geopandas 미설치 — sgg_name, sgg_code 컬럼을 추가하지 않습니다.")
            df = df.copy()
            df["sgg_name"] = None
            df["sgg_code"] = None
            return df

        df = df.copy()
        boundaries = self._get_boundaries()

        valid_mask = df["latitude"].notna() & df["longitude"].notna()
        if not valid_mask.any():
            df["sgg_name"] = None
            df["sgg_code"] = None
            return df

        valid_df = df[valid_mask].copy()
        geometry = [
            Point(lng, lat)
            for lat, lng in zip(valid_df["latitude"], valid_df["longitude"])
        ]
        points_gdf = gpd.GeoDataFrame(
            valid_df[["latitude", "longitude"]],
            geometry=geometry,
            crs="EPSG:4326",
        )

        # 경계 GeoDataFrame CRS 맞추기
        if boundaries.crs is None:
            boundaries = boundaries.set_crs("EPSG:4326")
        elif boundaries.crs.to_epsg() != 4326:
            boundaries = boundaries.to_crs("EPSG:4326")

        joined = gpd.sjoin(
            points_gdf,
            boundaries[["geometry", "name", "code"]],
            how="left",
            predicate="within",
        )

        df["sgg_name"] = None
        df["sgg_code"] = None
        df.loc[valid_mask, "sgg_name"] = joined["name"].values
        df.loc[valid_mask, "sgg_code"] = joined["code"].values

        return df

    def build_features(self, df: pd.DataFrame, use_boundaries: bool = True) -> pd.DataFrame:
        """
        좌표 기반 공간 피처 일괄 생성

        Args:
            df:              latitude, longitude 컬럼 포함 DataFrame
            use_boundaries:  False면 GeoJSON 공간 조인 스킵 (geopandas 미설치 환경용)

        Returns:
            dist_to_gangnam_km, dist_to_subway_km, dist_to_cityhall_km,
            sgg_name, sgg_code 컬럼이 추가된 DataFrame
        """
        if "latitude" not in df.columns or "longitude" not in df.columns:
            logger.warning("latitude/longitude 컬럼이 없습니다. 공간 피처를 추가하지 않습니다.")
            return df

        df = self.add_gangnam_distance(df)
        df = self.add_subway_distance(df)
        df = self.add_office_distance(df)

        if use_boundaries:
            df = self.add_sgg_info(df)

        filled_pct = df["dist_to_gangnam_km"].notna().mean() * 100
        logger.info(
            f"공간 피처 생성 완료 | 좌표 보유율: {filled_pct:.1f}% | "
            f"지하철 거리 min={df['dist_to_subway_km'].min():.2f}km | "
            f"구청 거리 min={df['dist_to_cityhall_km'].min():.2f}km"
        )
        return df
