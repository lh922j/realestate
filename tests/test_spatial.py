"""
test_spatial.py - 공간 피처 생성 단위 테스트

실행:
    pytest tests/test_spatial.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.realestate.spatial import (
    _haversine_km,
    _haversine_km_matrix,
    SpatialFeatureBuilder,
)

GANGNAM_LAT, GANGNAM_LNG = 37.4979, 127.0276


class TestHaversineKm:
    """_haversine_km 함수 테스트"""

    def test_same_point_is_zero(self):
        dist = _haversine_km(37.5, 127.0, 37.5, 127.0)
        assert float(dist) < 1e-6

    def test_gangnam_to_incheon_approx(self):
        """강남역 → 인천역 직선 약 35km"""
        INCHEON_LAT, INCHEON_LNG = 37.4739, 126.6217
        dist = float(_haversine_km(GANGNAM_LAT, GANGNAM_LNG, INCHEON_LAT, INCHEON_LNG))
        assert 30.0 < dist < 45.0

    def test_vectorized_returns_array(self):
        """배열 입력 시 numpy 배열 반환"""
        lats = np.array([37.4, 37.5, 37.6])
        lngs = np.array([127.0, 127.1, 127.2])
        dists = _haversine_km(lats, lngs, GANGNAM_LAT, GANGNAM_LNG)
        assert isinstance(dists, np.ndarray)
        assert len(dists) == 3

    def test_symmetry(self):
        """A→B 거리 == B→A 거리"""
        d1 = float(_haversine_km(37.5, 127.0, 37.6, 127.1))
        d2 = float(_haversine_km(37.6, 127.1, 37.5, 127.0))
        assert abs(d1 - d2) < 1e-9

    def test_positive_distance(self):
        """거리는 항상 양수"""
        dist = float(_haversine_km(37.4, 126.9, 37.6, 127.1))
        assert dist > 0


class TestHaversineKmMatrix:
    """_haversine_km_matrix 함수 테스트"""

    def test_output_shape(self):
        """M×N 형태 거리 행렬 반환"""
        apts = np.array([[37.4, 37.5, 37.6], [127.0, 127.1, 127.2]]).T
        targets = np.array([[37.5, 37.6], [127.05, 127.15]]).T
        mat = _haversine_km_matrix(apts[:, 0], apts[:, 1], targets[:, 0], targets[:, 1])
        assert mat.shape == (3, 2)

    def test_diagonal_near_zero_when_same(self):
        """같은 좌표 입력 시 거리 0"""
        pts = np.array([37.5, 37.6])
        mat = _haversine_km_matrix(pts, pts, pts, pts)
        assert mat[0, 0] < 1e-6
        assert mat[1, 1] < 1e-6

    def test_all_positive(self):
        """모든 거리 양수"""
        lats = np.random.uniform(37.0, 38.0, 5)
        lngs = np.random.uniform(126.5, 127.5, 5)
        mat = _haversine_km_matrix(lats, lngs, lats[:3], lngs[:3])
        assert np.all(mat >= 0)


class TestSpatialFeatureBuilder:
    """SpatialFeatureBuilder 테스트"""

    @pytest.fixture
    def sample_df(self):
        """테스트용 아파트 DataFrame (위경도 포함)"""
        return pd.DataFrame({
            "apt_name":       ["테스트아파트A", "테스트아파트B", "테스트아파트C"],
            "lawd_cd":        ["11680", "11650", "28185"],
            "latitude":       [37.4979, 37.5045, 37.4065],   # 강남, 반포, 인천 연수
            "longitude":      [127.0276, 127.0058, 126.6780],
            "deal_amount":    [100000, 80000, 40000],
            "area_exclusive": [84.0, 59.0, 74.0],
        })

    def test_gangnam_distance_added(self, sample_df):
        """dist_to_gangnam_km 컬럼 추가 확인"""
        builder = SpatialFeatureBuilder()
        result = builder.build_features(sample_df, use_boundaries=False)
        assert "dist_to_gangnam_km" in result.columns

    def test_gangnam_distance_near_zero_for_gangnam(self, sample_df):
        """강남역 좌표 입력 시 강남 거리 ≈ 0"""
        builder = SpatialFeatureBuilder()
        result = builder.build_features(sample_df, use_boundaries=False)
        assert result.loc[0, "dist_to_gangnam_km"] < 1.0

    def test_incheon_farther_than_gangnam(self, sample_df):
        """인천이 강남보다 강남역에서 멀어야 함"""
        builder = SpatialFeatureBuilder()
        result = builder.build_features(sample_df, use_boundaries=False)
        assert result.loc[2, "dist_to_gangnam_km"] > result.loc[0, "dist_to_gangnam_km"]

    def test_subway_distance_added(self, sample_df):
        """dist_to_subway_km 컬럼 추가 확인"""
        builder = SpatialFeatureBuilder()
        result = builder.build_features(sample_df, use_boundaries=False)
        assert "dist_to_subway_km" in result.columns
        assert (result["dist_to_subway_km"] >= 0).all()

    def test_office_distance_added(self, sample_df):
        """dist_to_cityhall_km 컬럼 추가 확인"""
        builder = SpatialFeatureBuilder()
        result = builder.build_features(sample_df, use_boundaries=False)
        assert "dist_to_cityhall_km" in result.columns
        assert (result["dist_to_cityhall_km"] >= 0).all()

    def test_original_columns_preserved(self, sample_df):
        """기존 컬럼은 변경 없이 유지"""
        builder = SpatialFeatureBuilder()
        result = builder.build_features(sample_df, use_boundaries=False)
        for col in ["apt_name", "lawd_cd", "deal_amount"]:
            pd.testing.assert_series_equal(result[col], sample_df[col])

    def test_missing_lat_lng_skips_gracefully(self):
        """위경도 없는 데이터프레임은 빈 결과 또는 원본 반환"""
        df_no_geo = pd.DataFrame({
            "apt_name":    ["A"],
            "deal_amount": [10000],
        })
        builder = SpatialFeatureBuilder()
        result = builder.build_features(df_no_geo, use_boundaries=False)
        # lat/lng 없으면 spatial 피처 컬럼이 추가되지 않아야 함
        assert "dist_to_gangnam_km" not in result.columns
