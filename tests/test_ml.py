"""
test_ml.py - MLPipeline 단위 테스트

실행:
    pytest tests/test_ml.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.realestate.ml import MLPipeline, _add_time_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES


@pytest.fixture
def sample_df():
    """최소 구성의 학습용 DataFrame"""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "deal_date":       pd.date_range("2020-01-01", periods=n, freq="3D"),
        "deal_amount":     np.random.randint(20000, 150000, n).astype(float),
        "area_exclusive":  np.random.uniform(40, 130, n),
        "floor":           np.random.randint(1, 30, n).astype(float),
        "build_year":      np.random.randint(1990, 2022, n).astype(float),
        "latitude":        np.random.uniform(37.3, 37.7, n),
        "longitude":       np.random.uniform(126.8, 127.2, n),
        "dist_to_gangnam_km":  np.random.uniform(0, 40, n),
        "dist_to_subway_km":   np.random.uniform(0, 3, n),
        "dist_to_cityhall_km": np.random.uniform(0, 5, n),
        "dealing_type":    np.random.choice(["중개거래", "직거래"], n),
        "lawd_cd":         np.random.choice(["11680", "11650", "11710"], n),
    })


class TestAddTimeFeatures:
    """_add_time_features 헬퍼 함수 테스트"""

    def test_columns_added(self):
        df = pd.DataFrame({"deal_date": ["2023-03-15", "2022-12-01"]})
        result = _add_time_features(df)
        assert "deal_year" in result.columns
        assert "deal_month" in result.columns
        assert "deal_season" in result.columns

    def test_season_winter(self):
        df = pd.DataFrame({"deal_date": ["2023-01-10"]})
        result = _add_time_features(df)
        assert result["deal_season"].iloc[0] == 0  # 겨울

    def test_season_spring(self):
        df = pd.DataFrame({"deal_date": ["2023-04-01"]})
        result = _add_time_features(df)
        assert result["deal_season"].iloc[0] == 1  # 봄

    def test_no_deal_date_column(self):
        df = pd.DataFrame({"other_col": [1, 2]})
        result = _add_time_features(df)
        assert "deal_year" not in result.columns


class TestBuildFeatureMatrix:
    """MLPipeline.build_feature_matrix 테스트"""

    def test_returns_tuple(self, sample_df):
        pipeline = MLPipeline()
        sample_df["building_age"] = 2025 - sample_df["build_year"]
        sample_df["sgg_code"] = sample_df["lawd_cd"]
        X, y = pipeline.build_feature_matrix(sample_df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_X_y_same_length(self, sample_df):
        pipeline = MLPipeline()
        sample_df["building_age"] = 2025 - sample_df["build_year"]
        sample_df["sgg_code"] = sample_df["lawd_cd"]
        X, y = pipeline.build_feature_matrix(sample_df)
        assert len(X) == len(y)

    def test_y_is_log_transformed(self, sample_df):
        """y는 log1p 변환된 값이어야 함 → expm1 역변환 시 원래 금액"""
        pipeline = MLPipeline()
        sample_df["building_age"] = 2025 - sample_df["build_year"]
        sample_df["sgg_code"] = sample_df["lawd_cd"]
        X, y = pipeline.build_feature_matrix(sample_df)
        recovered = np.expm1(y)
        # 원래 deal_amount와 일치해야 함
        original = sample_df["deal_amount"].astype(float)
        np.testing.assert_array_almost_equal(recovered.values, original.values, decimal=2)

    def test_no_nan_in_X(self, sample_df):
        """결측값 처리 후 X에 NaN 없어야 함"""
        sample_df.loc[0, "area_exclusive"] = np.nan
        sample_df.loc[1, "floor"] = np.nan
        sample_df["building_age"] = 2025 - sample_df["build_year"]
        sample_df["sgg_code"] = sample_df["lawd_cd"]
        pipeline = MLPipeline()
        X, _ = pipeline.build_feature_matrix(sample_df)
        assert not X.isna().any().any()

    def test_categorical_encoded(self, sample_df):
        """범주형 컬럼이 정수로 인코딩되어야 함"""
        sample_df["building_age"] = 2025 - sample_df["build_year"]
        sample_df["sgg_code"] = sample_df["lawd_cd"]
        pipeline = MLPipeline()
        X, _ = pipeline.build_feature_matrix(sample_df)
        if "dealing_type" in X.columns:
            assert X["dealing_type"].dtype == float


class TestMLPipelineTrain:
    """MLPipeline.train / evaluate 테스트"""

    @pytest.fixture
    def trained_pipeline(self, sample_df):
        sample_df["building_age"] = 2025 - sample_df["build_year"]
        sample_df["sgg_code"] = sample_df["lawd_cd"]

        pipeline = MLPipeline(model_type="lgbm")
        X, y = pipeline.build_feature_matrix(sample_df)

        split = int(len(X) * 0.8)
        metrics = pipeline.train(X.iloc[:split], y.iloc[:split],
                                 X.iloc[split:], y.iloc[split:])
        return pipeline, metrics, X.iloc[split:], y.iloc[split:]

    def test_metrics_keys_present(self, trained_pipeline):
        _, metrics, _, _ = trained_pipeline
        for key in ("mae", "rmse", "r2", "mae_eok"):
            assert key in metrics

    def test_r2_is_float(self, trained_pipeline):
        """R²는 계산 가능한 float이어야 함 (랜덤 데이터이므로 부호 미검증)"""
        _, metrics, _, _ = trained_pipeline
        assert isinstance(metrics["r2"], float)
        assert not np.isnan(metrics["r2"])

    def test_mae_eok_reasonable(self, trained_pipeline):
        """MAE가 학습 데이터 금액 범위 내 (0~15억)"""
        _, metrics, _, _ = trained_pipeline
        assert 0 < metrics["mae_eok"] < 15

    def test_predict_returns_array(self, trained_pipeline):
        pipeline, _, X_test, _ = trained_pipeline
        preds = pipeline.predict(X_test)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(X_test)

    def test_predict_positive_prices(self, trained_pipeline):
        """예측 거래가는 양수 (만원 단위)"""
        pipeline, _, X_test, _ = trained_pipeline
        preds = pipeline.predict(X_test)
        assert np.all(preds > 0)


class TestDetectAnomalies:
    """MLPipeline.detect_anomalies 테스트"""

    @pytest.fixture
    def large_df(self):
        np.random.seed(0)
        n = 500
        df = pd.DataFrame({
            "deal_amount":         np.random.randint(20000, 100000, n).astype(float),
            "area_exclusive":      np.random.uniform(40, 130, n),
            "floor":               np.random.randint(1, 30, n).astype(float),
            "build_year":          np.random.randint(1990, 2022, n).astype(float),
            "deal_year":           np.random.randint(2020, 2025, n).astype(float),
            "deal_month":          np.random.randint(1, 13, n).astype(float),
            "deal_season":         np.random.randint(0, 4, n).astype(float),
            "latitude":            np.random.uniform(37.3, 37.7, n),
            "longitude":           np.random.uniform(126.8, 127.2, n),
            "dist_to_gangnam_km":  np.random.uniform(0, 40, n),
            "dist_to_subway_km":   np.random.uniform(0, 3, n),
            "dist_to_cityhall_km": np.random.uniform(0, 5, n),
            "dealing_type":        np.random.choice(["중개거래", "직거래"], n),
            "lawd_cd":             np.random.choice(["11680", "11650", "11710"], n),
        })
        # 고의 이상치 5건 추가
        df.loc[:4, "deal_amount"] = 1_000_000.0
        return df

    def test_columns_added(self, large_df):
        pipeline = MLPipeline()
        result = pipeline.detect_anomalies(large_df, contamination=0.02, by_district=False)
        assert "anomaly_score" in result.columns
        assert "is_anomaly" in result.columns

    def test_contamination_ratio_respected(self, large_df):
        pipeline = MLPipeline()
        contamination = 0.05
        result = pipeline.detect_anomalies(large_df, contamination=contamination, by_district=False)
        actual_ratio = result["is_anomaly"].mean()
        assert abs(actual_ratio - contamination) < 0.02  # ±2% 허용

    def test_anomaly_score_order(self, large_df):
        """이상치(is_anomaly=True)의 평균 score < 정상 거래 평균 score (낮을수록 이상)"""
        pipeline = MLPipeline()
        result = pipeline.detect_anomalies(large_df, contamination=0.02, by_district=False)
        score_anomaly = result.loc[result["is_anomaly"], "anomaly_score"].mean()
        score_normal  = result.loc[~result["is_anomaly"], "anomaly_score"].mean()
        assert score_anomaly < score_normal

    def test_price_per_sqm_added(self, large_df):
        pipeline = MLPipeline()
        result = pipeline.detect_anomalies(large_df, contamination=0.02, by_district=False)
        assert "price_per_sqm" in result.columns

    def test_by_district_mode(self, large_df):
        """구 단위 분리 탐지 시 anomaly_score 결측 없어야 함"""
        pipeline = MLPipeline()
        result = pipeline.detect_anomalies(
            large_df, contamination=0.02, by_district=True, min_samples=50
        )
        assert result["anomaly_score"].notna().all()
