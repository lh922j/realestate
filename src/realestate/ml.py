"""
ml.py - 아파트 매매가 예측 ML 파이프라인

지오코딩으로 확보한 위경도 좌표와 MOLIT 실거래가 데이터를 결합하여
아파트 매매가를 예측합니다.

학습 방법:
  train()         - LightGBM / XGBoost / Ridge (기본)
  train_optuna()  - Optuna 하이퍼파라미터 최적화 후 LightGBM/XGBoost
  train_pycaret() - PyCaret AutoML (여러 모델 자동 비교)

학습 목표: deal_amount (만원)
  train / train_optuna: log1p 변환 후 학습, expm1로 역변환
  train_pycaret: PyCaret이 내부적으로 처리
시계열 분리: deal_date 기준 (랜덤 분리 시 데이터 누수 발생)
"""

from datetime import datetime
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

from .config import MODEL_DIR

# ─────────────────────────────────────────────────────────────────
# 피처 정의
# ─────────────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "area_exclusive",         # 전용면적 (㎡)
    "floor",                  # 층
    "building_age",           # 건물 나이 (2025 - 건축년도)
    "deal_year",
    "deal_month",
    "deal_season",            # 0=겨울 1=봄 2=여름 3=가을
    "latitude",
    "longitude",
    "dist_to_gangnam_km",     # 강남역까지 직선 거리
    "dist_to_subway_km",      # 가장 가까운 지하철역까지 거리
    "dist_to_cityhall_km",    # 가장 가까운 시청/구청까지 거리
]

CATEGORICAL_FEATURES = [
    "dealing_type",   # 거래유형 (중개/직거래)
    "sgg_code",       # 시군구 코드
]

TARGET = "deal_amount"   # 만원 단위


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """deal_date로부터 deal_year, deal_month, deal_season 생성"""
    df = df.copy()
    if "deal_date" in df.columns:
        dt = pd.to_datetime(df["deal_date"], errors="coerce")
        df["deal_year"]  = dt.dt.year.astype("Int16")
        df["deal_month"] = dt.dt.month.astype("Int8")
        month = dt.dt.month
        df["deal_season"] = (
            month.map(lambda m: 0 if m in (12, 1, 2) else
                                1 if m in (3, 4, 5) else
                                2 if m in (6, 7, 8) else 3)
            .astype("Int8")
        )
    return df


class MLPipeline:
    """
    아파트 가격 예측 파이프라인

    사용 예시:
        pipeline = MLPipeline(model_type="lgbm")
        X_train, y_train = pipeline.build_feature_matrix(train_df)
        X_test,  y_test  = pipeline.build_feature_matrix(test_df)
        pipeline.train(X_train, y_train, X_test, y_test)

        metrics = pipeline.evaluate(X_test, y_test)
        print(f"MAE: {metrics['mae_eok']:.2f} 억원")

        pipeline.save("price_model_lgbm")
    """

    def __init__(
        self,
        model_type: Literal["lgbm", "xgboost", "ridge"] = "lgbm",
        model_dir: Path = MODEL_DIR,
    ):
        self.model_type    = model_type
        self.model_dir     = model_dir
        self.model         = None
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.feature_names_: list[str] = []
        self.best_params_: dict = {}   # Optuna가 찾은 최적 파라미터

    # ─────────────────────────────────────────────────────────────
    # 피처 행렬 생성
    # ─────────────────────────────────────────────────────────────

    def build_feature_matrix(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        DataFrame → (X 피처 행렬, y 타깃 벡터)

        Args:
            df: deal_amount, 좌표, 거래 정보 포함 DataFrame

        Returns:
            (X, y_log) — y는 log1p 변환된 값
        """
        df = _add_time_features(df)

        avail_num = [c for c in NUMERIC_FEATURES if c in df.columns]
        avail_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]

        missing = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) - set(df.columns)
        if missing:
            logger.warning(f"피처 누락 (사용 가능한 컬럼만 사용): {sorted(missing)}")

        X = df[avail_num + avail_cat].copy()

        # 수치형: 결측값 중앙값으로 대체
        for col in avail_num:
            if X[col].isna().any():
                median = X[col].median()
                X[col] = X[col].fillna(median)

        # 범주형: LabelEncoder (학습/예측 일관성 유지)
        for col in avail_cat:
            X[col] = X[col].fillna("unknown").astype(str)
            if col not in self.label_encoders:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                # 학습 때 못 본 값은 'unknown'으로 처리
                known = set(le.classes_)
                X[col] = X[col].apply(lambda v: v if v in known else "unknown")
                X[col] = le.transform(X[col])

        # 전체 float64로 통일
        X = X.astype(float)

        self.feature_names_ = list(X.columns)

        # 타깃: log1p 변환 (우편향 분포 정규화)
        y = np.log1p(df[TARGET].astype(float))

        return X, y

    # ─────────────────────────────────────────────────────────────
    # 모델 빌드
    # ─────────────────────────────────────────────────────────────

    def _build_model(self):
        if self.model_type == "lgbm":
            try:
                import lightgbm as lgb
            except ImportError:
                raise ImportError("pip install lightgbm")
            return lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=127,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )

        elif self.model_type == "xgboost":
            try:
                import xgboost as xgb
            except ImportError:
                raise ImportError("pip install xgboost")
            return xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )

        elif self.model_type == "rf":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_leaf=10,
                max_features=0.7,
                n_jobs=-1,
                random_state=42,
            )

        elif self.model_type == "ridge":
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline as SkPipeline
            return SkPipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ])

        else:
            raise ValueError(f"지원하지 않는 모델: {self.model_type}")

    # ─────────────────────────────────────────────────────────────
    # 학습 / 평가
    # ─────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict:
        """
        모델 학습 및 즉시 평가

        Args:
            X_train, y_train: 학습 데이터 (deal_date < cutoff)
            X_test,  y_test:  테스트 데이터 (deal_date >= cutoff)

        Returns:
            평가 지표 dict
        """
        self.model = self._build_model()

        logger.info(
            f"모델 학습 시작: {self.model_type} | "
            f"train={len(X_train):,} | test={len(X_test):,}"
        )
        self.model.fit(X_train, y_train)
        logger.info("모델 학습 완료")

        self.metrics_ = self.evaluate(X_test, y_test)
        return self.metrics_

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, float]:
        """
        테스트셋 평가

        Returns:
            mae (만원), rmse (만원), r2, mae_eok (억원)
        """
        if self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다. train()을 먼저 호출하세요.")

        y_pred_log = self.model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(np.array(y_test))

        mae  = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2   = float(r2_score(y_true, y_pred))

        return {
            "mae":     mae,
            "rmse":    rmse,
            "r2":      r2,
            "mae_eok": mae / 10000,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        예측 (만원 단위 반환)

        Returns:
            예측 거래금액 배열 (만원)
        """
        if self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다.")
        if self.model_type == "pycaret":
            from pycaret.regression import predict_model
            return predict_model(self.model, data=X)["prediction_label"].to_numpy()
        return np.expm1(self.model.predict(X))

    # ─────────────────────────────────────────────────────────────
    # 모델 비교 (PyCaret 대체)
    # ─────────────────────────────────────────────────────────────

    def compare_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """
        여러 모델을 학습·평가하여 리더보드 반환 (PyCaret compare_models 대체)

        Returns:
            MAE 오름차순 정렬된 리더보드 DataFrame
        """
        import lightgbm as lgb
        import xgboost as xgb
        from sklearn.ensemble import (
            RandomForestRegressor,
            ExtraTreesRegressor,
            GradientBoostingRegressor,
        )
        from sklearn.linear_model import Ridge, ElasticNet
        from sklearn.pipeline import Pipeline as SkPipeline
        from sklearn.preprocessing import StandardScaler

        candidates = {
            "LightGBM": lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.05, num_leaves=127,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                n_jobs=-1, verbose=-1,
            ),
            "XGBoost": xgb.XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                n_jobs=-1, verbosity=0,
            ),
            "RandomForest": RandomForestRegressor(
                n_estimators=300, max_features="sqrt",
                random_state=42, n_jobs=-1,
            ),
            "ExtraTrees": ExtraTreesRegressor(
                n_estimators=300, max_features="sqrt",
                random_state=42, n_jobs=-1,
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=5,
                subsample=0.8, random_state=42,
            ),
            "Ridge": SkPipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ]),
            "ElasticNet": SkPipeline([
                ("scaler", StandardScaler()),
                ("net", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000)),
            ]),
        }

        rows = []
        for model_name, model in candidates.items():
            logger.info(f"학습 중: {model_name}")
            model.fit(X_train, y_train)
            y_pred = np.expm1(model.predict(X_test))
            y_true = np.expm1(np.array(y_test))
            mae  = float(mean_absolute_error(y_true, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            r2   = float(r2_score(y_true, y_pred))
            rows.append({
                "Model":      model_name,
                "MAE(만원)":  round(mae),
                "MAE(억원)":  round(mae / 10000, 2),
                "RMSE(만원)": round(rmse),
                "R²":         round(r2, 4),
            })
            # 1위 모델을 self.model로 저장 (리더보드 정렬 후 교체)
        leaderboard = pd.DataFrame(rows).sort_values("MAE(만원)").reset_index(drop=True)

        # 가장 좋은 모델을 self.model로 저장
        best_name = leaderboard.iloc[0]["Model"]
        self.model = candidates[best_name]
        self.model_type = best_name.lower().replace(" ", "_")
        logger.info(f"최적 모델: {best_name}")

        return leaderboard

    # ─────────────────────────────────────────────────────────────
    # 저장 / 로드
    # ─────────────────────────────────────────────────────────────

    def save(self, name: str = "price_model") -> Path:
        """
        모델을 joblib으로 저장

        Returns:
            저장된 파일 경로
        """
        if self.model is None:
            raise RuntimeError("저장할 모델이 없습니다.")
        path = self.model_dir / f"{name}.pkl"
        joblib.dump(
            {
                "model":          self.model,
                "model_type":     self.model_type,
                "feature_names":  self.feature_names_,
                "label_encoders": self.label_encoders,
                "trained_at":     datetime.now().isoformat(),
                "metrics":        getattr(self, "metrics_", {}),
                "split":          getattr(self, "split_", "time"),
            },
            path,
        )
        logger.info(f"모델 저장: {path}")
        return path

    def load(self, name: str = "price_model") -> "MLPipeline":
        """
        저장된 모델 로드

        Returns:
            self (체이닝용)
        """
        path = self.model_dir / f"{name}.pkl"
        obj = joblib.load(path)
        self.model          = obj["model"]
        self.model_type     = obj["model_type"]
        self.feature_names_ = obj["feature_names"]
        self.label_encoders = obj.get("label_encoders", {})
        logger.info(f"모델 로드: {path} (학습일: {obj.get('trained_at', 'N/A')})")
        return self

    # ─────────────────────────────────────────────────────────────
    # Optuna 하이퍼파라미터 최적화
    # ─────────────────────────────────────────────────────────────

    def train_optuna(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_trials: int = 50,
    ) -> dict:
        """
        Optuna로 하이퍼파라미터를 탐색한 뒤 최적 파라미터로 최종 학습

        탐색 방법:
          - X_train의 마지막 20%를 시간 기준 검증셋으로 사용 (시계열 일관성 유지)
          - 검증셋 MAE(log 공간)를 최소화하는 방향으로 n_trials회 탐색
          - 최적 파라미터 확정 후 X_train 전체로 최종 학습

        lgbm / xgboost 만 지원 (ridge는 탐색 파라미터가 없음)

        Args:
            X_train, y_train: 학습 데이터 (log1p 변환된 y)
            X_test,  y_test:  테스트 데이터
            n_trials:         Optuna 시도 횟수 (많을수록 좋지만 시간 ↑)

        Returns:
            평가 지표 dict (mae, rmse, r2, mae_eok)
        """
        try:
            import optuna
        except ImportError:
            raise ImportError("pip install optuna")

        if self.model_type not in ("lgbm", "xgboost"):
            raise ValueError(
                f"Optuna는 lgbm/xgboost만 지원합니다. 현재: {self.model_type}"
            )

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # 시간 순서 기준 검증셋 분리 (마지막 20%)
        val_size  = max(1, int(len(X_train) * 0.2))
        X_tr      = X_train.iloc[:-val_size]
        y_tr      = y_train.iloc[:-val_size]
        X_val     = X_train.iloc[-val_size:]
        y_val     = y_train.iloc[-val_size:]

        def objective(trial: "optuna.Trial") -> float:
            if self.model_type == "lgbm":
                import lightgbm as lgb
                params = {
                    "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
                    "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "num_leaves":        trial.suggest_int("num_leaves", 31, 255),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                    "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                    "random_state": 42, "n_jobs": -1, "verbose": -1,
                }
                model = lgb.LGBMRegressor(**params)
            else:
                import xgboost as xgb
                params = {
                    "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
                    "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "max_depth":        trial.suggest_int("max_depth", 3, 10),
                    "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                    "random_state": 42, "n_jobs": -1, "verbosity": 0,
                }
                model = xgb.XGBRegressor(**params)

            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            return float(mean_absolute_error(y_val, pred))  # log 공간 MAE

        logger.info(f"Optuna 탐색 시작: {n_trials}회 ({self.model_type})")
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.best_params_ = study.best_params
        logger.info(f"최적 파라미터: {self.best_params_}")
        logger.info(f"검증셋 MAE (log): {study.best_value:.6f}")

        # 최적 파라미터로 전체 학습 데이터 재학습
        if self.model_type == "lgbm":
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(
                **self.best_params_, random_state=42, n_jobs=-1, verbose=-1
            )
        else:
            import xgboost as xgb
            self.model = xgb.XGBRegressor(
                **self.best_params_, random_state=42, n_jobs=-1, verbosity=0
            )

        self.model.fit(X_train, y_train)
        logger.info("최적 파라미터로 최종 학습 완료")

        return self.evaluate(X_test, y_test)

    # ─────────────────────────────────────────────────────────────
    # PyCaret AutoML
    # ─────────────────────────────────────────────────────────────

    def train_pycaret(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        n_select: int = 3,
    ) -> dict:
        """
        PyCaret AutoML로 여러 모델을 자동 비교하고 최적 모델 선택

        기존 build_feature_matrix()를 거치지 않고 DataFrame 그대로 전달합니다.
        PyCaret이 인코딩 / 결측값 처리 / 정규화를 내부적으로 수행합니다.

        Args:
            train_df:  TARGET 컬럼 포함 학습용 DataFrame (deal_date < cutoff)
            test_df:   TARGET 컬럼 포함 테스트용 DataFrame (deal_date >= cutoff)
            n_select:  비교 후 최종 앙상블에 사용할 상위 모델 수

        Returns:
            평가 지표 dict + "leaderboard" 키로 모델 비교 결과 DataFrame 포함
        """
        try:
            from pycaret.regression import (
                setup, compare_models, blend_models,
                predict_model, pull,
            )
        except ImportError:
            raise ImportError("pip install pycaret")

        avail_num = [c for c in NUMERIC_FEATURES if c in train_df.columns]
        avail_cat = [c for c in CATEGORICAL_FEATURES if c in train_df.columns]
        feat_cols = avail_num + avail_cat

        # TARGET이 없는 컬럼은 제외
        keep_cols = feat_cols + [TARGET]
        train_pc  = train_df[[c for c in keep_cols if c in train_df.columns]].copy()
        test_pc   = test_df[[c  for c in keep_cols if c in test_df.columns]].copy()

        # 결측값 수치형 중앙값 / 범주형 'unknown' 으로 채움 (PyCaret setup 전 처리)
        for col in avail_num:
            if col in train_pc.columns:
                median = train_pc[col].median()
                train_pc[col] = train_pc[col].fillna(median)
                test_pc[col]  = test_pc[col].fillna(median)
        for col in avail_cat:
            if col in train_pc.columns:
                train_pc[col] = train_pc[col].fillna("unknown").astype(str)
                test_pc[col]  = test_pc[col].fillna("unknown").astype(str)

        logger.info(
            f"PyCaret setup: train={len(train_pc):,} | test={len(test_pc):,} | "
            f"features={len(feat_cols)}"
        )

        setup(
            data=train_pc,
            test_data=test_pc,          # 명시적 테스트셋 (시계열 누수 방지)
            target=TARGET,
            categorical_features=[c for c in avail_cat if c in train_pc.columns],
            numeric_features=[c for c in avail_num if c in train_pc.columns],
            session_id=42,
            verbose=False,
            html=False,
            log_experiment=False,
        )

        logger.info("PyCaret 모델 비교 중...")
        top_models = compare_models(n_select=n_select, sort="MAE", verbose=False)
        leaderboard = pull()   # compare_models 직후 결과 DataFrame

        # 상위 모델 블렌딩 앙상블
        if isinstance(top_models, list) and len(top_models) > 1:
            logger.info(f"상위 {len(top_models)}개 모델 블렌딩 앙상블")
            best = blend_models(top_models, verbose=False)
        else:
            best = top_models if not isinstance(top_models, list) else top_models[0]

        # 테스트셋 예측 및 지표 계산
        pred_df  = predict_model(best, data=test_pc.drop(columns=[TARGET]))
        y_pred   = pred_df["prediction_label"].to_numpy()
        y_true   = test_pc[TARGET].to_numpy()

        mae  = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2   = float(r2_score(y_true, y_pred))

        # PyCaret 모델을 self.model에 저장 (save()/predict() 호환)
        self.model      = best
        self.model_type = "pycaret"
        self.feature_names_ = feat_cols

        logger.info(f"PyCaret 완료 | MAE={mae:,.0f}만원 | R²={r2:.4f}")

        return {
            "mae":         mae,
            "rmse":        rmse,
            "r2":          r2,
            "mae_eok":     mae / 10000,
            "leaderboard": leaderboard,
        }

    # ─────────────────────────────────────────────────────────────
    # 분석
    # ─────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────
    # 이상 탐지 (Isolation Forest)
    # ─────────────────────────────────────────────────────────────

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        contamination: float = 0.02,
        n_estimators: int = 200,
        by_district: bool = True,
        min_samples: int = 50,
    ) -> pd.DataFrame:
        """
        Isolation Forest로 이상 거래 탐지

        Args:
            df:            deal_amount + 공간/시간 피처 포함 DataFrame
            contamination: 이상치 비율 (기본 2%)
            n_estimators:  트리 수
            by_district:   True면 lawd_cd 구 단위로 분리 탐지 (권장)
                           False면 전체 데이터 한 번에 탐지
            min_samples:   by_district 시 최소 샘플 수 (미달 구는 전체 탐지로 대체)

        Returns:
            anomaly_score, is_anomaly, price_per_sqm 컬럼이 추가된 DataFrame
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        df = df.copy()

        if "area_exclusive" in df.columns and "deal_amount" in df.columns:
            df["price_per_sqm"] = (
                df["deal_amount"] / df["area_exclusive"].replace(0, np.nan)
            ).round(1)
        else:
            df["price_per_sqm"] = np.nan

        anomaly_features = NUMERIC_FEATURES + ["price_per_sqm"]
        if "deal_amount" in df.columns:
            anomaly_features = ["deal_amount"] + anomaly_features
        # 위도·경도는 같은 구 안에서는 변별력이 없으므로 구 단위 탐지 시 제외
        district_drop = {"latitude", "longitude", "dist_to_gangnam_km",
                         "dist_to_subway_km", "dist_to_cityhall_km"}

        def _fit_predict(subset: pd.DataFrame, drop_geo: bool) -> tuple[np.ndarray, np.ndarray]:
            feats = [
                c for c in anomaly_features
                if c in subset.columns and not (drop_geo and c in district_drop)
            ]
            X = subset[feats].copy()
            for col in feats:
                if X[col].isna().any():
                    X[col] = X[col].fillna(X[col].median())
            X_scaled = StandardScaler().fit_transform(X)
            clf = IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=42,
                n_jobs=-1,
            )
            clf.fit(X_scaled)
            return clf.score_samples(X_scaled), clf.predict(X_scaled)

        df["anomaly_score"] = np.nan
        df["is_anomaly"]    = False

        if by_district and "lawd_cd" in df.columns:
            groups = df.groupby("lawd_cd")
            for lawd_cd, idx in groups.groups.items():
                group = df.loc[idx]
                drop_geo = len(group) >= min_samples
                scores, preds = _fit_predict(group, drop_geo=drop_geo)
                df.loc[idx, "anomaly_score"] = scores.round(5)
                df.loc[idx, "is_anomaly"]    = preds == -1
            logger.info(
                f"구 단위 이상 탐지 완료 | {len(groups)}개 구 | "
                f"이상치 {df['is_anomaly'].sum():,}건 ({df['is_anomaly'].mean()*100:.1f}%)"
            )
        else:
            scores, preds = _fit_predict(df, drop_geo=False)
            df["anomaly_score"] = scores.round(5)
            df["is_anomaly"]    = preds == -1
            logger.info(
                f"전체 이상 탐지 완료 | {len(df):,}건 중 "
                f"이상치 {df['is_anomaly'].sum():,}건 ({df['is_anomaly'].mean()*100:.1f}%)"
            )

        return df

    def shap_analysis(
        self,
        X: pd.DataFrame,
        max_samples: int = 2000,
    ) -> dict:
        """
        SHAP TreeExplainer로 피처 영향도 분석

        Args:
            X:           build_feature_matrix()로 생성한 피처 행렬
            max_samples: SHAP 계산에 사용할 최대 샘플 수

        Returns:
            {
                "shap_values": np.ndarray (n_samples × n_features, log1p 공간),
                "X_sample":    pd.DataFrame,
                "base_value":  float (log1p 공간 기준값),
                "feature_names": list[str],
            }
        """
        try:
            import shap
        except ImportError:
            raise ImportError("pip install shap")

        if self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다. train()을 먼저 호출하세요.")

        X_sample = X.sample(min(max_samples, len(X)), random_state=42)
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)

        return {
            "shap_values":   np.array(shap_values),
            "X_sample":      X_sample,
            "base_value":    float(explainer.expected_value),
            "feature_names": list(X_sample.columns),
        }

    def feature_importance(self) -> pd.DataFrame:
        """
        피처 중요도 DataFrame 반환 (importance 내림차순)

        Returns:
            feature, importance 컬럼을 가진 DataFrame
        """
        if self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다.")

        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "named_steps"):
            # sklearn Pipeline (Ridge)
            step = self.model.named_steps.get("ridge")
            if step and hasattr(step, "coef_"):
                importance = np.abs(step.coef_)
            else:
                return pd.DataFrame(columns=["feature", "importance"])
        else:
            return pd.DataFrame(columns=["feature", "importance"])

        return (
            pd.DataFrame({"feature": self.feature_names_, "importance": importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
