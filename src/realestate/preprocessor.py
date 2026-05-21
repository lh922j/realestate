"""
preprocessor.py - 원시 API 응답 데이터 전처리

파이프라인:
    원시 dict 리스트
    → DataFrame 생성
    → 컬럼명 변환 (API 응답 키 → 영문 스네이크케이스)
    → 타입 캐스팅 (거래금액 콤마 제거, 날짜 변환 등)
    → 결측값 처리
    → 파생 컬럼 생성 (단위면적당 가격, 건물 연령 등)
    → 중복 제거
    → 정렬된 DataFrame 반환
"""

from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger

DataType = Literal["trade", "rent"]

# ─────────────────────────────────────────────────────────────────
# 컬럼 매핑: API 응답 키 → 영문 스네이크케이스
# ─────────────────────────────────────────────────────────────────

# 아파트 매매 (getRTMSDataSvcAptTradeDev)
TRADE_COLUMN_MAP = {
    # 거래 정보
    "dealYear":     "deal_year",
    "dealMonth":    "deal_month",
    "dealDay":      "deal_day",
    "dealAmount":   "deal_amount",       # 거래금액 (만원, 콤마 포함 문자열)
    "dealingGbn":   "dealing_type",      # 거래유형 (직거래/중개)
    # 건물 정보
    "aptNm":        "apt_name",          # 아파트명
    "aptSeq":       "apt_seq",           # 아파트 일련번호 (고유 식별자)
    "buildYear":    "build_year",        # 건축년도
    "excluUseAr":   "area_exclusive",    # 전용면적 (㎡)
    "floor":        "floor",             # 층
    # 위치 정보
    "umdNm":        "dong_name",         # 법정동명
    "jibun":        "jibun",             # 지번
    "roadNm":       "road_name",         # 도로명
    "roadNmBonbun": "road_main_no",      # 도로명 건물 본번
    "roadNmBubun":  "road_sub_no",       # 도로명 건물 부번
    "sggCd":        "sgg_code",          # 시군구 코드
    "umdCd":        "umd_code",          # 읍면동 코드
    "bonbun":       "bonbun",            # 본번
    "bubun":        "bubun",             # 부번
    # 거래 취소 정보
    "rgstDate":     "register_date",     # 등기일자
    "cdealType":    "cancel_type",       # 해제여부 (Y=취소)
    "cdealDay":     "cancel_day",        # 해제사유발생일
    # 수집 메타데이터
    "_lawd_cd":     "lawd_cd",
    "_deal_ymd":    "deal_ymd",
    "_data_type":   "data_type",
}

# 아파트 전월세 (getRTMSDataSvcAptRent)
RENT_COLUMN_MAP = {
    # 거래 정보
    "dealYear":        "deal_year",
    "dealMonth":       "deal_month",
    "dealDay":         "deal_day",
    "deposit":         "deposit",           # 보증금액 (만원)
    "monthlyRent":     "monthly_rent",      # 월세금액 (만원, 0이면 전세)
    "contractType":    "contract_type",     # 계약구분 (신규/갱신)
    "contractPeriod":  "contract_period",   # 계약기간
    "preDeposit":      "pre_deposit",       # 종전 보증금 (만원)
    "preMonthlyRent":  "pre_monthly_rent",  # 종전 월세 (만원)
    # 건물 정보
    "aptNm":           "apt_name",
    "aptSeq":          "apt_seq",
    "buildYear":       "build_year",
    "excluUseAr":      "area_exclusive",
    "floor":           "floor",
    # 위치 정보
    "umdNm":           "dong_name",
    "jibun":           "jibun",
    "roadNm":          "road_name",
    "sggCd":           "sgg_code",
    "umdCd":           "umd_code",
    # 수집 메타데이터
    "_lawd_cd":        "lawd_cd",
    "_deal_ymd":       "deal_ymd",
    "_data_type":      "data_type",
}

COLUMN_MAPS = {"trade": TRADE_COLUMN_MAP, "rent": RENT_COLUMN_MAP}

# 핵심 컬럼: 결측 시 행 제거
REQUIRED_COLS = {
    "trade": ["deal_amount", "area_exclusive", "deal_date"],
    "rent":  ["deposit", "area_exclusive", "deal_date"],
}


def _clean_amount(value) -> float | None:
    """거래금액 문자열 정제: '85,000' → 85000.0"""
    if pd.isna(value) or str(value).strip() == "":
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


class Preprocessor:
    """
    원시 API 응답 데이터를 ML/분석에 바로 사용 가능한 DataFrame으로 변환

    사용 예시:
        preprocessor = Preprocessor()
        df = preprocessor.process(raw_records, data_type="trade")
        print(df.dtypes)
        print(df[["deal_date", "apt_name", "deal_amount", "price_per_sqm"]].head())
    """

    def process(
        self,
        raw_records: list[dict],
        data_type: DataType,
    ) -> pd.DataFrame:
        """
        원시 레코드 리스트 → 전처리된 DataFrame

        Args:
            raw_records: collector.collect_all()이 반환한 dict 리스트
            data_type:   "trade" (매매) 또는 "rent" (전월세)

        Returns:
            전처리된 pandas DataFrame. 빈 데이터 입력 시 빈 DataFrame 반환.
        """
        if not raw_records:
            logger.warning("입력 레코드가 없습니다.")
            return pd.DataFrame()

        logger.info(f"전처리 시작: {len(raw_records):,}건 ({data_type})")

        df = pd.DataFrame(raw_records)
        df = self._rename_columns(df, data_type)
        df = self._cast_types(df, data_type)
        df = self._handle_nulls(df, data_type)
        df = self._add_derived_columns(df, data_type)
        df = self._deduplicate(df)
        df = self._sort(df)

        logger.info(f"전처리 완료: {len(df):,}건")
        return df

    # ─────────────────────────────────────────────────────────────
    # 내부 메서드
    # ─────────────────────────────────────────────────────────────

    def _rename_columns(self, df: pd.DataFrame, data_type: DataType) -> pd.DataFrame:
        """API 응답 키를 영문 스네이크케이스로 변환 (매핑에 없는 컬럼 제거)"""
        col_map = COLUMN_MAPS[data_type]
        existing = {k: v for k, v in col_map.items() if k in df.columns}
        return df[list(existing.keys())].rename(columns=existing)

    def _cast_types(self, df: pd.DataFrame, data_type: DataType) -> pd.DataFrame:
        """문자열 → 적절한 타입으로 변환"""

        # 거래금액/보증금 (콤마 제거 후 float)
        amount_cols = (
            ["deal_amount"]
            if data_type == "trade"
            else ["deposit", "monthly_rent", "pre_deposit", "pre_monthly_rent"]
        )
        for col in amount_cols:
            if col in df.columns:
                df[col] = df[col].apply(_clean_amount)

        # 전용면적
        if "area_exclusive" in df.columns:
            df["area_exclusive"] = pd.to_numeric(
                df["area_exclusive"], errors="coerce"
            ).round(2)

        # 건축년도
        if "build_year" in df.columns:
            df["build_year"] = pd.to_numeric(
                df["build_year"], errors="coerce"
            ).astype("Int16")

        # 층 (음수=지하 허용)
        if "floor" in df.columns:
            df["floor"] = pd.to_numeric(
                df["floor"], errors="coerce"
            ).astype("Int16")

        # 거래일 조합 (deal_year + deal_month + deal_day → deal_date)
        date_cols = ["deal_year", "deal_month", "deal_day"]
        if all(c in df.columns for c in date_cols):
            df["deal_date"] = pd.to_datetime(
                df["deal_year"].astype(str).str.zfill(4) + "-"
                + df["deal_month"].astype(str).str.zfill(2) + "-"
                + df["deal_day"].astype(str).str.zfill(2),
                format="%Y-%m-%d",
                errors="coerce",
            )
            df["deal_year"]  = df["deal_year"].astype("Int16")
            df["deal_month"] = df["deal_month"].astype("Int8")
            df["deal_day"]   = df["deal_day"].astype("Int8")

        return df

    def _handle_nulls(self, df: pd.DataFrame, data_type: DataType) -> pd.DataFrame:
        """결측값 처리"""

        # 핵심 컬럼 결측 행 제거
        required = REQUIRED_COLS[data_type]
        before = len(df)
        df = df.dropna(subset=required)
        removed = before - len(df)
        if removed > 0:
            logger.warning(
                f"핵심 컬럼 결측으로 {removed:,}건 제거 ({required})"
            )

        # 아파트명 결측 → "미상"
        if "apt_name" in df.columns:
            df["apt_name"] = df["apt_name"].fillna("미상").str.strip()

        # 지번 결측 → 빈 문자열
        if "jibun" in df.columns:
            df["jibun"] = df["jibun"].fillna("").str.strip()

        # 면적 이상치 제거 (0 이하)
        if "area_exclusive" in df.columns:
            before = len(df)
            df = df[df["area_exclusive"] > 0]
            removed = before - len(df)
            if removed > 0:
                logger.warning(f"전용면적 0 이하 {removed:,}건 제거")

        return df

    def _add_derived_columns(self, df: pd.DataFrame, data_type: DataType) -> pd.DataFrame:
        """ML/분석에 유용한 파생 컬럼 추가"""

        if data_type == "trade" and "deal_amount" in df.columns:
            # 단위면적당 가격 (만원/㎡) - 정규화에 중요한 피처
            df["price_per_sqm"] = (
                df["deal_amount"] / df["area_exclusive"]
            ).round(2)
            # 억원 단위 (가독성)
            df["deal_amount_eok"] = (df["deal_amount"] / 10000).round(4)

        if data_type == "rent":
            # 전세/월세 구분 (monthly_rent가 0 또는 None이면 전세)
            if "monthly_rent" in df.columns:
                df["is_jeonse"] = df["monthly_rent"].fillna(0) == 0
            # 보증금 억원 단위
            if "deposit" in df.columns:
                df["deposit_eok"] = (df["deposit"] / 10000).round(4)

        # 건물 연령 (2025년 기준)
        if "build_year" in df.columns:
            df["building_age"] = (2025 - df["build_year"].astype("Int16")).astype("Int16")
            # 음수(미래 준공 예정) 처리
            df.loc[df["building_age"] < 0, "building_age"] = pd.NA

        return df

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """완전 중복 행 제거"""
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        if removed > 0:
            logger.info(f"중복 제거: {removed:,}건")
        return df

    def _sort(self, df: pd.DataFrame) -> pd.DataFrame:
        """날짜, 지역코드, 아파트명 순 정렬"""
        sort_cols = [c for c in ["deal_date", "lawd_cd", "apt_name"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)
        return df.reset_index(drop=True)
