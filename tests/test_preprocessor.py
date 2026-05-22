"""
test_preprocessor.py - Preprocessor 단위 테스트

실행:
    pytest tests/test_preprocessor.py -v
"""

import pandas as pd
import pytest

from src.realestate.preprocessor import Preprocessor, _clean_amount


class TestCleanAmount:
    """_clean_amount 함수 테스트"""

    def test_comma_in_string(self):
        assert _clean_amount("85,000") == 85000.0

    def test_plain_number_string(self):
        assert _clean_amount("50000") == 50000.0

    def test_none_returns_none(self):
        assert _clean_amount(None) is None

    def test_empty_string_returns_none(self):
        assert _clean_amount("") is None

    def test_nan_returns_none(self):
        assert _clean_amount(float("nan")) is None

    def test_invalid_string_returns_none(self):
        assert _clean_amount("미상") is None


class TestPreprocessorTrade:
    """매매 데이터 전처리 테스트"""

    @pytest.fixture
    def sample_trade_records(self):
        return [
            {
                "dealYear":   "2023",
                "dealMonth":  "1",
                "dealDay":    "15",
                "dealAmount": "85,000",
                "dealingGbn": "중개거래",
                "aptNm":      "래미안대치팰리스",
                "aptSeq":     "11680-1234",
                "buildYear":  "2015",
                "excluUseAr": "84.99",
                "floor":      "10",
                "umdNm":      "대치동",
                "jibun":      "316",
                "roadNm":     "선릉로",
                "roadNmBonbun": "100",
                "roadNmBubun":  "0",
                "sggCd":      "11680",
                "umdCd":      "10300",
                "bonbun":     "316",
                "bubun":      "0",
                "rgstDate":   "20230120",
                "cdealType":  "",
                "cdealDay":   "",
                "_lawd_cd":   "11680",
                "_deal_ymd":  "202301",
                "_data_type": "trade",
            }
        ]

    def test_basic_processing(self, sample_trade_records):
        preprocessor = Preprocessor()
        df = preprocessor.process(sample_trade_records, data_type="trade")

        assert not df.empty
        assert len(df) == 1

    def test_deal_amount_cleaned(self, sample_trade_records):
        preprocessor = Preprocessor()
        df = preprocessor.process(sample_trade_records, data_type="trade")

        assert df["deal_amount"].iloc[0] == 85000.0

    def test_deal_date_created(self, sample_trade_records):
        preprocessor = Preprocessor()
        df = preprocessor.process(sample_trade_records, data_type="trade")

        assert "deal_date" in df.columns
        assert str(df["deal_date"].iloc[0])[:10] == "2023-01-15"

    def test_derived_columns_added(self, sample_trade_records):
        preprocessor = Preprocessor()
        df = preprocessor.process(sample_trade_records, data_type="trade")

        assert "price_per_sqm" in df.columns
        assert "deal_amount_eok" in df.columns
        assert "building_age" in df.columns

        # 85000 / 84.99 ≈ 1000.12
        assert abs(df["price_per_sqm"].iloc[0] - 85000 / 84.99) < 1
        # 85000 만원 = 8.5 억원
        assert abs(df["deal_amount_eok"].iloc[0] - 8.5) < 0.01
        # 2025 - 2015 = 10
        assert df["building_age"].iloc[0] == 10

    def test_empty_records_returns_empty_df(self):
        preprocessor = Preprocessor()
        df = preprocessor.process([], data_type="trade")

        assert df.empty

    def test_missing_deal_amount_row_removed(self, sample_trade_records):
        records = sample_trade_records.copy()
        records[0]["dealAmount"] = ""  # 결측

        preprocessor = Preprocessor()
        df = preprocessor.process(records, data_type="trade")

        assert df.empty


class TestPreprocessorRent:
    """전월세 데이터 전처리 테스트"""

    @pytest.fixture
    def sample_rent_records(self):
        return [
            # 전세 레코드
            {
                "dealYear":       "2023",
                "dealMonth":      "3",
                "dealDay":        "10",
                "deposit":        "60,000",
                "monthlyRent":    "0",
                "contractType":   "신규",
                "contractPeriod": "24",
                "preDeposit":     "",
                "preMonthlyRent": "",
                "aptNm":          "아크로리버파크",
                "aptSeq":         "11650-5678",
                "buildYear":      "2016",
                "excluUseAr":     "59.98",
                "floor":          "5",
                "umdNm":          "반포동",
                "jibun":          "1-1",
                "roadNm":         "신반포로",
                "sggCd":          "11650",
                "umdCd":          "10500",
                "_lawd_cd":       "11650",
                "_deal_ymd":      "202303",
                "_data_type":     "rent",
            },
            # 월세 레코드
            {
                "dealYear":       "2023",
                "dealMonth":      "3",
                "dealDay":        "15",
                "deposit":        "5,000",
                "monthlyRent":    "200",
                "contractType":   "신규",
                "contractPeriod": "24",
                "preDeposit":     "",
                "preMonthlyRent": "",
                "aptNm":          "아크로리버파크",
                "aptSeq":         "11650-5678",
                "buildYear":      "2016",
                "excluUseAr":     "59.98",
                "floor":          "8",
                "umdNm":          "반포동",
                "jibun":          "1-1",
                "roadNm":         "신반포로",
                "sggCd":          "11650",
                "umdCd":          "10500",
                "_lawd_cd":       "11650",
                "_deal_ymd":      "202303",
                "_data_type":     "rent",
            },
        ]

    def test_jeonse_flag_set_correctly(self, sample_rent_records):
        preprocessor = Preprocessor()
        df = preprocessor.process(sample_rent_records, data_type="rent")

        assert len(df) == 2
        # 첫 번째: 전세 (monthly_rent=0)
        jeonse_row = df[df["monthly_rent"] == 0.0]
        assert jeonse_row["is_jeonse"].iloc[0] is True or jeonse_row["is_jeonse"].iloc[0] == True

        # 두 번째: 월세 (monthly_rent=200)
        wolse_row = df[df["monthly_rent"] == 200.0]
        assert wolse_row["is_jeonse"].iloc[0] is False or wolse_row["is_jeonse"].iloc[0] == False

    def test_deposit_cleaned(self, sample_rent_records):
        preprocessor = Preprocessor()
        df = preprocessor.process(sample_rent_records, data_type="rent")

        jeonse_row = df[df["monthly_rent"] == 0.0]
        assert jeonse_row["deposit"].iloc[0] == 60000.0
