"""
test_collector.py - MolitCollector 단위 테스트

실제 API 호출 없이 mock으로 테스트합니다.

실행:
    pytest tests/test_collector.py -v
"""

from unittest.mock import MagicMock, patch

import pytest

from src.realestate.collector import APIError, MolitCollector


class TestMolitCollector:
    """MolitCollector 테스트 (API mock 사용)"""

    @pytest.fixture
    def mock_api_response_trade(self):
        """정상 매매 API 응답 mock"""
        return {
            "response": {
                "header": {"resultCode": "00", "resultMsg": "NORMAL SERVICE."},
                "body": {
                    "totalCount": 2,
                    "numOfRows": 1000,
                    "pageNo": 1,
                    "items": {
                        "item": [
                            {
                                "dealYear": "2023",
                                "dealMonth": "1",
                                "dealDay": "5",
                                "dealAmount": "85,000",
                                "aptNm": "테스트아파트",
                                "aptSeq": "11680-0001",
                                "buildYear": "2010",
                                "excluUseAr": "84.99",
                                "floor": "10",
                                "umdNm": "대치동",
                                "sggCd": "11680",
                            },
                            {
                                "dealYear": "2023",
                                "dealMonth": "1",
                                "dealDay": "12",
                                "dealAmount": "90,000",
                                "aptNm": "테스트아파트",
                                "aptSeq": "11680-0001",
                                "buildYear": "2010",
                                "excluUseAr": "84.99",
                                "floor": "15",
                                "umdNm": "대치동",
                                "sggCd": "11680",
                            },
                        ]
                    },
                },
            }
        }

    @pytest.fixture
    def mock_api_response_empty(self):
        """빈 응답 mock (해당 월 거래 없음)"""
        return {
            "response": {
                "header": {"resultCode": "00", "resultMsg": "NORMAL SERVICE."},
                "body": {
                    "totalCount": 0,
                    "numOfRows": 1000,
                    "pageNo": 1,
                    "items": "",
                },
            }
        }

    @pytest.fixture
    def mock_api_response_error(self):
        """API 오류 응답 mock"""
        return {
            "response": {
                "header": {
                    "resultCode": "30",
                    "resultMsg": "SERVICE KEY IS NOT REGISTERED ERROR.",
                },
                "body": {},
            }
        }

    @patch("src.realestate.collector.API_KEY", "test_api_key")
    def test_collect_all_returns_records(
        self, mock_api_response_trade
    ):
        """정상 응답 시 레코드 반환 확인"""
        collector = MolitCollector()
        collector.session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = mock_api_response_trade
        mock_response.raise_for_status.return_value = None
        collector.session.get.return_value = mock_response

        with patch("src.realestate.collector.time.sleep"):
            records = collector.collect_all(
                data_type="trade",
                region_codes=["11680"],
                start_ym="202301",
                end_ym="202301",
            )

        assert len(records) == 2
        # 메타데이터 추가 확인
        assert records[0]["_lawd_cd"] == "11680"
        assert records[0]["_deal_ymd"] == "202301"
        assert records[0]["_data_type"] == "trade"

    @patch("src.realestate.collector.API_KEY", "test_api_key")
    def test_collect_all_empty_response(self, mock_api_response_empty):
        """빈 응답 시 빈 리스트 반환 확인"""
        collector = MolitCollector()
        collector.session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = mock_api_response_empty
        mock_response.raise_for_status.return_value = None
        collector.session.get.return_value = mock_response

        with patch("src.realestate.collector.time.sleep"):
            records = collector.collect_all(
                data_type="trade",
                region_codes=["11680"],
                start_ym="202301",
                end_ym="202301",
            )

        assert records == []

    @patch("src.realestate.collector.API_KEY", "test_api_key")
    def test_api_error_skipped_and_continues(self, mock_api_response_error):
        """API 오류 시 건너뛰고 다음 지역 계속 수집"""
        collector = MolitCollector()
        collector.session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = mock_api_response_error
        mock_response.raise_for_status.return_value = None
        collector.session.get.return_value = mock_response

        with patch("src.realestate.collector.time.sleep"):
            # 오류가 발생해도 예외 없이 빈 리스트 반환
            records = collector.collect_all(
                data_type="trade",
                region_codes=["11680"],
                start_ym="202301",
                end_ym="202301",
            )

        assert records == []

    @patch("src.realestate.collector.API_KEY", "test_api_key")
    def test_single_item_response_handled(self):
        """단건 응답 (item이 list 대신 dict인 경우) 처리 확인"""
        single_item_response = {
            "response": {
                "header": {"resultCode": "00", "resultMsg": "NORMAL SERVICE."},
                "body": {
                    "totalCount": 1,
                    "numOfRows": 1000,
                    "pageNo": 1,
                    "items": {
                        "item": {  # list가 아닌 dict
                            "dealYear": "2023",
                            "dealMonth": "1",
                            "dealDay": "5",
                            "dealAmount": "100,000",
                            "aptNm": "단건아파트",
                            "aptSeq": "11680-9999",
                            "buildYear": "2005",
                            "excluUseAr": "59.99",
                            "floor": "3",
                            "umdNm": "역삼동",
                            "sggCd": "11680",
                        }
                    },
                },
            }
        }

        collector = MolitCollector()
        collector.session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = single_item_response
        mock_response.raise_for_status.return_value = None
        collector.session.get.return_value = mock_response

        with patch("src.realestate.collector.time.sleep"):
            records = collector.collect_all(
                data_type="trade",
                region_codes=["11680"],
                start_ym="202301",
                end_ym="202301",
            )

        assert len(records) == 1
        assert records[0]["aptNm"] == "단건아파트"
