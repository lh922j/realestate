"""
collector.py - 국토교통부 실거래가 API 수집기

아파트 매매 / 전월세 데이터를 수집합니다.
- Rate limiting: 요청 간 delay로 API 한도 준수
- Pagination: totalCount 기반 전체 페이지 자동 수집
- Retry: tenacity로 일시적 네트워크 오류 자동 재시도
- 에러 처리: API 오류코드, 단건 응답(dict vs list) 처리
"""

import time
from typing import Literal

import requests
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
from tqdm import tqdm

from .config import (
    API_KEY,
    BASE_URLS,
    DEFAULT_END_YM,
    DEFAULT_START_YM,
    MAX_RETRIES,
    PAGE_SIZE,
    REGION_CODES,
    REQUEST_DELAY_SECONDS,
    RETRY_WAIT_SECONDS,
    generate_year_months,
)

DataType = Literal["trade", "rent"]


class APIError(Exception):
    """공공데이터포털 API 오류 응답 예외"""
    pass


class MolitCollector:
    """
    국토교통부 실거래가 API 수집기

    사용 예시:
        collector = MolitCollector()

        # 강남구 2023년 매매 데이터 수집
        records = collector.collect_all(
            data_type="trade",
            region_codes=["11680"],
            start_ym="202301",
            end_ym="202312",
        )
        print(f"수집 건수: {len(records)}")
    """

    def __init__(self):
        self.session = requests.Session()
        # 공통 파라미터 (type=json 제외 — 매매 엔드포인트는 XML만 지원)
        self.session.params = {
            "serviceKey": API_KEY,
            "numOfRows":  PAGE_SIZE,
        }

    # ─────────────────────────────────────────────────────────────
    # 단일 페이지 요청 (retry 포함)
    # ─────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=RETRY_WAIT_SECONDS, min=2, max=30),
        retry=retry_if_exception_type((requests.RequestException, APIError)),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )
    def _fetch_page(
        self,
        data_type: DataType,
        lawd_cd: str,
        deal_ymd: str,
        page_no: int = 1,
    ) -> dict:
        """
        단일 API 페이지 요청

        Args:
            data_type: "trade" (매매) 또는 "rent" (전월세)
            lawd_cd:   5자리 지역코드 (예: "11680" = 강남구)
            deal_ymd:  거래 연월 (YYYYMM, 예: "202301")
            page_no:   페이지 번호 (1부터 시작)

        Returns:
            API 응답 JSON dict

        Raises:
            APIError: API 오류 코드 응답 시
            requests.RequestException: 네트워크 오류 시 (tenacity가 재시도)
        """
        import xml.etree.ElementTree as ET

        url = BASE_URLS[data_type]
        params = {
            "LAWD_CD":  lawd_cd,
            "DEAL_YMD": deal_ymd,
            "pageNo":   page_no,
        }

        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()

        # 응답이 JSON인지 XML인지 감지
        content_type = response.headers.get("Content-Type", "")
        if "json" in content_type:
            data = response.json()
            header = data.get("response", {}).get("header", {})
            result_code = header.get("resultCode", "")
            if result_code not in ("00", "000"):
                raise APIError(
                    f"API 오류 [{result_code}]: {header.get('resultMsg', '')} "
                    f"(lawd_cd={lawd_cd}, deal_ymd={deal_ymd})"
                )
            return data

        # XML 응답 파싱 (매매 기본 엔드포인트)
        root = ET.fromstring(response.content)
        result_code = (root.findtext(".//resultCode") or "").strip()
        if result_code not in ("00", "000", "0000"):
            result_msg = root.findtext(".//resultMsg") or ""
            raise APIError(
                f"API 오류 [{result_code}]: {result_msg} "
                f"(lawd_cd={lawd_cd}, deal_ymd={deal_ymd}, page={page_no})"
            )

        total_count = int(root.findtext(".//totalCount") or 0)
        items = [
            {child.tag: (child.text or "").strip() for child in item}
            for item in root.findall(".//item")
        ]

        # JSON 응답과 같은 구조로 변환해서 반환
        return {
            "response": {
                "header": {"resultCode": result_code},
                "body": {
                    "totalCount": total_count,
                    "items": {"item": items},
                },
            }
        }

    # ─────────────────────────────────────────────────────────────
    # 특정 지역 + 월 전체 페이지 수집
    # ─────────────────────────────────────────────────────────────

    def _fetch_all_pages(
        self,
        data_type: DataType,
        lawd_cd: str,
        deal_ymd: str,
    ) -> list[dict]:
        """
        특정 지역코드 + 거래월의 모든 페이지 수집

        Args:
            data_type: "trade" 또는 "rent"
            lawd_cd:   5자리 지역코드
            deal_ymd:  거래 연월 (YYYYMM)

        Returns:
            해당 지역/월의 모든 거래 레코드 리스트
        """
        records = []
        page_no = 1

        while True:
            time.sleep(REQUEST_DELAY_SECONDS)

            data = self._fetch_page(data_type, lawd_cd, deal_ymd, page_no)
            body = data.get("response", {}).get("body", {})
            total_count = int(body.get("totalCount", 0))

            if total_count == 0:
                logger.debug(f"데이터 없음: lawd_cd={lawd_cd}, deal_ymd={deal_ymd}")
                break

            items = body.get("items", {})
            # totalCount > 0 인데 items가 비어있는 경우 (API 응답 이상)
            if not items or items == "":
                logger.warning(
                    f"items 비어있음 (totalCount={total_count}): "
                    f"lawd_cd={lawd_cd}, deal_ymd={deal_ymd}"
                )
                break

            item_list = items.get("item", [])
            # 단건 응답일 때 item이 list가 아닌 dict로 오는 경우 처리
            if isinstance(item_list, dict):
                item_list = [item_list]

            records.extend(item_list)

            fetched = page_no * PAGE_SIZE
            if fetched >= total_count:
                break

            page_no += 1

        return records

    # ─────────────────────────────────────────────────────────────
    # 공개 수집 인터페이스
    # ─────────────────────────────────────────────────────────────

    def collect_all(
        self,
        data_type: DataType,
        region_codes: list[str] | None = None,
        start_ym: str = DEFAULT_START_YM,
        end_ym: str = DEFAULT_END_YM,
    ) -> list[dict]:
        """
        여러 지역 × 여러 월 데이터 일괄 수집

        Args:
            data_type:    "trade" (매매) 또는 "rent" (전월세)
            region_codes: 5자리 지역코드 리스트. None이면 config의 REGION_CODES 전체
            start_ym:     시작 연월 (YYYYMM, 예: "202301")
            end_ym:       종료 연월 (YYYYMM, 예: "202312")

        Returns:
            수집된 레코드 딕셔너리 리스트.
            각 레코드에 수집 메타데이터(_lawd_cd, _deal_ymd, _data_type)가 추가됨.

        Example:
            collector = MolitCollector()
            records = collector.collect_all(
                data_type="trade",
                region_codes=["11680", "11710"],  # 강남구, 송파구
                start_ym="202301",
                end_ym="202312",
            )
        """
        if region_codes is None:
            region_codes = list(REGION_CODES.values())

        year_months = generate_year_months(start_ym, end_ym)
        all_records = []

        total_jobs = len(region_codes) * len(year_months)
        logger.info(
            f"수집 시작 | type={data_type} | "
            f"지역={len(region_codes)}개 | "
            f"기간={start_ym}~{end_ym} ({len(year_months)}개월) | "
            f"총 요청={total_jobs:,}회"
        )

        with tqdm(total=total_jobs, desc=f"{data_type} 수집", unit="req") as pbar:
            for lawd_cd in region_codes:
                for deal_ymd in year_months:
                    try:
                        records = self._fetch_all_pages(data_type, lawd_cd, deal_ymd)

                        # 수집 메타데이터 추가 (추적 및 디버깅용)
                        for r in records:
                            r["_lawd_cd"]   = lawd_cd
                            r["_deal_ymd"]  = deal_ymd
                            r["_data_type"] = data_type

                        all_records.extend(records)

                        if records:
                            logger.debug(
                                f"수집 완료: lawd_cd={lawd_cd}, "
                                f"deal_ymd={deal_ymd}, 건수={len(records)}"
                            )

                    except APIError as e:
                        logger.error(f"API 오류 (건너뜀): {e}")
                    except Exception as e:
                        logger.error(
                            f"예상치 못한 오류 "
                            f"(lawd_cd={lawd_cd}, deal_ymd={deal_ymd}): {e}"
                        )
                    finally:
                        pbar.update(1)

        logger.info(f"수집 완료 | 총 {len(all_records):,}건")
        return all_records
