"""
geocoder.py - 아파트 단지 주소 → 위경도 좌표 변환

Kakao 로컬 API를 1차로 사용하고, 실패 시 VWorld API로 대체합니다.
결과는 SQLite apt_geocode 테이블에 캐시되어 재실행 시 중복 호출을 방지합니다.

API 발급:
    Kakao: https://developers.kakao.com → 앱 등록 → REST API 키
    VWorld: https://www.vworld.kr/dev/v4api.do (선택사항)
"""

import time
from datetime import datetime
from typing import Literal

import requests
import pandas as pd
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    retry_if_exception,
)

from .config import (
    KAKAO_API_KEY,
    VWORLD_API_KEY,
    GEOCODE_DELAY,
    GEOCODE_MAX_RETRIES,
    GEOCODE_RETRY_WAIT,
)


class GeocoderError(Exception):
    pass


class KakaoGeocoder:
    """
    Kakao 로컬 API 기반 지오코더

    무료 한도: 일 30만 건
    쿼리 전략:
      1차: "{apt_name} {dong_name}" (단지명 + 법정동)
      2차: "{dong_name} {jibun}" (법정동 + 지번)
    """

    _KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"
    _ADDRESS_URL = "https://dapi.kakao.com/v2/local/search/address.json"

    def __init__(self):
        if not KAKAO_API_KEY:
            logger.warning(
                "KAKAO_API_KEY가 설정되지 않았습니다. "
                ".env 파일에 KAKAO_API_KEY를 추가하세요."
            )
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"KakaoAK {KAKAO_API_KEY}"})

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        if isinstance(exc, requests.HTTPError):
            return exc.response is not None and exc.response.status_code not in (401, 403)
        return isinstance(exc, requests.RequestException)

    @retry(
        stop=stop_after_attempt(GEOCODE_MAX_RETRIES),
        wait=wait_exponential(multiplier=GEOCODE_RETRY_WAIT, min=2, max=30),
        retry=retry_if_exception(lambda e: KakaoGeocoder._is_retryable(e)),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )
    def _call_keyword(self, query: str) -> list[dict]:
        resp = self.session.get(
            self._KEYWORD_URL,
            params={"query": query, "size": 1},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("documents", [])

    @retry(
        stop=stop_after_attempt(GEOCODE_MAX_RETRIES),
        wait=wait_exponential(multiplier=GEOCODE_RETRY_WAIT, min=2, max=30),
        retry=retry_if_exception(lambda e: KakaoGeocoder._is_retryable(e)),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )
    def _call_address(self, query: str) -> list[dict]:
        resp = self.session.get(
            self._ADDRESS_URL,
            params={"query": query, "size": 1},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("documents", [])

    def geocode(
        self,
        apt_name: str,
        dong_name: str,
        jibun: str = "",
    ) -> dict | None:
        """
        단지명 + 법정동으로 위경도 좌표 조회

        Args:
            apt_name:  아파트 단지명
            dong_name: 법정동명
            jibun:     지번 (1차 실패 시 대체 쿼리에 사용)

        Returns:
            {"lat": float, "lng": float, "source": "kakao", "query": str}
            또는 None (조회 실패)
        """
        time.sleep(GEOCODE_DELAY)

        # 1차 시도: 단지명 + 법정동
        query1 = f"{apt_name} {dong_name}"
        docs = self._call_keyword(query1)
        if docs:
            return {
                "lat":    float(docs[0]["y"]),
                "lng":    float(docs[0]["x"]),
                "source": "kakao",
                "query":  query1,
            }

        # 2차 시도: 법정동 + 지번 (주소 검색)
        if jibun:
            time.sleep(GEOCODE_DELAY)
            query2 = f"{dong_name} {jibun}"
            docs = self._call_address(query2)
            if docs:
                return {
                    "lat":    float(docs[0]["y"]),
                    "lng":    float(docs[0]["x"]),
                    "source": "kakao",
                    "query":  query2,
                }

        return None

    def geocode_batch(
        self,
        records: list[dict],
        storage,
        skip_existing: bool = True,
        retry_failed: bool = False,
    ) -> int:
        """
        단지 목록 일괄 지오코딩

        Args:
            records:        apt_name, lawd_cd, dong_name, jibun 포함 dict 리스트
            storage:        Storage 인스턴스 (캐시 저장용)
            skip_existing:  이미 처리된 apt_key는 스킵 (기본 True)
            retry_failed:   source='failed'인 항목도 재시도

        Returns:
            처리된 건수
        """
        from tqdm import tqdm

        # 기존 처리 결과 로드 (스킵 여부 판단용)
        existing_keys: set[str] = set()
        if skip_existing:
            with __import__("sqlite3").connect(storage.db_path) as conn:
                rows = conn.execute("SELECT apt_key, source FROM apt_geocode").fetchall()
            if retry_failed:
                existing_keys = {r[0] for r in rows if r[1] != "failed"}
            else:
                existing_keys = {r[0] for r in rows}

        to_process = [
            r for r in records
            if f"{r['apt_name']}_{r['lawd_cd']}" not in existing_keys
        ]
        logger.info(f"지오코딩 대상: {len(to_process):,}건 (전체 {len(records):,}건)")

        results: list[dict] = []
        success = failed = 0

        for rec in tqdm(to_process, desc="지오코딩", unit="단지"):
            apt_key = f"{rec['apt_name']}_{rec['lawd_cd']}"
            try:
                result = self.geocode(
                    apt_name=rec["apt_name"],
                    dong_name=rec.get("dong_name", ""),
                    jibun=rec.get("jibun", ""),
                )
            except Exception as e:
                logger.debug(f"Kakao 오류 [{apt_key}]: {e}")
                result = None

            # Kakao 실패 → VWorld 시도 (키가 있는 경우만)
            if result is None and VWORLD_API_KEY:
                vw = VWorldGeocoder()
                result = vw.geocode(
                    dong_name=rec.get("dong_name", ""),
                    jibun=rec.get("jibun", ""),
                )

            if result:
                success += 1
                row = {
                    "apt_key":     apt_key,
                    "apt_name":    rec["apt_name"],
                    "lawd_cd":     rec["lawd_cd"],
                    "dong_name":   rec.get("dong_name", ""),
                    "query_used":  result["query"],
                    "latitude":    result["lat"],
                    "longitude":   result["lng"],
                    "source":      result["source"],
                    "geocoded_at": datetime.now().isoformat(),
                }
            else:
                failed += 1
                row = {
                    "apt_key":     apt_key,
                    "apt_name":    rec["apt_name"],
                    "lawd_cd":     rec["lawd_cd"],
                    "dong_name":   rec.get("dong_name", ""),
                    "query_used":  f"{rec['apt_name']} {rec.get('dong_name', '')}",
                    "latitude":    None,
                    "longitude":   None,
                    "source":      "failed",
                    "geocoded_at": datetime.now().isoformat(),
                }

            results.append(row)

            # 100건마다 중간 저장 (중단 시 손실 최소화)
            if len(results) >= 100:
                storage.save_geocode(pd.DataFrame(results))
                results = []

        if results:
            storage.save_geocode(pd.DataFrame(results))

        logger.info(f"지오코딩 완료: 성공={success:,} | 실패={failed:,}")
        return success + failed


class VWorldGeocoder:
    """
    VWorld API 기반 지오코더 (Kakao 실패 시 대안)

    무료 한도: 일 10만 건
    발급: https://www.vworld.kr/dev/v4api.do
    """

    _URL = "https://api.vworld.kr/req/address"

    def __init__(self):
        self.session = requests.Session()

    @retry(
        stop=stop_after_attempt(GEOCODE_MAX_RETRIES),
        wait=wait_exponential(multiplier=GEOCODE_RETRY_WAIT, min=2, max=30),
        retry=retry_if_exception_type(requests.RequestException),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )
    def _call(self, address: str) -> dict | None:
        resp = self.session.get(
            self._URL,
            params={
                "service":  "address",
                "request":  "getcoord",
                "version":  "2.0",
                "crs":      "epsg:4326",
                "address":  address,
                "refine":   "true",
                "simple":   "false",
                "format":   "json",
                "type":     "parcel",
                "key":      VWORLD_API_KEY,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("response", {}).get("status") == "OK":
            pt = data["response"]["result"]["point"]
            return {"lat": float(pt["y"]), "lng": float(pt["x"])}
        return None

    def geocode(self, dong_name: str, jibun: str) -> dict | None:
        """
        법정동 + 지번으로 좌표 조회

        Returns:
            {"lat": float, "lng": float, "source": "vworld", "query": str}
            또는 None
        """
        if not VWORLD_API_KEY:
            return None

        time.sleep(GEOCODE_DELAY)
        query = f"{dong_name} {jibun}".strip()
        if not query:
            return None

        try:
            result = self._call(query)
            if result:
                result["source"] = "vworld"
                result["query"]  = query
                return result
        except Exception as e:
            logger.debug(f"VWorld 오류 [{query}]: {e}")

        return None
