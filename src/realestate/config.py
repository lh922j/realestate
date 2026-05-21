"""
config.py - 프로젝트 전체 설정 중앙 관리

모든 모듈이 이 파일을 import하여 설정을 가져옵니다.
API 키, 엔드포인트, 지역코드, 날짜 범위, 경로 등을 정의합니다.
"""

import os
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

# 프로젝트 루트 기준 경로 설정 (src/realestate/config.py 기준으로 3단계 위)
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ─────────────────────────────────────────────────────────────────
# API 설정
# ─────────────────────────────────────────────────────────────────

API_KEY = os.getenv("MOLIT_API_KEY")
if not API_KEY:
    raise EnvironmentError(
        "MOLIT_API_KEY가 설정되지 않았습니다.\n"
        ".env 파일을 생성하고 API 키를 입력하세요. (.env.example 참고)\n"
        "API 키 발급: https://www.data.go.kr"
    )

KAKAO_API_KEY  = os.getenv("KAKAO_API_KEY", "")
VWORLD_API_KEY = os.getenv("VWORLD_API_KEY", "")

BASE_URLS = {
    "trade": "https://apis.data.go.kr/1613000/RTMSDataSvcAptTrade/getRTMSDataSvcAptTrade",
    "rent":  "https://apis.data.go.kr/1613000/RTMSDataSvcAptRent/getRTMSDataSvcAptRent",
}

# ─────────────────────────────────────────────────────────────────
# Rate Limiting & Retry 설정
# ─────────────────────────────────────────────────────────────────

REQUEST_DELAY_SECONDS = 0.5   # 요청 간 대기 (공공데이터포털 분당 1000건 제한)
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 2
PAGE_SIZE = 1000              # 한 번에 가져올 최대 건수

# ─────────────────────────────────────────────────────────────────
# 지역코드 (lawd_cd) 매핑
# ─────────────────────────────────────────────────────────────────
# 행정구역 코드 5자리: 앞 2자리=시도, 뒤 3자리=구군
# 전체 코드 조회: https://www.code.go.kr (행정표준코드관리시스템)

REGION_CODES = {
    # 서울특별시 (25개 자치구)
    "서울_종로구":    "11110",
    "서울_중구":      "11140",
    "서울_용산구":    "11170",
    "서울_성동구":    "11200",
    "서울_광진구":    "11215",
    "서울_동대문구":  "11230",
    "서울_중랑구":    "11260",
    "서울_성북구":    "11290",
    "서울_강북구":    "11305",
    "서울_도봉구":    "11320",
    "서울_노원구":    "11350",
    "서울_은평구":    "11380",
    "서울_서대문구":  "11410",
    "서울_마포구":    "11440",
    "서울_양천구":    "11470",
    "서울_강서구":    "11500",
    "서울_구로구":    "11530",
    "서울_금천구":    "11545",
    "서울_영등포구":  "11560",
    "서울_동작구":    "11590",
    "서울_관악구":    "11620",
    "서울_서초구":    "11650",
    "서울_강남구":    "11680",
    "서울_송파구":    "11710",
    "서울_강동구":    "11740",
    # 경기도
    "경기_수원_장안구":   "41111",
    "경기_수원_권선구":   "41113",
    "경기_수원_팔달구":   "41115",
    "경기_수원_영통구":   "41117",
    "경기_성남_수정구":   "41131",
    "경기_성남_중원구":   "41133",
    "경기_성남_분당구":   "41135",
    "경기_의정부시":      "41150",
    "경기_광명시":        "41210",
    "경기_동두천시":      "41250",
    "경기_고양_덕양구":   "41281",
    "경기_고양_일산동구": "41285",
    "경기_고양_일산서구": "41287",
    "경기_구리시":        "41310",
    "경기_남양주시":      "41360",
    "경기_오산시":        "41370",
    "경기_시흥시":        "41390",
    "경기_군포시":        "41410",
    "경기_의왕시":        "41430",
    "경기_하남시":        "41450",
    "경기_용인_처인구":   "41461",
    "경기_용인_기흥구":   "41463",
    "경기_용인_수지구":   "41465",
    "경기_파주시":        "41480",
    "경기_안양_만안구":   "41171",
    "경기_안양_동안구":   "41173",
    "경기_부천_원미구":   "41192",
    "경기_부천_소사구":   "41194",
    "경기_부천_오정구":   "41196",
    "경기_화성시":        "41590",
    "경기_김포시":        "41570",
    "경기_과천시":        "41290",
    "경기_양주시":        "41630",
    "경기_포천시":        "41650",
    # 인천광역시
    "인천_중구":      "28110",
    "인천_동구":      "28140",
    "인천_미추홀구":  "28177",
    "인천_연수구":    "28185",
    "인천_남동구":    "28200",
    "인천_부평구":    "28237",
    "인천_계양구":    "28245",
    "인천_서구":      "28260",
    # 부산광역시
    "부산_중구":      "26110",
    "부산_서구":      "26140",
    "부산_동구":      "26170",
    "부산_영도구":    "26200",
    "부산_부산진구":  "26230",
    "부산_동래구":    "26260",
    "부산_남구":      "26290",
    "부산_북구":      "26320",
    "부산_해운대구":  "26350",
    "부산_사하구":    "26380",
    "부산_금정구":    "26410",
    "부산_연제구":    "26440",
    "부산_수영구":    "26470",
    # 대구광역시
    "대구_중구":      "27110",
    "대구_동구":      "27140",
    "대구_서구":      "27170",
    "대구_남구":      "27200",
    "대구_북구":      "27230",
    "대구_수성구":    "27260",
    "대구_달서구":    "27290",
    # 대전광역시
    "대전_동구":      "30110",
    "대전_중구":      "30140",
    "대전_서구":      "30170",
    "대전_유성구":    "30200",
    "대전_대덕구":    "30230",
}

# 시도 단위 그룹핑 (대량 수집 시 유용)
REGION_GROUPS = {
    "서울": [v for k, v in REGION_CODES.items() if k.startswith("서울")],
    "경기": [v for k, v in REGION_CODES.items() if k.startswith("경기")],
    "인천": [v for k, v in REGION_CODES.items() if k.startswith("인천")],
    "부산": [v for k, v in REGION_CODES.items() if k.startswith("부산")],
    "대구": [v for k, v in REGION_CODES.items() if k.startswith("대구")],
    "대전": [v for k, v in REGION_CODES.items() if k.startswith("대전")],
}

# ─────────────────────────────────────────────────────────────────
# 날짜 범위 헬퍼
# ─────────────────────────────────────────────────────────────────

def generate_year_months(start_ym: str, end_ym: str) -> list[str]:
    """
    YYYYMM 형식의 날짜 범위 생성

    Args:
        start_ym: 시작 연월 (YYYYMM, 예: "202001")
        end_ym:   종료 연월 (YYYYMM, 예: "202312")

    Returns:
        연월 문자열 리스트 ["202001", "202002", ..., "202312"]

    Example:
        >>> generate_year_months("202301", "202303")
        ['202301', '202302', '202303']
    """
    start = datetime.strptime(start_ym, "%Y%m")
    end   = datetime.strptime(end_ym,   "%Y%m")
    result = []
    current = start
    while current <= end:
        result.append(current.strftime("%Y%m"))
        month = current.month + 1
        year  = current.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        current = current.replace(year=year, month=month)
    return result


# ─────────────────────────────────────────────────────────────────
# 기본 수집 범위
# ─────────────────────────────────────────────────────────────────

DEFAULT_START_YM = "202001"   # 2020년 1월부터
DEFAULT_END_YM   = "202502"   # 2025년 2월까지

# ─────────────────────────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────────────────────────

DATA_DIR      = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LOG_DIR       = PROJECT_ROOT / "logs"
DB_PATH       = PROCESSED_DIR / "realestate.db"
GEO_DIR       = DATA_DIR / "geo"
MODEL_DIR     = DATA_DIR / "models"

# 디렉토리 자동 생성 (최초 실행 시)
for _dir in [RAW_DIR, PROCESSED_DIR, LOG_DIR, GEO_DIR, MODEL_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# 지오코딩 설정
# ─────────────────────────────────────────────────────────────────

GEOCODE_DELAY       = 0.1   # 초 (Kakao: 30만 건/일 한도)
GEOCODE_MAX_RETRIES = 3
GEOCODE_RETRY_WAIT  = 2     # 초

# 거리 피처 기준점 — 강남역 좌표
GANGNAM_LAT = 37.4979
GANGNAM_LNG = 127.0276

# 수도권 그룹 (geocode, train 명령 기본값)
REGION_GROUPS["수도권"] = (
    REGION_GROUPS["서울"] + REGION_GROUPS["경기"] + REGION_GROUPS["인천"]
)
