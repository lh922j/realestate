"""
storage.py - 데이터 저장 (CSV + SQLite)

두 가지 형태로 저장합니다:
- CSV: pandas/노트북에서 바로 읽기 편한 형식, Excel 한글 호환(utf-8-sig)
- SQLite: SQL 쿼리, 증분 업데이트, 향후 RAG/Agent 연동에 적합

증분 저장:
    UNIQUE 제약 조건으로 동일한 데이터를 중복 저장하지 않습니다.
    이미 수집된 기간의 데이터를 다시 수집해도 안전합니다.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from loguru import logger

from .config import DB_PATH, PROCESSED_DIR, RAW_DIR

DataType = Literal["trade", "rent"]


class Storage:
    """
    데이터 저장 클래스

    사용 예시:
        storage = Storage()

        # 원시 JSON 저장 (재처리 대비)
        storage.save_raw(records, data_type="trade", label="202301_202312")

        # 전처리된 DataFrame 저장
        storage.save_csv(df, data_type="trade", label="202301_202312")
        storage.save_db(df, data_type="trade")

        # DB에서 다시 로드
        df = storage.load_df("trade", lawd_cd="11680", start_date="2023-01-01")
    """

    # ─────────────────────────────────────────────────────────────
    # SQLite 테이블 스키마
    # ─────────────────────────────────────────────────────────────

    TRADE_SCHEMA = """
    CREATE TABLE IF NOT EXISTS apt_trade (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        -- 거래 정보
        deal_date        DATE    NOT NULL,
        deal_year        INTEGER,
        deal_month       INTEGER,
        deal_day         INTEGER,
        deal_amount      REAL    NOT NULL,   -- 거래금액 (만원)
        deal_amount_eok  REAL,               -- 거래금액 (억원)
        price_per_sqm    REAL,               -- 단위면적당 가격 (만원/㎡)
        dealing_type     TEXT,               -- 거래유형 (직거래/중개)
        -- 건물 정보
        apt_name         TEXT,
        apt_seq          TEXT,               -- 아파트 일련번호
        build_year       INTEGER,
        building_age     INTEGER,
        area_exclusive   REAL,               -- 전용면적 (㎡)
        floor            INTEGER,
        -- 위치 정보
        lawd_cd          TEXT    NOT NULL,   -- 5자리 지역코드
        sgg_code         TEXT,
        dong_name        TEXT,
        jibun            TEXT,
        road_name        TEXT,
        road_main_no     TEXT,
        road_sub_no      TEXT,
        -- 거래 취소 정보
        cancel_type      TEXT,               -- Y=취소
        cancel_day       TEXT,
        register_date    TEXT,
        -- 메타
        deal_ymd         TEXT,
        data_type        TEXT    DEFAULT 'trade',
        created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
        -- 중복 방지: 동일 아파트, 동일 날짜, 동일 금액, 동일 면적, 동일 층
        UNIQUE (apt_seq, deal_date, deal_amount, area_exclusive, floor)
    );
    CREATE INDEX IF NOT EXISTS idx_trade_lawd_cd   ON apt_trade(lawd_cd);
    CREATE INDEX IF NOT EXISTS idx_trade_deal_date ON apt_trade(deal_date);
    CREATE INDEX IF NOT EXISTS idx_trade_apt_name  ON apt_trade(apt_name);
    CREATE INDEX IF NOT EXISTS idx_trade_apt_seq   ON apt_trade(apt_seq);
    CREATE INDEX IF NOT EXISTS idx_trade_sgg_code  ON apt_trade(sgg_code);
    """

    RENT_SCHEMA = """
    CREATE TABLE IF NOT EXISTS apt_rent (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        -- 거래 정보
        deal_date        DATE    NOT NULL,
        deal_year        INTEGER,
        deal_month       INTEGER,
        deal_day         INTEGER,
        deposit          REAL    NOT NULL,   -- 보증금액 (만원)
        deposit_eok      REAL,               -- 보증금액 (억원)
        monthly_rent     REAL,               -- 월세금액 (만원, 0=전세)
        is_jeonse        INTEGER,            -- 전세 여부 (1=전세, 0=월세)
        contract_type    TEXT,               -- 계약구분 (신규/갱신)
        contract_period  TEXT,               -- 계약기간
        pre_deposit      REAL,               -- 종전 보증금 (만원)
        pre_monthly_rent REAL,               -- 종전 월세 (만원)
        -- 건물 정보
        apt_name         TEXT,
        apt_seq          TEXT,
        build_year       INTEGER,
        building_age     INTEGER,
        area_exclusive   REAL,
        floor            INTEGER,
        -- 위치 정보
        lawd_cd          TEXT    NOT NULL,
        sgg_code         TEXT,
        dong_name        TEXT,
        jibun            TEXT,
        road_name        TEXT,
        -- 메타
        deal_ymd         TEXT,
        data_type        TEXT    DEFAULT 'rent',
        created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (apt_seq, deal_date, deposit, monthly_rent, area_exclusive, floor)
    );
    CREATE INDEX IF NOT EXISTS idx_rent_lawd_cd   ON apt_rent(lawd_cd);
    CREATE INDEX IF NOT EXISTS idx_rent_deal_date ON apt_rent(deal_date);
    CREATE INDEX IF NOT EXISTS idx_rent_apt_seq   ON apt_rent(apt_seq);
    CREATE INDEX IF NOT EXISTS idx_rent_is_jeonse ON apt_rent(is_jeonse);
    CREATE INDEX IF NOT EXISTS idx_rent_sgg_code  ON apt_rent(sgg_code);
    """

    GEOCODE_SCHEMA = """
    CREATE TABLE IF NOT EXISTS apt_geocode (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        apt_key     TEXT NOT NULL UNIQUE,   -- "{apt_name}_{lawd_cd}"
        apt_name    TEXT,
        lawd_cd     TEXT,
        dong_name   TEXT,
        query_used  TEXT,                   -- 실제 검색에 사용한 쿼리
        latitude    REAL,
        longitude   REAL,
        source      TEXT,                   -- 'kakao' | 'vworld' | 'failed'
        geocoded_at TEXT,
        CONSTRAINT chk_source CHECK (source IN ('kakao', 'vworld', 'failed'))
    );
    CREATE INDEX IF NOT EXISTS idx_geocode_lawd ON apt_geocode(lawd_cd);
    CREATE INDEX IF NOT EXISTS idx_geocode_key  ON apt_geocode(apt_key);
    """

    ANOMALY_SCHEMA = """
    CREATE TABLE IF NOT EXISTS apt_anomaly (
        id             INTEGER PRIMARY KEY,   -- apt_trade.id 외래키
        anomaly_score  REAL,
        is_anomaly     INTEGER,               -- 1=이상치 0=정상
        price_per_sqm  REAL,
        detected_at    TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_anomaly_is    ON apt_anomaly(is_anomaly);
    CREATE INDEX IF NOT EXISTS idx_anomaly_score ON apt_anomaly(anomaly_score);
    """

    TABLE_NAMES = {"trade": "apt_trade", "rent": "apt_rent"}
    SCHEMAS     = {"trade": TRADE_SCHEMA, "rent": RENT_SCHEMA}

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """SQLite DB 초기화 및 테이블/인덱스 생성"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.TRADE_SCHEMA)
            conn.executescript(self.RENT_SCHEMA)
            conn.executescript(self.GEOCODE_SCHEMA)
            conn.executescript(self.ANOMALY_SCHEMA)
        logger.debug(f"DB 초기화 완료: {self.db_path}")

    # ─────────────────────────────────────────────────────────────
    # 저장 메서드
    # ─────────────────────────────────────────────────────────────

    def save_raw(
        self,
        records: list[dict],
        data_type: str,
        label: str = "",
    ) -> Path:
        """
        원시 API 응답을 JSON으로 저장 (재처리 또는 디버깅용)

        Args:
            records:   collector에서 반환된 원시 dict 리스트
            data_type: "trade" 또는 "rent"
            label:     파일명에 포함할 레이블 (예: "202301_202312")

        Returns:
            저장된 파일 경로
        """
        import json

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"{data_type}_raw_{label}_{timestamp}.json"
        filepath  = RAW_DIR / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        logger.info(f"원시 데이터 저장: {filepath} ({len(records):,}건)")
        return filepath

    def save_csv(
        self,
        df: pd.DataFrame,
        data_type: str,
        label: str = "",
    ) -> Path:
        """
        전처리된 DataFrame을 CSV로 저장

        Args:
            df:        전처리된 pandas DataFrame
            data_type: "trade" 또는 "rent"
            label:     파일명에 포함할 레이블

        Returns:
            저장된 파일 경로
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"{data_type}_{label}_{timestamp}.csv"
        filepath  = PROCESSED_DIR / filename

        # utf-8-sig: Excel에서 한글 깨짐 없이 열리는 인코딩
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        logger.info(f"CSV 저장: {filepath} ({len(df):,}건)")
        return filepath

    def save_db(
        self,
        df: pd.DataFrame,
        data_type: DataType,
    ) -> int:
        """
        전처리된 DataFrame을 SQLite에 증분 저장

        UNIQUE 제약 조건으로 이미 존재하는 레코드는 자동으로 무시됩니다.
        같은 기간 데이터를 반복 수집해도 중복 저장되지 않습니다.

        Args:
            df:        전처리된 pandas DataFrame
            data_type: "trade" 또는 "rent"

        Returns:
            실제로 신규 삽입된 건수
        """
        if df.empty:
            logger.warning("저장할 데이터가 없습니다.")
            return 0

        table = self.TABLE_NAMES[data_type]

        # DB 테이블 컬럼 조회 (실제로 존재하는 컬럼만 삽입)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f"PRAGMA table_info({table})")
            db_cols = {row[1] for row in cursor.fetchall()}

        # id, created_at은 DB 자동 생성 → 제외
        insert_cols = [
            c for c in df.columns
            if c in db_cols and c not in ("id", "created_at")
        ]
        df_insert = df[insert_cols].copy()

        # bool → int 변환 (SQLite는 bool 타입 없음)
        bool_cols = df_insert.select_dtypes(include="bool").columns
        df_insert[bool_cols] = df_insert[bool_cols].astype(int)

        # pandas nullable integer (Int8, Int16 등) → Python 호환 object
        for col in df_insert.columns:
            if hasattr(df_insert[col], "dtype") and str(df_insert[col].dtype).startswith("Int"):
                df_insert[col] = df_insert[col].astype(object).where(
                    df_insert[col].notna(), None
                )

        # deal_date: datetime → string (SQLite DATE 저장)
        if "deal_date" in df_insert.columns:
            df_insert["deal_date"] = df_insert["deal_date"].astype(str)

        before_count = self._get_count(table)

        with sqlite3.connect(self.db_path) as conn:
            # INSERT OR IGNORE: UNIQUE 위반 시 해당 행을 건너뜀
            placeholders = ", ".join(["?" for _ in insert_cols])
            col_names    = ", ".join(insert_cols)
            sql = f"INSERT OR IGNORE INTO {table} ({col_names}) VALUES ({placeholders})"

            rows = [
                tuple(row[col] for col in insert_cols)
                for _, row in df_insert.iterrows()
            ]
            conn.executemany(sql, rows)
            conn.commit()

        after_count = self._get_count(table)
        inserted    = after_count - before_count

        logger.info(
            f"DB 저장 완료: {table} | "
            f"시도={len(df_insert):,}건 | "
            f"신규삽입={inserted:,}건 | "
            f"중복제외={len(df_insert) - inserted:,}건"
        )
        return inserted

    # ─────────────────────────────────────────────────────────────
    # 조회 메서드
    # ─────────────────────────────────────────────────────────────

    def load_df(
        self,
        data_type: DataType,
        lawd_cd: str | None = None,
        start_date: str | None = None,   # "YYYY-MM-DD"
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        SQLite DB에서 DataFrame으로 로드

        Args:
            data_type:   "trade" 또는 "rent"
            lawd_cd:     5자리 지역코드 필터 (None=전체)
            start_date:  시작일 필터 (YYYY-MM-DD, None=제한없음)
            end_date:    종료일 필터 (YYYY-MM-DD, None=제한없음)

        Returns:
            필터링된 pandas DataFrame
        """
        table = self.TABLE_NAMES[data_type]
        conditions = []
        params = []

        if lawd_cd:
            conditions.append("lawd_cd = ?")
            params.append(lawd_cd)
        if start_date:
            conditions.append("deal_date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("deal_date <= ?")
            params.append(end_date)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM {table} {where} ORDER BY deal_date"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                query, conn, params=params, parse_dates=["deal_date"]
            )

        logger.info(f"DB 로드: {table} | {len(df):,}건")
        return df

    def get_stats(self) -> dict:
        """
        DB 저장 현황 통계 반환

        Returns:
            각 테이블의 통계 dict (총 건수, 기간, 지역 수)
        """
        stats = {}
        with sqlite3.connect(self.db_path) as conn:
            for data_type, table in self.TABLE_NAMES.items():
                try:
                    row = conn.execute(f"""
                        SELECT
                            COUNT(*)            AS total,
                            MIN(deal_date)      AS earliest,
                            MAX(deal_date)      AS latest,
                            COUNT(DISTINCT lawd_cd) AS regions
                        FROM {table}
                    """).fetchone()
                    stats[data_type] = {
                        "total":    row[0],
                        "earliest": row[1],
                        "latest":   row[2],
                        "regions":  row[3],
                    }
                except Exception as e:
                    stats[data_type] = {"error": str(e)}
        return stats

    # ─────────────────────────────────────────────────────────────
    # 지오코딩 저장/조회 메서드
    # ─────────────────────────────────────────────────────────────

    def save_geocode(self, df: pd.DataFrame) -> int:
        """
        지오코딩 결과를 apt_geocode 테이블에 저장

        INSERT OR REPLACE: 이미 존재하는 apt_key는 최신 값으로 덮어씀
        (재지오코딩 시 실패→성공으로 갱신 가능)

        Args:
            df: apt_key, apt_name, lawd_cd, dong_name, query_used,
                latitude, longitude, source, geocoded_at 컬럼 포함

        Returns:
            저장/갱신된 건수
        """
        if df.empty:
            return 0

        cols = [
            "apt_key", "apt_name", "lawd_cd", "dong_name",
            "query_used", "latitude", "longitude", "source", "geocoded_at",
        ]
        insert_cols = [c for c in cols if c in df.columns]
        df_insert   = df[insert_cols].copy()

        col_names    = ", ".join(insert_cols)
        placeholders = ", ".join(["?" for _ in insert_cols])
        sql = f"INSERT OR REPLACE INTO apt_geocode ({col_names}) VALUES ({placeholders})"

        rows = [tuple(row[c] for c in insert_cols) for _, row in df_insert.iterrows()]
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(sql, rows)
            conn.commit()

        logger.info(f"지오코딩 저장: {len(rows):,}건")
        return len(rows)

    def load_geocode(self, lawd_cd: str | None = None) -> pd.DataFrame:
        """
        지오코딩 결과 로드

        Args:
            lawd_cd: 5자리 지역코드 필터 (None=전체)

        Returns:
            apt_key, apt_name, lawd_cd, latitude, longitude, source 포함 DataFrame
        """
        conditions = ["source != 'failed'"]
        params: list = []
        if lawd_cd:
            conditions.append("lawd_cd = ?")
            params.append(lawd_cd)

        where = f"WHERE {' AND '.join(conditions)}"
        query = f"SELECT * FROM apt_geocode {where}"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        logger.debug(f"지오코딩 로드: {len(df):,}건")
        return df

    def get_geocode_stats(self) -> dict:
        """
        지오코딩 현황 통계

        Returns:
            total, kakao, vworld, failed, pending(미처리) 건수
        """
        with sqlite3.connect(self.db_path) as conn:
            geocode_row = conn.execute("""
                SELECT
                    COUNT(*)                                        AS total,
                    SUM(CASE WHEN source='kakao'  THEN 1 ELSE 0 END) AS kakao,
                    SUM(CASE WHEN source='vworld' THEN 1 ELSE 0 END) AS vworld,
                    SUM(CASE WHEN source='failed' THEN 1 ELSE 0 END) AS failed
                FROM apt_geocode
            """).fetchone()

            unique_apts = conn.execute("""
                SELECT COUNT(DISTINCT apt_name || '_' || lawd_cd)
                FROM apt_trade
            """).fetchone()[0]

        processed = geocode_row[0]
        return {
            "total_in_db":  unique_apts,
            "processed":    processed,
            "pending":      max(0, unique_apts - processed),
            "kakao":        geocode_row[1] or 0,
            "vworld":       geocode_row[2] or 0,
            "failed":       geocode_row[3] or 0,
        }

    # ─────────────────────────────────────────────────────────────
    # 이상탐지 저장/조회 메서드
    # ─────────────────────────────────────────────────────────────

    def save_anomalies(self, df: pd.DataFrame) -> int:
        """
        이상탐지 결과를 apt_anomaly 테이블에 저장

        df는 apt_trade의 id 컬럼과 anomaly_score, is_anomaly 컬럼을 포함해야 합니다.
        INSERT OR REPLACE로 재실행 시 덮어씁니다.

        Returns:
            저장된 건수
        """
        if "id" not in df.columns or "anomaly_score" not in df.columns:
            logger.warning("id 또는 anomaly_score 컬럼이 없습니다.")
            return 0

        save_df = df[["id", "anomaly_score", "is_anomaly"]].copy()
        save_df["price_per_sqm"] = df["price_per_sqm"] if "price_per_sqm" in df.columns else None
        save_df["detected_at"]   = datetime.now().isoformat()
        save_df["is_anomaly"]    = save_df["is_anomaly"].astype(int)
        save_df = save_df.drop_duplicates(subset=["id"])

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO apt_anomaly "
                "(id, anomaly_score, is_anomaly, price_per_sqm, detected_at) "
                "VALUES (?, ?, ?, ?, ?)",
                save_df[["id", "anomaly_score", "is_anomaly",
                          "price_per_sqm", "detected_at"]].values.tolist(),
            )
            conn.commit()

        logger.info(f"이상탐지 결과 저장: {len(save_df):,}건")
        return len(save_df)

    def load_anomalies(
        self,
        only_anomalies: bool = False,
        lawd_cd: str | None = None,
    ) -> pd.DataFrame:
        """
        apt_trade + apt_anomaly 조인하여 로드

        Args:
            only_anomalies: True면 is_anomaly=1 만 반환
            lawd_cd:        지역코드 필터

        Returns:
            apt_name, lawd_cd, deal_date, deal_amount 등 + anomaly_score, is_anomaly
        """
        conditions = ["a.id IS NOT NULL"]
        params: list = []

        if only_anomalies:
            conditions.append("a.is_anomaly = 1")
        if lawd_cd:
            conditions.append("t.lawd_cd = ?")
            params.append(lawd_cd)

        where = "WHERE " + " AND ".join(conditions)

        query = f"""
        SELECT
            t.id, t.apt_name, t.lawd_cd, t.dong_name,
            t.deal_date, t.deal_year, t.deal_month,
            t.deal_amount, t.area_exclusive, t.floor,
            t.build_year, t.dealing_type,
            a.anomaly_score, a.is_anomaly, a.price_per_sqm
        FROM apt_trade t
        JOIN apt_anomaly a ON t.id = a.id
        {where}
        ORDER BY t.deal_date
        """

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=["deal_date"])

        logger.info(f"이상탐지 로드: {len(df):,}건 (only_anomalies={only_anomalies})")
        return df

    def _get_count(self, table: str) -> int:
        """테이블 총 건수 조회"""
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
