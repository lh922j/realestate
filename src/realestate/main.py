"""
main.py - CLI 진입점

Click 기반 커맨드라인 인터페이스.
터미널에서 직접 데이터 수집, 통계 확인, CSV 내보내기를 실행할 수 있습니다.

사용법:
    # 패키지로 실행 (프로젝트 루트에서)
    python -m src.realestate.main [COMMAND] [OPTIONS]

    # 서울 전체 매매 데이터 수집 (2023년)
    python -m src.realestate.main collect --type trade --region 서울 --start 202301 --end 202312

    # 강남구 전월세 데이터 수집 (원본 JSON도 저장)
    python -m src.realestate.main collect --type rent --lawd-cd 11680 --start 202401 --end 202412 --save-raw

    # 여러 지역 동시 지정
    python -m src.realestate.main collect --type trade --lawd-cd 11680 --lawd-cd 11710 --start 202301 --end 202312

    # DB 현황 확인
    python -m src.realestate.main stats

    # 강남구 매매 데이터 CSV 내보내기
    python -m src.realestate.main export --type trade --lawd-cd 11680 --output ./강남_매매.csv
"""

import sys

import click
from loguru import logger

from .config import (
    DEFAULT_END_YM,
    DEFAULT_START_YM,
    LOG_DIR,
    REGION_CODES,
    REGION_GROUPS,
)
from .collector import MolitCollector
from .preprocessor import Preprocessor
from .storage import Storage


def setup_logging(log_level: str = "INFO"):
    """Loguru 로거 설정: 콘솔 + 파일 동시 출력"""
    from datetime import datetime

    log_file = LOG_DIR / f"collect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    logger.add(
        str(log_file),
        level="DEBUG",
        rotation="50 MB",
        encoding="utf-8",
    )


@click.group()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    show_default=True,
    help="로그 레벨",
)
def cli(log_level: str):
    """국토교통부 아파트 실거래가 데이터 수집기

    \b
    지원 데이터:
        trade  - 아파트 매매 실거래가
        rent   - 아파트 전월세 실거래가
    """
    setup_logging(log_level)


@cli.command()
@click.option(
    "--type", "data_type",
    type=click.Choice(["trade", "rent"], case_sensitive=False),
    required=True,
    help="수집 유형: trade(매매) / rent(전월세)",
)
@click.option(
    "--region",
    type=click.Choice(list(REGION_GROUPS.keys())),
    default=None,
    help=f"시도 단위 수집 ({'/'.join(REGION_GROUPS.keys())})",
)
@click.option(
    "--lawd-cd", "lawd_cds",
    multiple=True,
    help="5자리 지역코드. 복수 지정 가능 (예: --lawd-cd 11680 --lawd-cd 11710)",
)
@click.option(
    "--start",
    default=DEFAULT_START_YM,
    show_default=True,
    help="시작 연월 (YYYYMM)",
)
@click.option(
    "--end",
    default=DEFAULT_END_YM,
    show_default=True,
    help="종료 연월 (YYYYMM)",
)
@click.option(
    "--save-raw",
    is_flag=True,
    default=False,
    help="원시 API 응답을 JSON으로도 저장",
)
def collect(
    data_type: str,
    region: str | None,
    lawd_cds: tuple[str, ...],
    start: str,
    end: str,
    save_raw: bool,
):
    """API 데이터 수집 및 저장

    \b
    예시:
        # 서울 전체 매매 (2023년)
        collect --type trade --region 서울 --start 202301 --end 202312

        # 강남구 + 송파구 전월세 (2024년)
        collect --type rent --lawd-cd 11680 --lawd-cd 11710 --start 202401 --end 202412
    """
    # 지역코드 결정 우선순위: --lawd-cd > --region > 전체
    if lawd_cds:
        codes = list(lawd_cds)
        logger.info(f"지정 지역코드 {len(codes)}개 수집")
    elif region:
        codes = REGION_GROUPS[region]
        logger.info(f"{region} 전체 {len(codes)}개 구군 수집")
    else:
        codes = list(REGION_CODES.values())
        logger.info(f"전체 지역 {len(codes)}개 수집")

    label = f"{start}_{end}"

    # 1. 수집
    collector = MolitCollector()
    raw_records = collector.collect_all(
        data_type=data_type,
        region_codes=codes,
        start_ym=start,
        end_ym=end,
    )

    if not raw_records:
        logger.warning("수집된 데이터가 없습니다. API 키와 지역코드를 확인하세요.")
        sys.exit(0)

    storage = Storage()

    # 2. 원시 데이터 저장 (선택)
    if save_raw:
        storage.save_raw(raw_records, data_type, label)

    # 3. 전처리
    preprocessor = Preprocessor()
    df = preprocessor.process(raw_records, data_type=data_type)

    if df.empty:
        logger.warning("전처리 후 유효한 데이터가 없습니다.")
        sys.exit(0)

    # 4. CSV 저장
    csv_path = storage.save_csv(df, data_type, label)

    # 5. SQLite 저장
    inserted = storage.save_db(df, data_type)

    click.echo(f"\n수집 완료!")
    click.echo(f"  전처리 건수: {len(df):,}건")
    click.echo(f"  신규 저장  : {inserted:,}건")
    click.echo(f"  CSV 경로   : {csv_path}")


@cli.command()
def stats():
    """DB 저장 현황 통계 출력"""
    storage = Storage()
    stats_data = storage.get_stats()

    click.echo("\n=== 데이터베이스 현황 ===")
    for data_type, info in stats_data.items():
        label = "아파트 매매" if data_type == "trade" else "아파트 전월세"
        click.echo(f"\n[{label} ({data_type})]")
        if "error" in info:
            click.echo(f"  오류: {info['error']}")
        else:
            click.echo(f"  총 건수  : {info['total']:,}건")
            click.echo(f"  기간     : {info['earliest']} ~ {info['latest']}")
            click.echo(f"  지역 수  : {info['regions']}개")


@cli.command()
@click.option(
    "--type", "data_type",
    type=click.Choice(["trade", "rent"], case_sensitive=False),
    required=True,
    help="내보낼 데이터 유형",
)
@click.option("--lawd-cd", default=None, help="5자리 지역코드 필터")
@click.option("--start", default=None, help="시작일 필터 (YYYY-MM-DD)")
@click.option("--end",   default=None, help="종료일 필터 (YYYY-MM-DD)")
@click.option(
    "--output", default="./export.csv", show_default=True,
    help="출력 CSV 파일 경로",
)
def export(
    data_type: str,
    lawd_cd: str | None,
    start: str | None,
    end: str | None,
    output: str,
):
    """DB 데이터를 CSV 파일로 내보내기

    \b
    예시:
        # 강남구 전체 매매 데이터 내보내기
        export --type trade --lawd-cd 11680 --output ./강남_매매.csv

        # 2023년 전월세 데이터 내보내기
        export --type rent --start 2023-01-01 --end 2023-12-31 --output ./2023_전월세.csv
    """
    storage = Storage()
    df = storage.load_df(
        data_type,
        lawd_cd=lawd_cd,
        start_date=start,
        end_date=end,
    )

    if df.empty:
        click.echo("해당 조건의 데이터가 없습니다.")
        sys.exit(0)

    # utf-8-sig: Excel에서 한글 깨짐 방지
    df.to_csv(output, index=False, encoding="utf-8-sig")
    click.echo(f"내보내기 완료: {output} ({len(df):,}건)")


@cli.command()
@click.option(
    "--lawd-cd",
    default="11680",
    show_default=True,
    help="테스트할 5자리 지역코드 (기본: 강남구)",
)
@click.option(
    "--deal-ymd",
    default="202401",
    show_default=True,
    help="테스트할 거래 연월 (YYYYMM)",
)
def test_api(lawd_cd: str, deal_ymd: str):
    """API 연결 테스트 (1건만 요청)

    API 키가 정상인지, 응답 구조가 예상과 맞는지 확인합니다.

    \b
    예시:
        test-api --lawd-cd 11680 --deal-ymd 202401
    """
    import requests
    import xml.etree.ElementTree as ET
    from .config import API_KEY, BASE_URLS

    click.echo(f"\nAPI 연결 테스트: lawd_cd={lawd_cd}, deal_ymd={deal_ymd}")
    click.echo(f"API 키 앞 10자리: {API_KEY[:10]}...")

    for data_type, url in BASE_URLS.items():
        label = "매매" if data_type == "trade" else "전월세"
        try:
            resp = requests.get(
                url,
                params={
                    "serviceKey": API_KEY,
                    "LAWD_CD":    lawd_cd,
                    "DEAL_YMD":   deal_ymd,
                    "numOfRows":  1,
                    "pageNo":     1,
                },
                timeout=15,
            )
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            if "json" in content_type:
                data = resp.json()
                header = data.get("response", {}).get("header", {})
                body   = data.get("response", {}).get("body", {})
                result_code = header.get("resultCode", "")
                total_count = body.get("totalCount", 0)
                result_msg  = header.get("resultMsg", "")
            else:
                root = ET.fromstring(resp.content)
                result_code = (root.findtext(".//resultCode") or "").strip()
                result_msg  = (root.findtext(".//resultMsg") or "").strip()
                total_count = int(root.findtext(".//totalCount") or 0)

            if result_code in ("00", "000", "0000"):
                click.echo(
                    f"  [{label}] OK - "
                    f"총 {int(total_count):,}건 (이번 달 기준)"
                )
            else:
                click.echo(
                    f"  [{label}] API 오류 [{result_code}]: {result_msg}"
                )
        except Exception as e:
            click.echo(f"  [{label}] 연결 실패: {e}")


@cli.command()
@click.option(
    "--region",
    type=click.Choice(list(REGION_GROUPS.keys())),
    default="수도권",
    show_default=True,
    help="지오코딩 대상 시도 (기본: 수도권)",
)
@click.option(
    "--lawd-cd", "lawd_cds",
    multiple=True,
    help="특정 지역코드만 처리 (복수 지정 가능)",
)
@click.option(
    "--retry-failed",
    is_flag=True,
    default=False,
    help="이전에 실패한 항목 재시도",
)
@click.option(
    "--stats-only",
    is_flag=True,
    default=False,
    help="지오코딩 현황 통계만 출력 (처리 없음)",
)
def geocode(
    region: str,
    lawd_cds: tuple[str, ...],
    retry_failed: bool,
    stats_only: bool,
):
    """아파트 단지 주소 → 위경도 좌표 변환 (Kakao 로컬 API)

    MOLIT DB에 저장된 아파트 단지를 Kakao API로 지오코딩합니다.
    결과는 apt_geocode 테이블에 캐시되어 재실행 시 스킵됩니다.

    \b
    사전 준비:
        .env 파일에 KAKAO_API_KEY 추가 (developers.kakao.com)

    \b
    예시:
        # 수도권 전체 지오코딩
        geocode

        # 서울만 지오코딩
        geocode --region 서울

        # 이전 실패 항목 재시도
        geocode --retry-failed

        # 현황 확인만
        geocode --stats-only
    """
    from .config import KAKAO_API_KEY

    storage = Storage()

    if stats_only:
        s = storage.get_geocode_stats()
        click.echo("\n=== 지오코딩 현황 ===")
        click.echo(f"  DB 내 유니크 단지: {s['total_in_db']:,}개")
        click.echo(f"  처리 완료        : {s['processed']:,}개")
        click.echo(f"  미처리           : {s['pending']:,}개")
        click.echo(f"  성공 (Kakao)     : {s['kakao']:,}개")
        click.echo(f"  성공 (VWorld)    : {s['vworld']:,}개")
        click.echo(f"  실패             : {s['failed']:,}개")
        return

    if not KAKAO_API_KEY:
        click.echo(
            "KAKAO_API_KEY가 설정되지 않았습니다.\n"
            ".env 파일에 KAKAO_API_KEY를 추가하세요.\n"
            "발급: https://developers.kakao.com"
        )
        sys.exit(1)

    # 대상 지역코드 결정
    if lawd_cds:
        target_codes = set(lawd_cds)
    else:
        target_codes = set(REGION_GROUPS[region])

    # DB에서 유니크 단지 목록 추출
    import pandas as pd
    with __import__("sqlite3").connect(storage.db_path) as conn:
        df_apts = pd.read_sql_query(
            """
            SELECT DISTINCT apt_name, lawd_cd, dong_name, jibun
            FROM apt_trade
            WHERE lawd_cd IN ({})
            """.format(",".join("?" * len(target_codes))),
            conn,
            params=list(target_codes),
        )

    if df_apts.empty:
        click.echo(
            "지오코딩할 아파트 데이터가 없습니다. "
            "먼저 collect 명령으로 데이터를 수집하세요."
        )
        sys.exit(0)

    click.echo(f"유니크 단지 수: {len(df_apts):,}개")

    from .geocoder import KakaoGeocoder

    geocoder = KakaoGeocoder()
    records = df_apts.to_dict("records")
    processed = geocoder.geocode_batch(
        records=records,
        storage=storage,
        skip_existing=True,
        retry_failed=retry_failed,
    )

    s = storage.get_geocode_stats()
    click.echo(f"\n지오코딩 완료!")
    click.echo(f"  처리 건수: {processed:,}개")
    click.echo(f"  성공 합계: {s['kakao'] + s['vworld']:,}개")
    click.echo(f"  실패     : {s['failed']:,}개")


@cli.command()
@click.option(
    "--type", "data_type",
    type=click.Choice(["trade", "rent"]),
    default="trade",
    show_default=True,
    help="학습 데이터 종류: trade(매매가) / rent(전세 보증금)",
)
@click.option(
    "--model", "model_type",
    type=click.Choice(["lgbm", "xgboost", "ridge"]),
    default="lgbm",
    show_default=True,
    help="사용할 ML 모델",
)
@click.option(
    "--cutoff",
    default="2024-01-01",
    show_default=True,
    help="Train/Test 분리 기준일 (이 날짜 이후=테스트셋)",
)
@click.option(
    "--region",
    type=click.Choice(list(REGION_GROUPS.keys())),
    default="수도권",
    show_default=True,
    help="학습 대상 지역",
)
@click.option(
    "--name",
    default="price_model",
    show_default=True,
    help="저장할 모델 파일명",
)
@click.option(
    "--importance",
    is_flag=True,
    default=False,
    help="피처 중요도 출력",
)
@click.option(
    "--optuna-trials",
    default=0,
    show_default=True,
    type=int,
    help="Optuna 하이퍼파라미터 탐색 횟수 (0=사용 안 함, 권장 50~100)",
)
@click.option(
    "--compare",
    "use_compare",
    is_flag=True,
    default=False,
    help="7개 모델 자동 비교 후 리더보드 출력 (LightGBM, XGBoost, RF, ExtraTrees 등)",
)
@click.option(
    "--no-spatial",
    is_flag=True,
    default=False,
    help="공간 피처(시군구 경계) 없이 학습 (geopandas 미설치 환경용)",
)
@click.option(
    "--complex-split",
    is_flag=True,
    default=False,
    help="단지 단위 분리: 테스트셋 단지를 학습에서 완전히 제외 (일반화 성능 측정)",
)
@click.option(
    "--test-ratio",
    default=0.2,
    show_default=True,
    type=float,
    help="--complex-split 사용 시 테스트 단지 비율 (기본 0.2 = 20%%)",
)
def train(
    data_type: str,
    model_type: str,
    cutoff: str,
    region: str,
    name: str,
    importance: bool,
    optuna_trials: int,
    use_compare: bool,
    no_spatial: bool,
    complex_split: bool,
    test_ratio: float,
):
    """아파트 매매가 예측 ML 모델 학습

    MOLIT 실거래가 + Kakao 지오코딩 결과를 결합하여
    매매가 예측 모델을 학습합니다.

    \b
    학습 방법:
        기본:   LightGBM / XGBoost / Ridge
        Optuna: 하이퍼파라미터 자동 최적화 (--optuna-trials 50)
        PyCaret: 여러 모델 자동 비교 후 최적 앙상블 (--pycaret)

    \b
    사전 준비:
        1. collect 명령으로 매매 데이터 수집
        2. geocode 명령으로 좌표 확보

    \b
    예시:
        # LightGBM 기본
        train

        # Optuna로 LightGBM 하이퍼파라미터 최적화 (50회 탐색)
        train --model lgbm --optuna-trials 50

        # PyCaret AutoML (모델 자동 비교)
        train --pycaret

        # geopandas 없이 학습
        train --no-spatial
    """
    import pandas as pd
    from .ml import MLPipeline
    from .spatial import SpatialFeatureBuilder

    import numpy as np
    storage = Storage()
    target_codes = REGION_GROUPS[region]
    label = "매매" if data_type == "trade" else "전월세(보증금)"

    # 1. 데이터 로드
    logger.info(f"{label} 데이터 로드 중...")
    df_trade = storage.load_df(data_type)
    df_trade = df_trade[df_trade["lawd_cd"].isin(target_codes)].copy()

    if df_trade.empty:
        click.echo(f"학습 데이터가 없습니다. collect --type {data_type} 명령을 먼저 실행하세요.")
        sys.exit(0)

    # 전월세: deposit → deal_amount 로 rename해서 기존 파이프라인 재활용
    if data_type == "rent":
        if "deposit" not in df_trade.columns:
            click.echo("deposit 컬럼이 없습니다. 전월세 데이터를 다시 수집하세요.")
            sys.exit(1)
        df_trade = df_trade[df_trade["deposit"] > 0].copy()
        df_trade["deal_amount"] = df_trade["deposit"]

    click.echo(f"{label} 데이터: {len(df_trade):,}건")

    # 2. 지오코딩 결과 병합
    df_geo = storage.load_geocode()
    if df_geo.empty:
        click.echo("지오코딩 결과가 없습니다. geocode 명령을 먼저 실행하세요.")
        sys.exit(0)

    df_trade["apt_key"] = df_trade["apt_name"] + "_" + df_trade["lawd_cd"]
    df_trade = df_trade.merge(
        df_geo[["apt_key", "latitude", "longitude"]],
        on="apt_key",
        how="left",
    )
    geocoded_pct = df_trade["latitude"].notna().mean() * 100
    click.echo(f"좌표 보유율: {geocoded_pct:.1f}%")

    # 3. 공간 피처 생성
    builder = SpatialFeatureBuilder()
    df_trade = builder.build_features(df_trade, use_boundaries=not no_spatial)

    # 4. Train/Test 분리
    df_trade["deal_date"] = pd.to_datetime(df_trade["deal_date"], errors="coerce")

    if complex_split:
        all_keys = df_trade["apt_key"].dropna().unique()
        rng = np.random.default_rng(42)
        rng.shuffle(all_keys)
        n_test = max(1, int(len(all_keys) * test_ratio))
        test_keys  = set(all_keys[:n_test])
        train_keys = set(all_keys[n_test:])
        train_mask = df_trade["apt_key"].isin(train_keys)
        test_mask  = df_trade["apt_key"].isin(test_keys)
        click.echo(
            f"[단지 분리] Train: {len(train_keys):,}개 단지 {train_mask.sum():,}건 | "
            f"Test: {len(test_keys):,}개 단지 {test_mask.sum():,}건"
        )
    else:
        cutoff_ts  = pd.Timestamp(cutoff)
        train_mask = df_trade["deal_date"] < cutoff_ts
        test_mask  = ~train_mask & df_trade["deal_date"].notna()
        click.echo(
            f"[시계열 분리] Train: {train_mask.sum():,}건 (< {cutoff}) | "
            f"Test:  {test_mask.sum():,}건 (>= {cutoff})"
        )

    if train_mask.sum() < 100:
        click.echo("학습 데이터가 너무 적습니다 (< 100건). 데이터를 더 수집하세요.")
        sys.exit(1)

    pipeline = MLPipeline(model_type=model_type)

    # 5. 피처 행렬 생성 (compare / Optuna / 기본 공통)
    X, y = pipeline.build_feature_matrix(df_trade)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    # 6. 학습 방법 분기
    if use_compare:
        # ── 모델 비교 리더보드 ────────────────────────────────────
        click.echo("\n모델 비교 시작 (7개 모델)...")
        leaderboard = pipeline.compare_models(X_train, y_train, X_test, y_test)
        click.echo(f"\n=== 모델 비교 리더보드 ===")
        click.echo(leaderboard.to_string(index=False))
        best = leaderboard.iloc[0]
        metrics = {
            "mae":     best["MAE(만원)"],
            "rmse":    best["RMSE(만원)"],
            "r2":      best["R²"],
            "mae_eok": best["MAE(억원)"],
        }

    else:
        if optuna_trials > 0:
            # ── Optuna HPO ────────────────────────────────────────
            click.echo(f"\nOptuna 하이퍼파라미터 탐색: {optuna_trials}회 ({model_type})")
            metrics = pipeline.train_optuna(
                X_train, y_train, X_test, y_test, n_trials=optuna_trials
            )
            click.echo(f"\n최적 파라미터: {pipeline.best_params_}")
        else:
            # ── 기본 학습 ─────────────────────────────────────────
            metrics = pipeline.train(X_train, y_train, X_test, y_test)

    # 7. 공통 평가 결과 출력
    if not use_compare:  # compare는 이미 리더보드 출력함
        method = "Optuna+" + model_type if optuna_trials > 0 else model_type
        click.echo(f"\n=== 평가 결과 ({method}) ===")
        click.echo(f"  MAE  : {metrics['mae']:,.0f} 만원 ({metrics['mae_eok']:.2f} 억원)")
        click.echo(f"  RMSE : {metrics['rmse']:,.0f} 만원")
        click.echo(f"  R²   : {metrics['r2']:.4f}")

    # 8. 피처 중요도
    if importance:
        fi_df = pipeline.feature_importance()
        click.echo("\n=== 피처 중요도 (상위 15개) ===")
        click.echo(fi_df.head(15).to_string(index=False))

    # 8. 모델 저장 (기본명은 data_type 포함)
    save_name = name if name != "price_model" else f"price_model_{data_type}"
    model_path = pipeline.save(save_name)
    click.echo(f"\n모델 저장: {model_path}")


@cli.command()
@click.option(
    "--region",
    type=click.Choice(list(REGION_GROUPS.keys())),
    default="수도권",
    show_default=True,
    help="분석 대상 지역",
)
@click.option(
    "--contamination",
    default=0.02,
    show_default=True,
    type=float,
    help="이상치 비율 (0.01=1%%, 0.05=5%%, 기본 0.02=2%%)",
)
@click.option(
    "--top",
    default=30,
    show_default=True,
    type=int,
    help="출력할 상위 이상치 건수",
)
@click.option(
    "--no-spatial",
    is_flag=True,
    default=False,
    help="공간 피처(시군구 경계) 없이 실행 (geopandas 미설치 환경용)",
)
@click.option(
    "--by-district",
    "by_district",
    is_flag=True,
    default=True,
    help="구 단위로 분리하여 탐지 (기본 ON — 지역 간 가격차 제거)",
)
@click.option(
    "--output",
    default=None,
    help="이상치 결과 저장 CSV 경로 (미지정 시 저장 안 함)",
)
def anomaly(
    region: str,
    contamination: float,
    top: int,
    no_spatial: bool,
    by_district: bool,
    output: str | None,
):
    """Isolation Forest 기반 이상 거래 탐지

    가격·면적·위치·시간 피처를 동시에 고려하여
    전체 분포에서 밀도가 낮은 거래(이상치)를 탐지합니다.

    \b
    이상치 유형:
        - 같은 지역·면적대 대비 비정상 고가/저가 거래
        - 특이한 층·건축년도 조합
        - 직거래 급매 등 시세 이탈 거래

    \b
    예시:
        # 수도권 전체 이상 탐지 (이상치 2%)
        anomaly

        # 서울만, 이상치 비율 5%로 설정 후 CSV 저장
        anomaly --region 서울 --contamination 0.05 --output anomalies.csv

        # 상위 50건만 출력
        anomaly --top 50
    """
    import pandas as pd
    from .ml import MLPipeline
    from .spatial import SpatialFeatureBuilder

    storage = Storage()
    target_codes = REGION_GROUPS[region]

    # 1. 데이터 로드
    logger.info("매매 데이터 로드 중...")
    df = storage.load_df("trade")
    df = df[df["lawd_cd"].isin(target_codes)].copy()

    if df.empty:
        click.echo("데이터가 없습니다. collect --type trade 명령을 먼저 실행하세요.")
        sys.exit(0)

    click.echo(f"로드 완료: {len(df):,}건")

    # 2. 지오코딩 결과 병합
    df_geo = storage.load_geocode()
    if not df_geo.empty:
        df["apt_key"] = df["apt_name"] + "_" + df["lawd_cd"]
        df = df.merge(
            df_geo[["apt_key", "latitude", "longitude"]],
            on="apt_key",
            how="left",
        )
        geocoded_pct = df["latitude"].notna().mean() * 100
        click.echo(f"좌표 보유율: {geocoded_pct:.1f}%")
    else:
        click.echo("지오코딩 데이터 없음 — 위치 피처 미사용")

    # 3. 공간 피처 생성
    if "latitude" in df.columns:
        builder = SpatialFeatureBuilder()
        df = builder.build_features(df, use_boundaries=not no_spatial)

    # 4. Isolation Forest 이상 탐지
    pipeline = MLPipeline()
    mode = "구 단위 분리" if by_district else "전체"
    click.echo(f"\nIsolation Forest 실행 중 ({mode}, contamination={contamination:.0%})...")
    df = pipeline.detect_anomalies(df, contamination=contamination, by_district=by_district)

    # 5. 결과 요약 출력
    anomalies = df[df["is_anomaly"]].copy()
    normals   = df[~df["is_anomaly"]].copy()

    click.echo(f"\n=== 이상 탐지 결과 ===")
    click.echo(f"  전체 거래 수  : {len(df):,}건")
    click.echo(f"  이상치 건수   : {len(anomalies):,}건 ({len(anomalies)/len(df)*100:.1f}%)")

    if "deal_amount" in df.columns:
        click.echo(f"\n  [정상 거래 통계]")
        click.echo(f"    평균가: {normals['deal_amount'].mean()/10000:.2f} 억원")
        click.echo(f"    중앙값: {normals['deal_amount'].median()/10000:.2f} 억원")
        click.echo(f"  [이상 거래 통계]")
        click.echo(f"    평균가: {anomalies['deal_amount'].mean()/10000:.2f} 억원")
        click.echo(f"    중앙값: {anomalies['deal_amount'].median()/10000:.2f} 억원")
        if "price_per_sqm" in anomalies.columns:
            click.echo(f"    평당가 평균: {anomalies['price_per_sqm'].mean():,.0f} 만원/㎡")

    # 6. 상위 이상치 출력 (anomaly_score 낮은 순 = 더 이상한 순)
    display_cols = [
        c for c in [
            "deal_date", "apt_name", "dong_name", "area_exclusive",
            "floor", "deal_amount", "price_per_sqm",
            "dealing_type", "anomaly_score",
        ] if c in anomalies.columns
    ]
    top_anomalies = anomalies.sort_values("anomaly_score").head(top)

    click.echo(f"\n=== 상위 {min(top, len(anomalies))}건 이상 거래 (anomaly_score 낮은 순) ===")
    if "deal_amount" in top_anomalies.columns:
        top_anomalies = top_anomalies.copy()
        top_anomalies["가격(억)"] = (top_anomalies["deal_amount"] / 10000).round(2)

    click.echo(top_anomalies[display_cols].to_string(index=False))

    # 7. 직거래 비율 비교
    if "dealing_type" in df.columns:
        normal_direct  = (normals["dealing_type"] == "직거래").mean() * 100
        anomaly_direct = (anomalies["dealing_type"] == "직거래").mean() * 100
        click.echo(f"\n  직거래 비율 — 정상: {normal_direct:.1f}% | 이상치: {anomaly_direct:.1f}%")

    # 8. DB 저장 (전체 결과 — 정상 + 이상치 모두)
    saved = storage.save_anomalies(df)
    click.echo(f"\nDB 저장: {saved:,}건 (정상 + 이상치)")

    # 9. CSV 저장 (이상치만)
    if output:
        save_cols = display_cols + [
            c for c in ["lawd_cd", "build_year", "latitude", "longitude"]
            if c in anomalies.columns and c not in display_cols
        ]
        anomalies[save_cols].sort_values("anomaly_score").to_csv(
            output, index=False, encoding="utf-8-sig"
        )
        click.echo(f"CSV 저장: {output} ({len(anomalies):,}건)")


if __name__ == "__main__":
    cli()
