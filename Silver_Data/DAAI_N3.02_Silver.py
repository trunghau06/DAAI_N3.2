from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# // Khai báo file Silver
OUTPUT_NAMES = {
    "customers.csv": "customers_silver.csv",
    "epd.json": "epd_silver.csv",
    "eprom.json": "eprom_silver.csv",
    "geography.csv": "geography_silver.csv",
    "inventory.csv": "inventory_silver.csv",
    "order_items.csv": "order_items_silver.csv",
    "orders_enriched.csv": "orders_silver.csv",
    "payments.csv": "payments_silver.csv",
    "products.csv": "products_silver.csv",
    "promotions.csv": "promotions_silver.csv",
    "returns.csv": "returns_silver.csv",
    "reviews.csv": "reviews_silver.csv",
    "shipments_realistic.csv": "shipments_silver.csv",
    "tf.json": "tf_silver.csv",
    "web_traffic.csv": "web_traffic_silver.csv",
}

DATE_COLUMNS = {
    "customers.csv": ["signup_date"],
    "eprom.json": ["start_date", "end_date"],
    "inventory.csv": ["snapshot_date"],
    "orders_enriched.csv": ["order_date"],
    "promotions.csv": ["start_date", "end_date"],
    "returns.csv": ["return_date"],
    "reviews.csv": ["review_date"],
    "shipments_realistic.csv": ["ship_date", "delivery_date", "join_date"],
    "tf.json": ["date"],
    "web_traffic.csv": ["date"],
}

INTEGER_COLUMNS = {
    "customers.csv": ["customer_id", "zip"],
    "epd.json": ["product_id"],
    "eprom.json": ["stackable_flag", "min_order_value"],
    "geography.csv": ["zip"],
    "inventory.csv": [
        "product_id", "stock_on_hand", "units_received", "units_sold",
        "stockout_days", "stockout_flag", "overstock_flag", "reorder_flag",
        "year", "month",
    ],
    "order_items.csv": ["order_id", "product_id", "quantity"],
    "orders_enriched.csv": ["order_id", "customer_id", "zip", "years_experience"],
    "payments.csv": ["order_id", "installments"],
    "products.csv": ["product_id"],
    "promotions.csv": ["stackable_flag", "min_order_value"],
    "returns.csv": ["order_id", "product_id", "return_quantity"],
    "reviews.csv": ["order_id", "product_id", "customer_id", "rating"],
    "shipments_realistic.csv": [
        "order_id", "shipper_experience_years", "average_delivery_time",
        "shipper_age",
    ],
    "tf.json": ["sessions", "unique_visitors", "page_views"],
    "web_traffic.csv": ["sessions", "unique_visitors", "page_views"],
}

FLOAT_COLUMNS = {
    "epd.json": ["price", "cogs"],
    "eprom.json": ["discount_value"],
    "inventory.csv": ["days_of_supply", "fill_rate", "sell_through_rate"],
    "order_items.csv": ["unit_price", "discount_amount"],
    "payments.csv": ["payment_value"],
    "products.csv": ["price", "cogs"],
    "promotions.csv": ["discount_value"],
    "returns.csv": ["refund_amount"],
    "shipments_realistic.csv": [
        "shipping_fee", "shipper_rating", "delivery_success_rate"
    ],
    "tf.json": ["bounce_rate", "avg_session_duration_sec"],
    "web_traffic.csv": ["bounce_rate", "avg_session_duration_sec"],
}

FLAG_COLUMNS = {
    "eprom.json": ["stackable_flag"],
    "inventory.csv": ["stockout_flag", "overstock_flag", "reorder_flag"],
    "promotions.csv": ["stackable_flag"],
}

NONNEGATIVE_COLUMNS = {
    "epd.json": ["price", "cogs"],
    "eprom.json": ["discount_value", "min_order_value"],
    "inventory.csv": [
        "stock_on_hand", "units_received", "units_sold", "stockout_days",
        "days_of_supply", "fill_rate", "sell_through_rate",
    ],
    "order_items.csv": ["quantity", "unit_price", "discount_amount"],
    "orders_enriched.csv": ["years_experience"],
    "payments.csv": ["payment_value", "installments"],
    "products.csv": ["price", "cogs"],
    "promotions.csv": ["discount_value", "min_order_value"],
    "returns.csv": ["return_quantity", "refund_amount"],
    "reviews.csv": ["rating"],
    "shipments_realistic.csv": [
        "shipping_fee", "shipper_experience_years", "shipper_rating",
        "delivery_success_rate", "average_delivery_time", "shipper_age",
    ],
    "tf.json": [
        "sessions", "unique_visitors", "page_views", "bounce_rate",
        "avg_session_duration_sec",
    ],
    "web_traffic.csv": [
        "sessions", "unique_visitors", "page_views", "bounce_rate",
        "avg_session_duration_sec",
    ],
}

RATE_0_1_COLUMNS = {
    "inventory.csv": ["fill_rate", "sell_through_rate"],
    "tf.json": ["bounce_rate"],
    "web_traffic.csv": ["bounce_rate"],
}

VALID_CATEGORIES = {"Casual", "GenZ", "Outdoor", "Streetwear"}
VALID_SEGMENTS = {
    "Activewear", "All-weather", "Balanced", "Everyday",
    "Performance", "Premium", "Standard", "Trendy",
}

MISSING_TOKENS = {"", "n/a", "na", "null", "none", "unknown"}


def _strip_text(value):
    if pd.isna(value):
        return pd.NA
    value = str(value).strip()
    if value.lower() in MISSING_TOKENS:
        return pd.NA
    return value


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    # // Làm sạch chuỗi
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].map(_strip_text)
    return df


def clean_dates(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    # // Chuẩn hóa ngày
    for col in DATE_COLUMNS.get(source_name, []):
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            df[col] = parsed.dt.strftime("%Y-%m-%d")
    return df


def parse_special_numeric(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if source_name == "eprom.json" and "discount_value" in df.columns:
        s = df["discount_value"].astype("string").str.strip().str.lower()
        s = s.str.replace("twenty percent", "20", regex=False)
        s = s.str.replace("%", "", regex=False)
        df["discount_value"] = s
    return df


def clean_numeric(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    # // Chuẩn hóa số
    for col in INTEGER_COLUMNS.get(source_name, []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")

    for col in FLOAT_COLUMNS.get(source_name, []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")

    for col in NONNEGATIVE_COLUMNS.get(source_name, []):
        if col in df.columns:
            df.loc[df[col] < 0, col] = pd.NA

    for col in FLAG_COLUMNS.get(source_name, []):
        if col in df.columns:
            df.loc[~df[col].isin([0, 1]) & df[col].notna(), col] = pd.NA
            df[col] = df[col].astype("Int64")

    for col in RATE_0_1_COLUMNS.get(source_name, []):
        if col in df.columns:
            df.loc[((df[col] < 0) | (df[col] > 1)) & df[col].notna(), col] = pd.NA

    return df


def clean_common_business_rules(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    # // Kiểm tra giá trị hợp lệ
    if "category" in df.columns:
        df.loc[~df["category"].isin(VALID_CATEGORIES) & df["category"].notna(), "category"] = pd.NA
    if "segment" in df.columns:
        df.loc[~df["segment"].isin(VALID_SEGMENTS) & df["segment"].notna(), "segment"] = pd.NA

    if source_name == "reviews.csv" and "rating" in df.columns:
        df.loc[~df["rating"].between(1, 5) & df["rating"].notna(), "rating"] = pd.NA

    if source_name == "inventory.csv":
        if "month" in df.columns:
            df.loc[~df["month"].between(1, 12) & df["month"].notna(), "month"] = pd.NA
        if "stockout_days" in df.columns:
            df.loc[df["stockout_days"] > 31, "stockout_days"] = pd.NA

    if source_name in {"promotions.csv", "eprom.json"}:
        if {"promo_type", "discount_value"}.issubset(df.columns):
            pct = df["promo_type"].astype("string").str.lower().eq("percentage")
            invalid = pct & df["discount_value"].notna() & ~df["discount_value"].between(0, 100)
            df.loc[invalid, "discount_value"] = pd.NA

        if {"start_date", "end_date"}.issubset(df.columns):
            start = pd.to_datetime(df["start_date"], errors="coerce")
            end = pd.to_datetime(df["end_date"], errors="coerce")
            df.loc[end < start, "end_date"] = pd.NA

    if source_name == "shipments_realistic.csv":
        if "shipper_rating" in df.columns:
            df.loc[~df["shipper_rating"].between(0, 5) & df["shipper_rating"].notna(), "shipper_rating"] = pd.NA
        if "delivery_success_rate" in df.columns:
            df.loc[~df["delivery_success_rate"].between(0, 100) & df["delivery_success_rate"].notna(), "delivery_success_rate"] = pd.NA
        if "shipper_age" in df.columns:
            df.loc[~df["shipper_age"].between(18, 80) & df["shipper_age"].notna(), "shipper_age"] = pd.NA
        if "shipper_phone" in df.columns:
            df["shipper_phone"] = df["shipper_phone"].map(
                lambda x: pd.NA if pd.isna(x) else re.sub(r"\.0$", "", str(x).strip())
            )

    return df


def clean_epd(df: pd.DataFrame) -> pd.DataFrame:
    # // Sửa mã sản phẩm lỗi
    ids = pd.to_numeric(df["product_id"], errors="coerce")
    used = set(ids.dropna().astype(int).tolist())
    duplicated = ids.duplicated(keep=False) & ids.notna()

    for idx in df.index:
        expected = int(idx) + 1
        current = ids.loc[idx]
        needs_repair = pd.isna(current) or (duplicated.loc[idx] and expected not in used)
        if needs_repair and expected not in used:
            ids.loc[idx] = expected
            used.add(expected)

    df["product_id"] = ids.round().astype("Int64")
    return df


def clean_tf(df: pd.DataFrame) -> pd.DataFrame:
    # // Chuẩn hóa traffic
    if "traffic_source" in df.columns:
        s = df["traffic_source"].astype("string")
        s = s.str.replace(r"\s+Variant\s+\d+\s*$", "", regex=True).str.strip()
        s = s.mask(s.str.lower().isin(MISSING_TOKENS), pd.NA)
        df["traffic_source"] = s

    if {"sessions", "page_views"}.issubset(df.columns):
        bad = df["sessions"].notna() & df["page_views"].notna() & (df["sessions"] > df["page_views"])
        df.loc[bad, "sessions"] = pd.NA

    if {"unique_visitors", "sessions"}.issubset(df.columns):
        bad = df["unique_visitors"].notna() & df["sessions"].notna() & (df["unique_visitors"] > df["sessions"])
        df.loc[bad, "unique_visitors"] = pd.NA

    return df


def clean_frame(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    # // Chạy quy trình làm sạch
    original_columns = list(df.columns)

    df = clean_text_columns(df)
    df = parse_special_numeric(df, source_name)
    df = clean_dates(df, source_name)
    df = clean_numeric(df, source_name)
    df = clean_common_business_rules(df, source_name)

    if source_name == "epd.json":
        df = clean_epd(df)
    if source_name == "tf.json":
        df = clean_tf(df)

    return df[original_columns]


def discover_source_dir(path: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    # // Tìm dữ liệu nguồn
    if path.is_file() and path.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory(prefix="silver_clean_")
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(tmp.name)
        root = Path(tmp.name)
        candidates = [p for p in root.rglob("customers.csv")]
        if not candidates:
            raise FileNotFoundError("Không tìm thấy customers.csv trong ZIP.")
        return candidates[0].parent, tmp

    if path.is_dir():
        if (path / "customers.csv").exists():
            return path, None
        candidates = [p for p in path.rglob("customers.csv")]
        if candidates:
            return candidates[0].parent, None

    raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu trong: {path}")


def read_json(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"JSON phải là danh sách record: {path.name}")
    return pd.DataFrame(obj)


def clean_one_file(source_path: Path, output_path: Path, chunksize: int = 100_000) -> tuple[int, int]:
    # // Làm sạch từng file
    source_name = source_path.name

    if source_path.suffix.lower() == ".json":
        df = read_json(source_path)
        cleaned = clean_frame(df, source_name)
        cleaned.to_csv(output_path, index=False, encoding="utf-8-sig")
        return len(cleaned), len(cleaned.columns)

    first = True
    rows = 0
    cols = None
    for chunk in pd.read_csv(source_path, chunksize=chunksize, low_memory=False):
        cleaned = clean_frame(chunk, source_name)
        cleaned.to_csv(
            output_path,
            mode="w" if first else "a",
            header=first,
            index=False,
            encoding="utf-8-sig" if first else "utf-8",
        )
        first = False
        rows += len(cleaned)
        cols = len(cleaned.columns)

    return rows, int(cols or 0)


def validate_output(source_dir: Path, output_dir: Path) -> None:
    # // Kiểm tra dữ liệu đầu ra
    for source_name, output_name in OUTPUT_NAMES.items():
        src = source_dir / source_name
        out = output_dir / output_name
        if not out.exists():
            raise FileNotFoundError(f"Thiếu output: {out}")

        if src.suffix.lower() == ".json":
            src_df = read_json(src)
            out_df = pd.read_csv(out, low_memory=False)
            src_rows, src_cols = src_df.shape
            out_rows, out_cols = out_df.shape
            src_names = list(src_df.columns)
        else:
            src_header = pd.read_csv(src, nrows=0)
            out_header = pd.read_csv(out, nrows=0)
            src_names = list(src_header.columns)
            if list(out_header.columns) != src_names:
                raise ValueError(f"Sai cột ở {output_name}")
            src_rows = sum(len(chunk) for chunk in pd.read_csv(src, chunksize=200_000, low_memory=False))
            out_rows = sum(len(chunk) for chunk in pd.read_csv(out, chunksize=200_000, low_memory=False))
            src_cols = len(src_names)
            out_cols = len(out_header.columns)

        if out_rows != src_rows:
            raise ValueError(f"Sai số dòng {output_name}: source={src_rows}, output={out_rows}")
        if out_cols != src_cols:
            raise ValueError(f"Sai số cột {output_name}: source={src_cols}, output={out_cols}")
        if list(pd.read_csv(out, nrows=0).columns) != src_names:
            raise ValueError(f"Sai tên/thứ tự cột ở {output_name}")


def main():
    parser = argparse.ArgumentParser(description="Làm sạch 15 file raw thành Silver CSV.")
    parser.add_argument(
        "--input",
        default="student_data.zip",
        help="Đường dẫn student_data.zip hoặc thư mục student_data",
    )
    parser.add_argument(
        "--output-dir",
        default="silver_output/silver_data",
        help="Thư mục xuất Silver CSV",
    )
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_dir, temp_dir = discover_source_dir(input_path)

    try:
        print(f"Nguồn: {source_dir}")
        print(f"Silver: {output_dir}\n")

        for source_name, output_name in OUTPUT_NAMES.items():
            src = source_dir / source_name
            if not src.exists():
                raise FileNotFoundError(f"Thiếu source file: {source_name}")
            out = output_dir / output_name
            rows, cols = clean_one_file(src, out, chunksize=args.chunksize)
            print(f"[OK] {source_name:28s} -> {output_name:28s} | {rows:,} rows | {cols} cols")

        validate_output(source_dir, output_dir)
        print("\nHoàn tất: 15/15 file Silver; số dòng và số cột được giữ nguyên theo từng source file.")
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
