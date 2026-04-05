"""
Reconciliation engine.

The ONLY four fields compared are:
  1. SB Number           (SB) ↔ Shipping Bill Number          (eBRC)
  2. SB Date             (SB) ↔ Shipping Bill / Invoice Date  (eBRC)
  3. Invoice Number      (SB) ↔ GST Invoice Number            (eBRC)
  4. Invoice Date        (SB) ↔ GST Invoice Date              (eBRC)

Currency and amount are NOT used for matching (kept as informational only).
"""
from __future__ import annotations

import io
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pdfplumber
from dateutil import parser as dateutil_parser

from models import AuditRow, ReconciliationConfig, ReconciliationStatus, ReconciliationSummary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column alias tables
# ---------------------------------------------------------------------------

SB_ALIASES: Dict[str, List[str]] = {
    # ── Mandatory ──────────────────────────────────────────────────────────
    "shipping_bill_number": [
        "shipping bill number", "sb number", "sb no", "sb_no", "s.b. no",
        "s.b no", "sb.no", "shipping_bill_no", "shipping bill no", "sb num",
        "bill number", "shipment bill number", "shipping bill", "sb",
        "s b number", "shippingbillnumber",
    ],
    "shipping_bill_date": [
        "shipping bill date", "sb date", "sb_date", "bill date",
        "shipment date", "shipping date", "s.b. date", "s b date",
    ],
    "invoice_number": [
        "invoice number", "invoice no", "invoice_no", "inv no", "inv number",
        "inv_no", "invoice num", "invoice#", "invoice_number",
        "export invoice number", "export invoice no",
    ],
    "invoice_date": [
        "invoice date", "inv date", "invoice_date", "export invoice date",
    ],
    # ── ICEGATE combined column: "2.Invoice No. & Dt." ─────────────────────
    # The value looks like "215 29/10/2025" — invoice number + space + date.
    # When detected, the code splits it into invoice_number + invoice_date.
    "invoice_no_date_combined": [
        "invoice no. & dt.", "invoice no. & dt", "invoice no & dt",
        "invoice no.& dt.", "invoice no &dt", "invoice no. and dt.",
        "invoice no. & date", "inv no. & dt.", "inv no & dt",
        # ICEGATE prefixes each column with a digit and dot, e.g. "2.invoice no. & dt."
        "2.invoice no. & dt.", "2.invoice no. & dt", "2.inv no. & dt.",
    ],
    # ── Informational (not used for matching) ──────────────────────────────
    "port_code":  ["port code", "port_code", "port", "customs port"],
    "iec":        ["iec", "iec code", "iec_code", "importer exporter code"],
    "currency":   ["currency", "curr", "currency code", "currency_code"],
    "fob_value":  [
        "fob value", "export value", "fob_value", "export_value", "fob",
        "invoice value", "invoice_value", "total fob", "fob amount",
        "exported value", "export amount", "1.invoice value",
    ],
    "ad_code":    ["ad code", "ad_code", "authorised dealer code", "ad bank code"],
}

EBRC_ALIASES: Dict[str, List[str]] = {
    # ── Mandatory ──────────────────────────────────────────────────────────
    "shipping_bill_number": [
        # Exact names seen in eBRC PDFs
        "shipping bill number", "sb number", "sb no", "sb_no",
        "shipping_bill_no", "shipping bill no", "sb", "bill number",
        "s.b. no", "s b number",
    ],
    "shipping_bill_date": [
        # Exact eBRC column name reported by user
        "shipping bill / invoice date",
        "shipping bill/invoice date",
        "shipping bill/ invoice date",
        "shipping bill /invoice date",
        "sb/invoice date", "sb / invoice date",
        # Fallback generics
        "shipping bill date", "sb date", "sb_date", "bill date",
        "shipment date",
    ],
    "invoice_number": [
        # Exact eBRC column names reported by user
        "gst invoice number", "gst invoice no", "gst inv no",
        "gst invoice num", "gst_invoice_number",
        # Fallback generics
        "invoice number", "invoice no", "invoice_no", "inv no",
        "inv number", "invoice#", "invoice_number",
    ],
    "invoice_date": [
        # Exact eBRC column names reported by user
        "gst invoice date", "gst inv date", "gst_invoice_date",
        # Fallback generics
        "invoice date", "inv date", "invoice_date",
    ],
    # ── Informational ──────────────────────────────────────────────────────
    "ebrc_number": [
        "ebrc number", "ebrc no", "ebrc_no", "brc number", "brc no",
        "brc_no", "ebrc", "brc",
    ],
    "brc_date":        ["brc date", "brc_date", "ebrc date", "realisation date"],
    "currency":        ["currency", "curr", "currency code", "currency_code"],
    "realised_amount": [
        "realised amount", "realized amount", "realised_amount",
        "realized_amount", "amount", "realization amount",
        "realisation amount", "inward remittance", "fcy amount",
        "foreign currency amount",
    ],
    "port_code":       ["port code", "port_code", "port", "customs port"],
    "iec":             ["iec", "iec code", "iec_code", "importer exporter code"],
    "ad_code":         ["ad code", "ad_code", "authorised dealer code", "ad bank code"],
}

# Mandatory fields for eBRC (always separate columns)
EBRC_REQUIRED = ["shipping_bill_number", "shipping_bill_date", "invoice_number", "invoice_date"]

# SB mandatory — invoice_number + invoice_date may come from a single combined column
SB_REQUIRED_BASE   = ["shipping_bill_number", "shipping_bill_date"]
SB_REQUIRED_INVOICE = ["invoice_number", "invoice_date"]   # OR invoice_no_date_combined

# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

def read_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read CSV, Excel, or PDF bytes into a DataFrame."""
    fname = filename.lower()
    if fname.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), dtype=str)
        except Exception:
            df = pd.read_csv(io.BytesIO(file_bytes), dtype=str, encoding="latin-1")
    elif fname.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(file_bytes), dtype=str)
    elif fname.endswith(".pdf"):
        df = _read_pdf(file_bytes, filename)
    else:
        raise ValueError(f"Unsupported file format: {filename}. Use CSV, Excel, or PDF.")

    if df.empty:
        raise ValueError(f"File '{filename}' is empty or no table could be extracted.")
    return df


def _read_pdf(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Extract tabular data from a PDF.
    Strategy 1: pdfplumber table extraction across all pages (deduplicates repeated headers).
    Strategy 2: Raw text fallback for non-grid PDFs.
    """
    all_rows: List[List[str]] = []
    header: Optional[List[str]] = None

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if not tables:
                continue
            table = max(tables, key=lambda t: len(t))
            if not table:
                continue
            for row in table:
                clean_row = [re.sub(r"\s+", " ", str(cell or "")).strip() for cell in row]
                if all(c == "" for c in clean_row):
                    continue
                if header is None:
                    header = clean_row
                    all_rows.append(clean_row)
                else:
                    if clean_row == header:   # repeated header on next page
                        continue
                    all_rows.append(clean_row)

    if all_rows and len(all_rows) > 1:
        df = pd.DataFrame(all_rows[1:], columns=all_rows[0], dtype=str)
        df = df.dropna(how="all").reset_index(drop=True)
        df = df[~df.apply(lambda r: r.str.strip().eq("").all(), axis=1)]
        if not df.empty:
            return df

    logger.warning("%s: no structured tables found; attempting text fallback.", filename)
    return _read_pdf_text_fallback(file_bytes, filename)


def _read_pdf_text_fallback(file_bytes: bytes, filename: str) -> pd.DataFrame:
    lines: List[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            lines.extend(text.splitlines())

    lines = [l for l in lines if l.strip()]
    if not lines:
        raise ValueError(
            f"Could not extract any text from '{filename}'. "
            "The PDF may be scanned/image-based. Please export as CSV or Excel."
        )

    sample = "\n".join(lines[:10])
    delimiter = "\t" if "\t" in sample else r"\s{2,}"

    parsed: List[List[str]] = []
    for line in lines:
        parts = line.split("\t") if delimiter == "\t" else re.split(r"\s{2,}", line)
        parsed.append([p.strip() for p in parts])

    if len(parsed) < 2:
        raise ValueError(
            f"'{filename}': extracted text but could not parse table structure. "
            "Please export as CSV or Excel."
        )

    max_cols = max(len(r) for r in parsed)
    padded = [r + [""] * (max_cols - len(r)) for r in parsed]
    df = pd.DataFrame(padded[1:], columns=padded[0], dtype=str)
    return df.dropna(how="all").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Column mapping helpers
# ---------------------------------------------------------------------------

def normalise_col_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def map_columns(df: pd.DataFrame, alias_table: Dict[str, List[str]]) -> Dict[str, str]:
    norm_df_cols = {normalise_col_name(c): c for c in df.columns}
    mapping: Dict[str, str] = {}
    for canonical, aliases in alias_table.items():
        for alias in aliases:
            if alias in norm_df_cols:
                mapping[canonical] = norm_df_cols[alias]
                break
    return mapping


def validate_columns(mapping: Dict[str, str], required: List[str], file_label: str) -> None:
    missing = [r for r in required if r not in mapping]
    if missing:
        raise ValueError(
            f"{file_label}: required columns not found — "
            + ", ".join(missing)
            + ". Use the '🔍 Preview Columns' button to see what headers were detected."
        )


# ---------------------------------------------------------------------------
# Data normalisation helpers
# ---------------------------------------------------------------------------

def clean_text(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return re.sub(r"\s+", " ", str(val).strip())


def normalise_text(val) -> str:
    return clean_text(val).upper()


def normalise_sb_number(val) -> str:
    v = clean_text(val).upper()
    v = re.sub(r"[^A-Z0-9/]", "", v)
    return v.lstrip("0") if v else v


def normalise_invoice_number(val) -> str:
    v = clean_text(val).upper()
    return re.sub(r"[^A-Z0-9/\-]", "", v)


def parse_date(val) -> Optional[str]:
    """Return ISO date YYYY-MM-DD or None."""
    s = clean_text(val)
    if not s or s.lower() in ("nan", "none", "na", "n/a", "-"):
        return None
    for fmt in (
        "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y",
        "%d-%b-%Y", "%d %b %Y", "%d/%m/%y", "%m/%d/%y",
        "%Y/%m/%d", "%d.%m.%Y",
    ):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    try:
        return dateutil_parser.parse(s, dayfirst=True).strftime("%Y-%m-%d")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# ICEGATE combined field: "215 29/10/2025" → ("215", "2025-10-29")
# ---------------------------------------------------------------------------

# Date patterns to detect inside a combined value (tried in order)
_DATE_PATTERNS = [
    r"\b(\d{2}/\d{2}/\d{4})\b",   # DD/MM/YYYY  ← most common in ICEGATE
    r"\b(\d{2}-\d{2}-\d{4})\b",   # DD-MM-YYYY
    r"\b(\d{2}\.\d{2}\.\d{4})\b", # DD.MM.YYYY
    r"\b(\d{2}/\d{2}/\d{2})\b",   # DD/MM/YY
    r"\b(\d{4}-\d{2}-\d{2})\b",   # YYYY-MM-DD
]


def split_invoice_no_date(val) -> Tuple[str, Optional[str]]:
    """
    Split a combined ICEGATE 'Invoice No. & Dt.' cell.

    Examples:
        "215 29/10/2025"      → ("215",        "2025-10-29")
        "EXP/001 15-03-2025"  → ("EXP/001",    "2025-03-15")
        "INV-42 2025-01-10"   → ("INV-42",     "2025-01-10")
        "123"                 → ("123",         None)   # no date found
    """
    s = clean_text(val)
    if not s or s.lower() in ("nan", "none", "na", "n/a", "-"):
        return "", None

    for pattern in _DATE_PATTERNS:
        m = re.search(pattern, s)
        if m:
            raw_date  = m.group(1)
            inv_part  = (s[: m.start()] + s[m.end() :]).strip()
            return normalise_invoice_number(inv_part), parse_date(raw_date)

    # No date found — treat entire value as invoice number
    return normalise_invoice_number(s), None


# ---------------------------------------------------------------------------
# Build normalised DataFrames
# ---------------------------------------------------------------------------

def _base_normalise(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for idx, row in df.iterrows():
        r: Dict = {"_orig_index": idx}
        for canon, col in mapping.items():
            r[canon] = row.get(col, "")
        rows.append(r)
    return pd.DataFrame(rows)


def build_normalised_sb(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    ndf = _base_normalise(df, mapping)

    ndf["shipping_bill_number"] = ndf["shipping_bill_number"].apply(normalise_sb_number)
    ndf["shipping_bill_date"]   = ndf["shipping_bill_date"].apply(parse_date)

    has_separate_inv = "invoice_number" in mapping and "invoice_date" in mapping
    has_combined     = "invoice_no_date_combined" in mapping

    if has_separate_inv:
        # Straightforward — each is its own column
        ndf["invoice_number"] = ndf["invoice_number"].apply(normalise_invoice_number)
        ndf["invoice_date"]   = ndf["invoice_date"].apply(parse_date)

    elif has_combined:
        # ICEGATE format: "215 29/10/2025" in a single column → split it
        split_results = ndf["invoice_no_date_combined"].apply(split_invoice_no_date)
        ndf["invoice_number"] = split_results.apply(lambda x: x[0])
        ndf["invoice_date"]   = split_results.apply(lambda x: x[1])
        logger.info(
            "SB: used combined 'Invoice No. & Dt.' column — "
            "split into invoice_number + invoice_date for all rows."
        )

    elif "invoice_number" in mapping:
        # Only invoice number present (no date) — keep number, leave date as None
        ndf["invoice_number"] = ndf["invoice_number"].apply(normalise_invoice_number)
        ndf["invoice_date"]   = None
    else:
        ndf["invoice_number"] = None
        ndf["invoice_date"]   = None

    # Informational — fill with None if absent
    for col in ("invoice_no_date_combined", "port_code", "iec", "currency", "fob_value", "ad_code"):
        if col not in ndf.columns:
            ndf[col] = None

    return ndf


def build_normalised_ebrc(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    ndf = _base_normalise(df, mapping)

    ndf["shipping_bill_number"] = ndf["shipping_bill_number"].apply(normalise_sb_number)
    ndf["shipping_bill_date"]   = ndf["shipping_bill_date"].apply(parse_date)
    ndf["invoice_number"]       = ndf["invoice_number"].apply(normalise_invoice_number)
    ndf["invoice_date"]         = ndf["invoice_date"].apply(parse_date)

    # Informational — fill with None if absent
    for col in ("ebrc_number", "brc_date", "currency", "realised_amount", "port_code", "iec", "ad_code"):
        if col not in ndf.columns:
            ndf[col] = None

    return ndf


# ---------------------------------------------------------------------------
# Core comparison — exactly 4 fields
# ---------------------------------------------------------------------------

def compare_sb_ebrc(
    sb: pd.Series,
    ebrc: pd.Series,
) -> Tuple[ReconciliationStatus, str, str]:
    """
    Compare one SB row against one eBRC row on the 4 agreed fields.
    SB Number is already confirmed equal (used as lookup key).

    Returns (status, mismatch_reason, remarks).
    """
    mismatches: List[str] = []

    # ── Field 2: SB Date ────────────────────────────────────────────────────
    sb_date   = sb.get("shipping_bill_date")
    ebrc_date = ebrc.get("shipping_bill_date")
    if sb_date and ebrc_date:
        if sb_date != ebrc_date:
            mismatches.append(
                f"SB Date mismatch (SB file: {sb_date}, eBRC: {ebrc_date})"
            )
    elif sb_date and not ebrc_date:
        mismatches.append("SB Date missing in eBRC")
    elif not sb_date and ebrc_date:
        mismatches.append("SB Date missing in Shipping Bill")

    # ── Field 3: Invoice Number ─────────────────────────────────────────────
    sb_inv   = sb.get("invoice_number")
    ebrc_inv = ebrc.get("invoice_number")
    if sb_inv and ebrc_inv:
        if sb_inv != ebrc_inv:
            mismatches.append(
                f"Invoice Number mismatch (SB file: {sb_inv}, eBRC GST Invoice: {ebrc_inv})"
            )
    elif sb_inv and not ebrc_inv:
        mismatches.append("GST Invoice Number missing in eBRC")
    elif not sb_inv and ebrc_inv:
        mismatches.append("Invoice Number missing in Shipping Bill")

    # ── Field 4: Invoice Date ───────────────────────────────────────────────
    sb_inv_date   = sb.get("invoice_date")
    ebrc_inv_date = ebrc.get("invoice_date")
    if sb_inv_date and ebrc_inv_date:
        if sb_inv_date != ebrc_inv_date:
            mismatches.append(
                f"Invoice Date mismatch (SB file: {sb_inv_date}, eBRC GST Invoice Date: {ebrc_inv_date})"
            )
    elif sb_inv_date and not ebrc_inv_date:
        mismatches.append("GST Invoice Date missing in eBRC")
    elif not sb_inv_date and ebrc_inv_date:
        mismatches.append("Invoice Date missing in Shipping Bill")

    # ── Verdict ─────────────────────────────────────────────────────────────
    if not mismatches:
        return (
            ReconciliationStatus.EXACT_MATCH,
            "",
            "Exact match — SB Number, SB Date, Invoice Number, and Invoice Date all match",
        )

    reason_str = "; ".join(mismatches)

    # 1 field off → PARTIAL_MATCH; 2+ fields off → MISMATCH
    if len(mismatches) == 1:
        return ReconciliationStatus.PARTIAL_MATCH, reason_str, reason_str
    return ReconciliationStatus.MISMATCH, reason_str, reason_str


# ---------------------------------------------------------------------------
# Main reconciliation entry point
# ---------------------------------------------------------------------------

def reconcile(
    sb_bytes: bytes,
    sb_filename: str,
    ebrc_bytes: bytes,
    ebrc_filename: str,
    config: ReconciliationConfig,
) -> Tuple[List[Dict], ReconciliationSummary]:
    summary = ReconciliationSummary()
    results: List[Dict] = []
    now = datetime.utcnow()

    # ── Parse ────────────────────────────────────────────────────────────────
    try:
        sb_raw = read_file(sb_bytes, sb_filename)
    except Exception as exc:
        raise ValueError(f"Cannot read Shipping Bill file: {exc}") from exc

    try:
        ebrc_raw = read_file(ebrc_bytes, ebrc_filename)
    except Exception as exc:
        raise ValueError(f"Cannot read eBRC file: {exc}") from exc

    # ── Map & validate columns ───────────────────────────────────────────────
    sb_map   = map_columns(sb_raw,   SB_ALIASES)
    ebrc_map = map_columns(ebrc_raw, EBRC_ALIASES)

    # SB validation: base fields always required;
    # invoice fields satisfied by separate columns OR the combined column
    validate_columns(sb_map, SB_REQUIRED_BASE, "Shipping Bill")
    has_sb_inv_separate = all(f in sb_map for f in SB_REQUIRED_INVOICE)
    has_sb_inv_combined = "invoice_no_date_combined" in sb_map
    if not has_sb_inv_separate and not has_sb_inv_combined:
        raise ValueError(
            "Shipping Bill: could not find invoice number/date columns. "
            "Expected either separate 'Invoice Number' + 'Invoice Date' columns, "
            "or a combined 'Invoice No. & Dt.' column (ICEGATE format like '215 29/10/2025'). "
            "Use the 🔍 Preview Columns button to see what was detected."
        )

    validate_columns(ebrc_map, EBRC_REQUIRED, "eBRC")

    # ── Normalise ────────────────────────────────────────────────────────────
    sb_df   = build_normalised_sb(sb_raw,   sb_map)
    ebrc_df = build_normalised_ebrc(ebrc_raw, ebrc_map)

    # ── Build eBRC lookup by SB Number ───────────────────────────────────────
    ebrc_by_sb: Dict[str, pd.DataFrame] = {}
    for _, row in ebrc_df.iterrows():
        key = str(row["shipping_bill_number"])
        ebrc_by_sb.setdefault(key, []).append(row)
    ebrc_by_sb = {k: pd.DataFrame(v) for k, v in ebrc_by_sb.items()}

    matched_ebrc_sbs: set = set()

    # ── Process each SB row ──────────────────────────────────────────────────
    for _, sb_row in sb_df.iterrows():
        sb_num = str(sb_row["shipping_bill_number"])
        orig_sb_num = sb_raw.iloc[int(sb_row["_orig_index"])].get(
            sb_map.get("shipping_bill_number", ""), sb_num
        )

        try:
            candidates = ebrc_by_sb.get(sb_num, pd.DataFrame())

            if candidates.empty:
                status     = ReconciliationStatus.MISSING_IN_EBRC
                reason     = "No corresponding eBRC found for this SB Number"
                remarks    = reason
                matched_id = None

            elif len(candidates) == 1:
                ebrc_row   = candidates.iloc[0]
                matched_id = str(ebrc_row.get("ebrc_number") or sb_num)
                status, reason, remarks = compare_sb_ebrc(sb_row, ebrc_row)
                matched_ebrc_sbs.add(sb_num)

            else:
                # Multiple eBRC rows share the same SB Number
                matched_ebrc_sbs.add(sb_num)

                # Narrow down by invoice number first, then invoice date
                inv_matches = candidates[
                    candidates["invoice_number"] == sb_row.get("invoice_number", "")
                ]
                date_matches = inv_matches[
                    inv_matches["invoice_date"] == sb_row.get("invoice_date", "")
                ] if not inv_matches.empty else pd.DataFrame()

                best_pool = (
                    date_matches if not date_matches.empty
                    else inv_matches if not inv_matches.empty
                    else candidates
                )

                if len(best_pool) == 1:
                    ebrc_row   = best_pool.iloc[0]
                    matched_id = str(ebrc_row.get("ebrc_number") or sb_num)
                    status, reason, remarks = compare_sb_ebrc(sb_row, ebrc_row)
                else:
                    status     = ReconciliationStatus.AMBIGUOUS_MATCH
                    reason     = (
                        f"{len(candidates)} eBRC rows share SB Number {sb_num} "
                        "and could not be narrowed to a single match"
                    )
                    remarks    = reason
                    matched_id = None

        except Exception as exc:
            logger.exception("Error processing SB row %s", sb_num)
            status     = ReconciliationStatus.ERROR
            reason     = str(exc)
            remarks    = f"Processing error: {exc}"
            matched_id = None

        _emit(results, summary, sb_num, orig_sb_num, matched_id, status, reason, remarks, now)

    # ── Flag eBRC rows with no matching SB ──────────────────────────────────
    for ebrc_sb_num, ebrc_group in ebrc_by_sb.items():
        if ebrc_sb_num not in matched_ebrc_sbs:
            for _, ebrc_row in ebrc_group.iterrows():
                ebrc_ref = str(ebrc_row.get("ebrc_number") or ebrc_sb_num)
                audit = AuditRow(
                    source_identifier=ebrc_ref,
                    matched_identifier=None,
                    status=ReconciliationStatus.MISSING_IN_SHIPPING_BILL,
                    mismatch_reason="No corresponding Shipping Bill found",
                    remarks="No corresponding Shipping Bill found",
                    timestamp=now,
                    sheets_update_result="NOT_APPLICABLE",
                )
                summary.increment(ReconciliationStatus.MISSING_IN_SHIPPING_BILL)
                summary.audit_trail.append(audit)
                results.append({
                    "sb_number": ebrc_sb_num,
                    "orig_sb_number": ebrc_sb_num,
                    "matched_id": None,
                    "status": ReconciliationStatus.MISSING_IN_SHIPPING_BILL,
                    "remarks": "No corresponding Shipping Bill found",
                    "sheets_update": False,
                })

    return results, summary


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _emit(
    results: List[Dict],
    summary: ReconciliationSummary,
    sb_num: str,
    orig_sb_num: str,
    matched_id: Optional[str],
    status: ReconciliationStatus,
    reason: str,
    remarks: str,
    now: datetime,
) -> None:
    audit = AuditRow(
        source_identifier=orig_sb_num,
        matched_identifier=matched_id,
        status=status,
        mismatch_reason=reason or None,
        remarks=remarks,
        timestamp=now,
    )
    summary.increment(status)
    summary.audit_trail.append(audit)
    results.append({
        "sb_number": sb_num,
        "orig_sb_number": orig_sb_num,
        "matched_id": matched_id,
        "status": status,
        "remarks": remarks,
        "sheets_update": True,
    })
