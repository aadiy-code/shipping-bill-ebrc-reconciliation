"""
Reconciliation engine: parses Shipping Bill and eBRC files,
normalises data, runs layered matching, and returns per-row results.
"""
from __future__ import annotations

import io
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil import parser as dateutil_parser

from models import AuditRow, ReconciliationConfig, ReconciliationStatus, ReconciliationSummary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column alias tables
# ---------------------------------------------------------------------------

SB_ALIASES: Dict[str, List[str]] = {
    "shipping_bill_number": [
        "shipping bill number", "sb number", "sb no", "sb_no", "s.b. no",
        "shipping_bill_no", "shipping bill no", "sb num", "bill number",
        "shipment bill number", "shipping bill", "sb",
    ],
    "shipping_bill_date": [
        "shipping bill date", "sb date", "sb_date", "bill date",
        "shipment date", "shipping date",
    ],
    "invoice_number": [
        "invoice number", "invoice no", "invoice_no", "inv no", "inv number",
        "inv_no", "invoice num", "invoice#", "invoice_number",
    ],
    "invoice_date": ["invoice date", "inv date", "invoice_date"],
    "port_code": ["port code", "port_code", "port", "customs port"],
    "iec": ["iec", "iec code", "iec_code", "importer exporter code"],
    "currency": ["currency", "curr", "currency code", "currency_code"],
    "fob_value": [
        "fob value", "export value", "fob_value", "export_value", "fob",
        "invoice value", "invoice_value", "total fob", "total_fob",
        "fob amount", "exported value", "export amount",
    ],
    "ad_code": ["ad code", "ad_code", "authorised dealer code", "ad bank code"],
}

EBRC_ALIASES: Dict[str, List[str]] = {
    "ebrc_number": [
        "ebrc number", "ebrc no", "ebrc_no", "brc number", "brc no",
        "brc_no", "ebrc", "brc",
    ],
    "shipping_bill_number": [
        "shipping bill number", "sb number", "sb no", "sb_no",
        "shipping_bill_no", "shipping bill no", "sb", "bill number",
    ],
    "shipping_bill_date": ["shipping bill date", "sb date", "sb_date", "bill date"],
    "invoice_number": [
        "invoice number", "invoice no", "invoice_no", "inv no",
        "inv number", "invoice#", "invoice_number",
    ],
    "currency": ["currency", "curr", "currency code", "currency_code"],
    "realised_amount": [
        "realised amount", "realized amount", "realised_amount",
        "realized_amount", "amount", "realization amount",
        "realisation amount", "inward remittance", "fcy amount",
        "foreign currency amount",
    ],
    "brc_date": ["brc date", "brc_date", "ebrc date", "realisation date"],
    "port_code": ["port code", "port_code", "port", "customs port"],
    "iec": ["iec", "iec code", "iec_code", "importer exporter code"],
    "ad_code": ["ad code", "ad_code", "authorised dealer code", "ad bank code"],
}

SB_REQUIRED = ["shipping_bill_number", "invoice_number", "currency", "fob_value"]
EBRC_REQUIRED = ["shipping_bill_number", "invoice_number", "currency", "realised_amount"]

# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

def read_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read CSV or Excel bytes into a DataFrame."""
    fname = filename.lower()
    if fname.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), dtype=str)
        except Exception:
            df = pd.read_csv(io.BytesIO(file_bytes), dtype=str, encoding="latin-1")
    elif fname.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(file_bytes), dtype=str)
    else:
        raise ValueError(f"Unsupported file format: {filename}. Use CSV or Excel.")

    if df.empty:
        raise ValueError(f"File '{filename}' is empty.")
    return df


def normalise_col_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def map_columns(df: pd.DataFrame, alias_table: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Returns a mapping from canonical_name -> actual_df_column_name.
    Raises ValueError listing any required-but-missing canonical columns.
    """
    norm_df_cols = {normalise_col_name(c): c for c in df.columns}
    mapping: Dict[str, str] = {}
    for canonical, aliases in alias_table.items():
        for alias in aliases:
            if alias in norm_df_cols:
                mapping[canonical] = norm_df_cols[alias]
                break
    return mapping


def validate_columns(
    mapping: Dict[str, str],
    required: List[str],
    file_label: str,
) -> None:
    missing = [r for r in required if r not in mapping]
    if missing:
        raise ValueError(
            f"{file_label}: required columns not found — "
            + ", ".join(missing)
            + ". Check column headers."
        )


# ---------------------------------------------------------------------------
# Data normalisation helpers
# ---------------------------------------------------------------------------

def clean_text(val: Optional[str]) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return re.sub(r"\s+", " ", str(val).strip())


def normalise_text(val: Optional[str]) -> str:
    return clean_text(val).upper()


def normalise_sb_number(val: Optional[str]) -> str:
    """Remove leading zeros, spaces, special chars; keep alphanumeric."""
    v = clean_text(val).upper()
    v = re.sub(r"[^A-Z0-9/]", "", v)
    return v.lstrip("0") if v else v


def normalise_invoice_number(val: Optional[str]) -> str:
    v = clean_text(val).upper()
    v = re.sub(r"[^A-Z0-9/\-]", "", v)
    return v


def parse_date(val: Optional[str]) -> Optional[str]:
    """Return ISO date string YYYY-MM-DD or None."""
    if not val:
        return None
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


def parse_amount(val: Optional[str]) -> Optional[float]:
    """Parse numeric amount string to float."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "na", "n/a", "-"):
        return None
    # remove currency symbols, commas
    s = re.sub(r"[₹$€£¥,\s]", "", s)
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Build normalised DataFrames
# ---------------------------------------------------------------------------

def build_normalised_sb(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for idx, row in df.iterrows():
        r: Dict[str, object] = {"_orig_index": idx}
        for canon, col in mapping.items():
            r[canon] = row.get(col, "")
        rows.append(r)
    ndf = pd.DataFrame(rows)

    ndf["shipping_bill_number"] = ndf["shipping_bill_number"].apply(normalise_sb_number)
    ndf["invoice_number"] = ndf["invoice_number"].apply(normalise_invoice_number)
    ndf["currency"] = ndf.get("currency", pd.Series(dtype=str)).apply(normalise_text)
    ndf["fob_value"] = ndf["fob_value"].apply(parse_amount)

    if "shipping_bill_date" in ndf.columns:
        ndf["shipping_bill_date"] = ndf["shipping_bill_date"].apply(parse_date)
    else:
        ndf["shipping_bill_date"] = None

    for opt in ("invoice_date", "port_code", "iec", "ad_code"):
        if opt not in ndf.columns:
            ndf[opt] = None
        elif opt in ("port_code", "iec", "ad_code"):
            ndf[opt] = ndf[opt].apply(normalise_text)

    return ndf


def build_normalised_ebrc(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for idx, row in df.iterrows():
        r: Dict[str, object] = {"_orig_index": idx}
        for canon, col in mapping.items():
            r[canon] = row.get(col, "")
        rows.append(r)
    ndf = pd.DataFrame(rows)

    ndf["shipping_bill_number"] = ndf["shipping_bill_number"].apply(normalise_sb_number)
    ndf["invoice_number"] = ndf["invoice_number"].apply(normalise_invoice_number)
    ndf["currency"] = ndf.get("currency", pd.Series(dtype=str)).apply(normalise_text)
    ndf["realised_amount"] = ndf["realised_amount"].apply(parse_amount)

    if "shipping_bill_date" in ndf.columns:
        ndf["shipping_bill_date"] = ndf["shipping_bill_date"].apply(parse_date)
    else:
        ndf["shipping_bill_date"] = None

    for opt in ("ebrc_number", "brc_date", "port_code", "iec", "ad_code"):
        if opt not in ndf.columns:
            ndf[opt] = None
        elif opt in ("port_code", "iec", "ad_code"):
            ndf[opt] = ndf[opt].apply(normalise_text)

    return ndf


# ---------------------------------------------------------------------------
# Amount comparison
# ---------------------------------------------------------------------------

def amounts_match(a: Optional[float], b: Optional[float], tolerance: float) -> bool:
    if a is None or b is None:
        return False
    base = max(abs(a), abs(b), 1.0)
    return abs(a - b) <= tolerance * base


# ---------------------------------------------------------------------------
# Single-row comparison
# ---------------------------------------------------------------------------

def compare_sb_ebrc(
    sb: pd.Series,
    ebrc: pd.Series,
    tolerance: float,
) -> Tuple[ReconciliationStatus, str, str]:
    """
    Compare a single SB row against a single eBRC row.
    Returns (status, mismatch_reason, remarks).
    """
    mismatches: List[str] = []

    # Invoice number
    if sb["invoice_number"] and ebrc["invoice_number"]:
        if sb["invoice_number"] != ebrc["invoice_number"]:
            mismatches.append(
                f"Invoice mismatch (SB: {sb['invoice_number']}, eBRC: {ebrc['invoice_number']})"
            )

    # Shipping bill date
    if sb.get("shipping_bill_date") and ebrc.get("shipping_bill_date"):
        if sb["shipping_bill_date"] != ebrc["shipping_bill_date"]:
            mismatches.append(
                f"Date mismatch (SB: {sb['shipping_bill_date']}, eBRC: {ebrc['shipping_bill_date']})"
            )

    # Currency
    if sb["currency"] and ebrc["currency"]:
        if sb["currency"] != ebrc["currency"]:
            mismatches.append(
                f"Currency mismatch (SB: {sb['currency']}, eBRC: {ebrc['currency']})"
            )

    # Amount
    if sb["fob_value"] is not None and ebrc["realised_amount"] is not None:
        if not amounts_match(sb["fob_value"], ebrc["realised_amount"], tolerance):
            mismatches.append(
                f"Amount mismatch (SB: {sb['fob_value']}, eBRC: {ebrc['realised_amount']})"
            )
    else:
        mismatches.append("Amount not available for comparison")

    # Optional fields — only checked if both sides have values
    for field, label in (("iec", "IEC"), ("port_code", "Port Code"), ("ad_code", "AD Code")):
        sv = sb.get(field)
        ev = ebrc.get(field)
        if sv and ev and sv != ev:
            mismatches.append(f"{label} mismatch (SB: {sv}, eBRC: {ev})")

    if not mismatches:
        return ReconciliationStatus.EXACT_MATCH, "", "Exact match"

    # Determine severity
    critical_keywords = ("Amount", "Currency", "Invoice")
    critical = [m for m in mismatches if any(k in m for k in critical_keywords)]
    reason_str = "; ".join(mismatches)

    if critical:
        return ReconciliationStatus.MISMATCH, reason_str, reason_str
    return ReconciliationStatus.PARTIAL_MATCH, reason_str, reason_str


# ---------------------------------------------------------------------------
# Grouped (one-to-many) matching
# ---------------------------------------------------------------------------

def try_grouped_match(
    sb: pd.Series,
    candidates: pd.DataFrame,
    tolerance: float,
) -> Tuple[bool, str]:
    """
    Check if sum of realised_amounts across candidates equals sb fob_value.
    Returns (matched, remarks).
    """
    valid_amounts = candidates["realised_amount"].dropna()
    if valid_amounts.empty:
        return False, ""
    total = valid_amounts.sum()
    if amounts_match(sb["fob_value"], total, tolerance):
        ebrc_refs = ", ".join(
            str(v) for v in candidates.get("ebrc_number", pd.Series()).dropna().unique()
        )
        return True, (
            f"Grouped amount matched successfully across {len(candidates)} eBRC rows"
            + (f" (eBRC refs: {ebrc_refs})" if ebrc_refs else "")
        )
    return False, ""


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
    """
    Parse, normalise, match, and return per-row result dicts + summary.
    Each result dict contains keys needed for Google Sheets update.
    """
    summary = ReconciliationSummary()
    results: List[Dict] = []
    now = datetime.utcnow()

    # --- Parse files ---
    try:
        sb_raw = read_file(sb_bytes, sb_filename)
    except Exception as exc:
        raise ValueError(f"Cannot read Shipping Bill file: {exc}") from exc

    try:
        ebrc_raw = read_file(ebrc_bytes, ebrc_filename)
    except Exception as exc:
        raise ValueError(f"Cannot read eBRC file: {exc}") from exc

    # --- Map columns ---
    sb_map = map_columns(sb_raw, SB_ALIASES)
    ebrc_map = map_columns(ebrc_raw, EBRC_ALIASES)

    validate_columns(sb_map, SB_REQUIRED, "Shipping Bill")
    validate_columns(ebrc_map, EBRC_REQUIRED, "eBRC")

    # --- Normalise ---
    sb_df = build_normalised_sb(sb_raw, sb_map)
    ebrc_df = build_normalised_ebrc(ebrc_raw, ebrc_map)

    # --- Build eBRC lookup indexed by SB number ---
    ebrc_by_sb: Dict[str, pd.DataFrame] = {}
    for _, row in ebrc_df.iterrows():
        key = str(row["shipping_bill_number"])
        ebrc_by_sb.setdefault(key, []).append(row)
    ebrc_by_sb = {k: pd.DataFrame(v) for k, v in ebrc_by_sb.items()}

    # Track which eBRC SB numbers were consumed (for MISSING_IN_SHIPPING_BILL)
    matched_ebrc_sbs = set()

    # --- Process each SB row ---
    for _, sb_row in sb_df.iterrows():
        sb_num = str(sb_row["shipping_bill_number"])
        orig_sb_num = sb_raw.iloc[int(sb_row["_orig_index"])].get(
            sb_map.get("shipping_bill_number", ""), sb_num
        )

        try:
            candidates = ebrc_by_sb.get(sb_num, pd.DataFrame())

            if candidates.empty:
                status = ReconciliationStatus.MISSING_IN_EBRC
                reason = "No corresponding eBRC found"
                remarks = "No corresponding eBRC found"
                matched_id = None

            elif len(candidates) == 1:
                ebrc_row = candidates.iloc[0]
                matched_id = str(ebrc_row.get("ebrc_number", "") or sb_num)
                status, reason, remarks = compare_sb_ebrc(sb_row, ebrc_row, config.amount_tolerance)
                matched_ebrc_sbs.add(sb_num)

            else:
                # Multiple eBRC rows
                matched_ebrc_sbs.add(sb_num)

                # 1) Try grouped matching
                if config.enable_grouped_matching and sb_row["fob_value"] is not None:
                    grouped_ok, grouped_remarks = try_grouped_match(
                        sb_row, candidates, config.amount_tolerance
                    )
                    if grouped_ok:
                        status = ReconciliationStatus.EXACT_MATCH
                        reason = ""
                        remarks = grouped_remarks
                        matched_id = ",".join(
                            str(v) for v in candidates.get("ebrc_number", pd.Series()).dropna()
                        ) or sb_num
                        _emit(results, summary, sb_num, orig_sb_num, matched_id, status, reason, remarks, now)
                        continue

                # 2) Try to find a single best candidate (invoice + currency match)
                invoice_matches = candidates[
                    candidates["invoice_number"] == sb_row["invoice_number"]
                ]
                currency_matches = invoice_matches[
                    invoice_matches["currency"] == sb_row["currency"]
                ] if not invoice_matches.empty else pd.DataFrame()

                best_pool = (
                    currency_matches if not currency_matches.empty
                    else invoice_matches if not invoice_matches.empty
                    else candidates
                )

                if len(best_pool) == 1:
                    ebrc_row = best_pool.iloc[0]
                    matched_id = str(ebrc_row.get("ebrc_number", "") or sb_num)
                    status, reason, remarks = compare_sb_ebrc(
                        sb_row, ebrc_row, config.amount_tolerance
                    )
                else:
                    status = ReconciliationStatus.AMBIGUOUS_MATCH
                    reason = f"Multiple possible eBRC matches found ({len(candidates)} rows)"
                    remarks = reason
                    matched_id = None

        except Exception as exc:
            logger.exception("Error processing SB row %s", sb_num)
            status = ReconciliationStatus.ERROR
            reason = str(exc)
            remarks = f"Processing error: {exc}"
            matched_id = None

        _emit(results, summary, sb_num, orig_sb_num, matched_id, status, reason, remarks, now)

    # --- Flag eBRC rows with no matching SB ---
    for ebrc_sb_num, ebrc_group in ebrc_by_sb.items():
        if ebrc_sb_num not in matched_ebrc_sbs:
            for _, ebrc_row in ebrc_group.iterrows():
                ebrc_ref = str(ebrc_row.get("ebrc_number", "") or ebrc_sb_num)
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
                    "sheets_update": False,  # eBRC-only rows; nothing to write to SB sheet
                })

    return results, summary


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
