"""
Google Sheets update layer.
Authenticates with a service account, locates rows by key column,
writes status/remarks, and applies background colours.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from models import ReconciliationConfig, ReconciliationStatus, STATUS_COLORS

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def _get_credentials(service_account_json: str):
    """
    Resolve service account credentials from:
    1. The supplied JSON string (from the API request).
    2. The GOOGLE_SERVICE_ACCOUNT_JSON environment variable.
    Raises ValueError if neither is available.
    """
    raw = service_account_json.strip() if service_account_json else ""
    if not raw:
        raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if not raw:
        raise ValueError(
            "Google service account credentials not provided. "
            "Supply them in the form or set GOOGLE_SERVICE_ACCOUNT_JSON env var."
        )
    try:
        info = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid service account JSON: {exc}") from exc

    return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)


def _build_service(service_account_json: str):
    creds = _get_credentials(service_account_json)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


# ---------------------------------------------------------------------------
# Sheet helpers
# ---------------------------------------------------------------------------

def _col_letter(index: int) -> str:
    """Convert 0-based column index to spreadsheet column letter (A, B, ... Z, AA, ...)."""
    result = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _read_sheet(service, spreadsheet_id: str, tab: str) -> Tuple[List[List[str]], Dict[str, int]]:
    """
    Read entire sheet, return (rows, header_index_map).
    header_index_map maps normalised column name -> 0-based column index.
    """
    range_name = f"'{tab}'"
    result = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=range_name)
        .execute()
    )
    rows: List[List[str]] = result.get("values", [])
    if not rows:
        raise ValueError(f"Sheet '{tab}' appears to be empty (no data found).")

    header_row = rows[0]
    header_map: Dict[str, int] = {
        str(h).strip().lower(): i for i, h in enumerate(header_row)
    }
    return rows, header_map


def _col_index(header_map: Dict[str, int], column_name: str) -> int:
    """Resolve column name to index; raises ValueError if not found."""
    key = column_name.strip().lower()
    if key not in header_map:
        raise ValueError(
            f"Column '{column_name}' not found in sheet. "
            f"Available columns: {list(header_map.keys())}"
        )
    return header_map[key]


# ---------------------------------------------------------------------------
# Build batchUpdate requests
# ---------------------------------------------------------------------------

def _cell_color_request(
    sheet_id: int,
    row_index: int,      # 0-based
    col_index: int,      # 0-based
    color: Dict[str, float],
) -> Dict:
    return {
        "repeatCell": {
            "range": {
                "sheetId": sheet_id,
                "startRowIndex": row_index,
                "endRowIndex": row_index + 1,
                "startColumnIndex": col_index,
                "endColumnIndex": col_index + 1,
            },
            "cell": {
                "userEnteredFormat": {
                    "backgroundColor": {
                        "red": color["red"],
                        "green": color["green"],
                        "blue": color["blue"],
                        "alpha": 1.0,
                    }
                }
            },
            "fields": "userEnteredFormat.backgroundColor",
        }
    }


def _cell_value_request(
    sheet_id: int,
    row_index: int,
    col_index: int,
    value: str,
) -> Dict:
    return {
        "updateCells": {
            "range": {
                "sheetId": sheet_id,
                "startRowIndex": row_index,
                "endRowIndex": row_index + 1,
                "startColumnIndex": col_index,
                "endColumnIndex": col_index + 1,
            },
            "rows": [
                {
                    "values": [
                        {"userEnteredValue": {"stringValue": str(value)}}
                    ]
                }
            ],
            "fields": "userEnteredValue",
        }
    }


# ---------------------------------------------------------------------------
# Main update function
# ---------------------------------------------------------------------------

def update_sheet(
    results: List[Dict],
    config: ReconciliationConfig,
) -> List[Dict]:
    """
    Update Google Sheet rows based on reconciliation results.
    Modifies each result dict in-place with 'sheets_update_result'.
    Returns the modified results list.
    """
    try:
        service = _build_service(config.service_account_json)
    except ValueError as exc:
        logger.error("Sheets auth failed: %s", exc)
        for r in results:
            if r.get("sheets_update"):
                r["sheets_update_result"] = f"AUTH_ERROR: {exc}"
        return results

    try:
        rows, header_map = _read_sheet(service, config.spreadsheet_id, config.sheet_tab_name)
    except (HttpError, ValueError) as exc:
        logger.error("Cannot read sheet: %s", exc)
        for r in results:
            if r.get("sheets_update"):
                r["sheets_update_result"] = f"SHEET_READ_ERROR: {exc}"
        return results

    # Resolve column indices
    try:
        key_col_idx = _col_index(header_map, config.key_column)
        status_col_idx = _col_index(header_map, config.status_column)
        remarks_col_idx = _col_index(header_map, config.remarks_column)
        color_col_idx = _col_index(header_map, config.color_target_column)
    except ValueError as exc:
        logger.error("Column resolution error: %s", exc)
        for r in results:
            if r.get("sheets_update"):
                r["sheets_update_result"] = f"COLUMN_ERROR: {exc}"
        return results

    # Get sheet internal ID (needed for batchUpdate color requests)
    try:
        spreadsheet_meta = (
            service.spreadsheets().get(spreadsheetId=config.spreadsheet_id).execute()
        )
        sheet_id = None
        for sheet in spreadsheet_meta.get("sheets", []):
            if sheet["properties"]["title"] == config.sheet_tab_name:
                sheet_id = sheet["properties"]["sheetId"]
                break
        if sheet_id is None:
            raise ValueError(f"Tab '{config.sheet_tab_name}' not found in spreadsheet metadata.")
    except (HttpError, ValueError) as exc:
        logger.error("Cannot get sheet metadata: %s", exc)
        for r in results:
            if r.get("sheets_update"):
                r["sheets_update_result"] = f"METADATA_ERROR: {exc}"
        return results

    # Build a lookup: normalised key value -> list of sheet row indices (1-based data rows)
    key_lookup: Dict[str, List[int]] = {}
    for row_idx, row in enumerate(rows[1:], start=1):  # skip header
        if len(row) > key_col_idx:
            cell_val = str(row[key_col_idx]).strip().upper()
            key_lookup.setdefault(cell_val, []).append(row_idx)

    # Accumulate batchUpdate requests
    value_requests: List[Dict] = []
    format_requests: List[Dict] = []
    updates_map: Dict[int, Tuple[str, str]] = {}  # row_idx -> (status_str, remarks_str)

    for result in results:
        if not result.get("sheets_update"):
            result["sheets_update_result"] = "NOT_APPLICABLE"
            continue

        orig_key = str(result.get("orig_sb_number", result.get("sb_number", ""))).strip().upper()
        status: ReconciliationStatus = result["status"]
        remarks: str = result.get("remarks", "")

        matching_rows = key_lookup.get(orig_key, [])
        if not matching_rows:
            result["sheets_update_result"] = f"KEY_NOT_FOUND: '{orig_key}' not in sheet"
            continue

        for row_idx in matching_rows:
            updates_map[row_idx] = (status.value, remarks)

    # Build requests in sorted row order (avoids conflicts)
    for row_idx, (status_str, remarks_str) in sorted(updates_map.items()):
        value_requests.append(
            _cell_value_request(sheet_id, row_idx, status_col_idx, status_str)
        )
        value_requests.append(
            _cell_value_request(sheet_id, row_idx, remarks_col_idx, remarks_str)
        )
        color = STATUS_COLORS.get(
            ReconciliationStatus(status_str),
            {"red": 0.741, "green": 0.741, "blue": 0.741},
        )
        format_requests.append(
            _cell_color_request(sheet_id, row_idx, color_col_idx, color)
        )

    all_requests = value_requests + format_requests
    if not all_requests:
        for r in results:
            if r.get("sheets_update") and "sheets_update_result" not in r:
                r["sheets_update_result"] = "KEY_NOT_FOUND"
        return results

    try:
        service.spreadsheets().batchUpdate(
            spreadsheetId=config.spreadsheet_id,
            body={"requests": all_requests},
        ).execute()

        # Mark results that had a successful key match
        updated_originals = set()
        for row_idx in updates_map:
            pass  # already batched

        for result in results:
            if result.get("sheets_update") and "sheets_update_result" not in result:
                orig_key = str(result.get("orig_sb_number", result.get("sb_number", ""))).strip().upper()
                if orig_key in key_lookup:
                    result["sheets_update_result"] = "SUCCESS"

    except HttpError as exc:
        logger.error("batchUpdate failed: %s", exc)
        for result in results:
            if result.get("sheets_update") and "sheets_update_result" not in result:
                result["sheets_update_result"] = f"UPDATE_ERROR: {exc}"

    return results
