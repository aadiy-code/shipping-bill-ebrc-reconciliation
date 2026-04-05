"""
FastAPI application entry point.
Serves the web UI and exposes the /reconcile endpoint.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from models import ReconciliationConfig, ReconciliationSummary
from reconciler import reconcile
from sheets_updater import update_sheet

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Shipping Bill ↔ eBRC Reconciliation",
    description=(
        "Upload a Shipping Bill file and an eBRC file to reconcile them "
        "and automatically update a Google Sheet."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static folder (HTML UI)
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "Shipping Bill ↔ eBRC Reconciliation API. POST /reconcile to start."})


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/reconcile")
async def reconcile_endpoint(
    # --- Files ---
    shipping_bill_file: UploadFile = File(..., description="Shipping Bill report (CSV or Excel)"),
    ebrc_file: UploadFile = File(..., description="eBRC report (CSV or Excel)"),
    # --- Google Sheets config ---
    spreadsheet_id: str = Form(...),
    sheet_tab_name: str = Form("Sheet1"),
    key_column: str = Form("Shipping Bill Number"),
    status_column: str = Form("Status"),
    remarks_column: str = Form("Remarks"),
    color_target_column: str = Form("Status"),
    # --- Reconciliation config ---
    amount_tolerance: float = Form(0.01),
    enable_grouped_matching: bool = Form(True),
    # --- Auth ---
    service_account_json: Optional[str] = Form(
        default="",
        description="Google service account JSON (leave blank to use GOOGLE_SERVICE_ACCOUNT_JSON env var)",
    ),
):
    """
    Run full reconciliation and update Google Sheet.

    Returns a JSON object with:
    - `summary`: aggregate counts
    - `audit_trail`: per-row detail
    """
    # Read file bytes
    try:
        sb_bytes = await shipping_bill_file.read()
        ebrc_bytes = await ebrc_file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded files: {exc}") from exc

    if not sb_bytes:
        raise HTTPException(status_code=400, detail="Shipping Bill file is empty.")
    if not ebrc_bytes:
        raise HTTPException(status_code=400, detail="eBRC file is empty.")

    config = ReconciliationConfig(
        spreadsheet_id=spreadsheet_id,
        sheet_tab_name=sheet_tab_name,
        key_column=key_column,
        status_column=status_column,
        remarks_column=remarks_column,
        color_target_column=color_target_column,
        amount_tolerance=amount_tolerance,
        enable_grouped_matching=enable_grouped_matching,
        service_account_json=service_account_json or "",
    )

    # --- Reconcile ---
    try:
        results, summary = reconcile(
            sb_bytes=sb_bytes,
            sb_filename=shipping_bill_file.filename or "sb_file",
            ebrc_bytes=ebrc_bytes,
            ebrc_filename=ebrc_file.filename or "ebrc_file",
            config=config,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during reconciliation")
        raise HTTPException(status_code=500, detail=f"Reconciliation failed: {exc}") from exc

    # --- Update Google Sheet ---
    sheets_error: Optional[str] = None
    try:
        results = update_sheet(results, config)
        # Propagate sheets_update_result back to audit trail
        result_map = {r["orig_sb_number"]: r.get("sheets_update_result", "") for r in results}
        for audit_row in summary.audit_trail:
            audit_row.sheets_update_result = result_map.get(
                audit_row.source_identifier, "NOT_ATTEMPTED"
            )
    except Exception as exc:
        logger.exception("Google Sheets update failed")
        sheets_error = str(exc)
        for audit_row in summary.audit_trail:
            audit_row.sheets_update_result = f"UPDATE_FAILED: {exc}"

    response_body = {
        "summary": {
            "total_processed": summary.total_processed,
            "exact_matches": summary.exact_matches,
            "partial_matches": summary.partial_matches,
            "mismatches": summary.mismatches,
            "missing_in_ebrc": summary.missing_in_ebrc,
            "missing_in_shipping_bill": summary.missing_in_shipping_bill,
            "duplicate_ambiguous": summary.duplicate_ambiguous,
            "errors": summary.errors,
        },
        "audit_trail": [
            {
                "source_identifier": a.source_identifier,
                "matched_identifier": a.matched_identifier,
                "status": a.status.value,
                "mismatch_reason": a.mismatch_reason,
                "remarks": a.remarks,
                "timestamp": a.timestamp.isoformat(),
                "sheets_update_result": a.sheets_update_result,
            }
            for a in summary.audit_trail
        ],
        "sheets_error": sheets_error,
    }

    return JSONResponse(content=response_body)


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
