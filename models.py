from pydantic import BaseModel, Field
from typing import Optional, List, Any
from enum import Enum
from datetime import datetime


class ReconciliationStatus(str, Enum):
    EXACT_MATCH = "EXACT_MATCH"
    PARTIAL_MATCH = "PARTIAL_MATCH"
    MISMATCH = "MISMATCH"
    MISSING_IN_EBRC = "MISSING_IN_EBRC"
    MISSING_IN_SHIPPING_BILL = "MISSING_IN_SHIPPING_BILL"
    DUPLICATE_MATCH = "DUPLICATE_MATCH"
    AMBIGUOUS_MATCH = "AMBIGUOUS_MATCH"
    ERROR = "ERROR"


STATUS_COLORS = {
    ReconciliationStatus.EXACT_MATCH:             {"red": 0.204, "green": 0.659, "blue": 0.325},  # green
    ReconciliationStatus.PARTIAL_MATCH:           {"red": 1.0,   "green": 0.851, "blue": 0.400},  # yellow
    ReconciliationStatus.MISMATCH:                {"red": 0.918, "green": 0.263, "blue": 0.208},  # red
    ReconciliationStatus.MISSING_IN_EBRC:         {"red": 0.918, "green": 0.263, "blue": 0.208},  # red
    ReconciliationStatus.MISSING_IN_SHIPPING_BILL:{"red": 0.918, "green": 0.263, "blue": 0.208},  # red
    ReconciliationStatus.DUPLICATE_MATCH:         {"red": 1.0,   "green": 0.596, "blue": 0.0},    # orange
    ReconciliationStatus.AMBIGUOUS_MATCH:         {"red": 1.0,   "green": 0.596, "blue": 0.0},    # orange
    ReconciliationStatus.ERROR:                   {"red": 0.741, "green": 0.741, "blue": 0.741},  # grey
}


class ReconciliationConfig(BaseModel):
    spreadsheet_id: str = Field(..., description="Google Sheet ID")
    sheet_tab_name: str = Field("Sheet1", description="Tab/worksheet name")
    key_column: str = Field("Shipping Bill Number", description="Column used to locate rows")
    status_column: str = Field("Status", description="Column to write reconciliation status")
    remarks_column: str = Field("Remarks", description="Column to write remarks")
    color_target_column: str = Field("Status", description="Column whose cell background color is updated")
    amount_tolerance: float = Field(0.01, description="Fractional tolerance for amount matching (0.01 = 1%)")
    enable_grouped_matching: bool = Field(True, description="Allow one SB to match multiple eBRC rows by sum")
    service_account_json: str = Field("", description="Google service account JSON string (leave blank to use env var)")


class AuditRow(BaseModel):
    source_identifier: str
    matched_identifier: Optional[str] = None
    status: ReconciliationStatus
    mismatch_reason: Optional[str] = None
    remarks: str = ""
    timestamp: datetime
    sheets_update_result: str = "NOT_ATTEMPTED"


class ReconciliationSummary(BaseModel):
    total_processed: int = 0
    exact_matches: int = 0
    partial_matches: int = 0
    mismatches: int = 0
    missing_in_ebrc: int = 0
    missing_in_shipping_bill: int = 0
    duplicate_ambiguous: int = 0
    errors: int = 0
    audit_trail: List[AuditRow] = []

    def increment(self, status: ReconciliationStatus) -> None:
        self.total_processed += 1
        if status == ReconciliationStatus.EXACT_MATCH:
            self.exact_matches += 1
        elif status == ReconciliationStatus.PARTIAL_MATCH:
            self.partial_matches += 1
        elif status == ReconciliationStatus.MISMATCH:
            self.mismatches += 1
        elif status == ReconciliationStatus.MISSING_IN_EBRC:
            self.missing_in_ebrc += 1
        elif status == ReconciliationStatus.MISSING_IN_SHIPPING_BILL:
            self.missing_in_shipping_bill += 1
        elif status in (ReconciliationStatus.DUPLICATE_MATCH, ReconciliationStatus.AMBIGUOUS_MATCH):
            self.duplicate_ambiguous += 1
        elif status == ReconciliationStatus.ERROR:
            self.errors += 1
