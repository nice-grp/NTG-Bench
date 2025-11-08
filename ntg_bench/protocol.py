from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class ProtocolCheckResult:
    pass_ratio: float
    cleaned_df: pd.DataFrame


UDP_PORTS = {
    53,
    67,
    123,
    137,
    138,
    5353,
    1900,
    3544,
    8612,
    3702,
}
TCP_PORTS = {25, 80, 84, 443, 445, 587, 993, 8000, 8080, 8088}


def _ensure_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df:
            LOGGER.warning("Column '%s' missing; filling with zeros.", column)
            df[column] = 0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return df


def compute_protocol_compliance(df: pd.DataFrame) -> ProtocolCheckResult:
    """
    Apply the deterministic knowledge checks (DKC) defined in the paper.

    The routine filters out traffic flows that violate simple protocol rules
    (e.g., using TCP on UDP-only ports) and returns the pass ratio together
    with the cleaned dataframe for downstream evaluation.
    """

    if df.empty:
        return ProtocolCheckResult(pass_ratio=1.0, cleaned_df=df.copy())

    working = df.copy()
    if "proto" not in working.columns:
        raise KeyError("Column 'proto' is required for protocol compliance checks.")
    working["proto"] = working["proto"].astype(str)
    working = _ensure_numeric(working, ["dstport", "pkt", "byt", "td"])

    cond1 = working["dstport"].isin(UDP_PORTS) & working["proto"].str.contains("TCP")
    cond2 = working["dstport"].isin(TCP_PORTS) & working["proto"].str.contains("UDP")
    cond3 = (working["dstport"] == 0.0) & (~working["proto"].str.contains("ICMP"))
    cond4 = (working["pkt"] * 42 > working["byt"]) & (
        ~working["proto"].str.contains("TCP")
    )
    cond5 = (working["pkt"] * 54 > working["byt"]) & working["proto"].str.contains("TCP")
    cond6 = working["pkt"] * 65535 < working["byt"]
    cond7 = working["td"] < 0
    cond8 = (working["td"] == 0) & (working["pkt"] > 0)
    cond9 = (working["td"] > 0) & (working["pkt"] == 1)

    failed_mask = cond1 | cond2 | cond3 | cond4 | cond5 | cond6 | cond7 | cond8 | cond9
    failed_count = int(failed_mask.sum())
    total = len(working)
    passed = total - failed_count
    ratio = passed / total if total else 0.0

    LOGGER.info(
        "Protocol checks: total=%d, failed=%d, passed_ratio=%.4f",
        total,
        failed_count,
        ratio,
    )

    cleaned = working.loc[~failed_mask].reset_index(drop=True)
    # Preserve original column ordering when returning.
    cleaned = cleaned[df.columns]

    return ProtocolCheckResult(pass_ratio=float(ratio), cleaned_df=cleaned)
