from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config.specs import ProductSpec


@dataclass(frozen=True, slots=True)
class ProductDetection:
    model_name: str | None
    prod_date: str
    detected_pn: str

    @property
    def report_date_code(self) -> str:
        digits = "".join(ch for ch in self.prod_date if ch.isdigit())
        if len(digits) >= 8:
            return digits[2:8]
        return digits or "unknown"

    @property
    def report_pn_code(self) -> str:
        cleaned = self.detected_pn.strip()
        if len(cleaned) >= 5:
            return cleaned[-5:]
        return cleaned or "unknown"

    @property
    def report_basename(self) -> str:
        return f"{self.report_date_code}_{self.report_pn_code}_REPORT"


@dataclass(frozen=True, slots=True)
class UploadedLog:
    df: pd.DataFrame
    sn_column: str
    limit_low: pd.Series
    limit_high: pd.Series
    test_data: pd.DataFrame


@dataclass(frozen=True, slots=True)
class MeasurementPoint:
    label: str
    display_value: str
    is_fail: bool


@dataclass(frozen=True, slots=True)
class ChannelAnalysis:
    mic_name: str
    status: str
    points: dict[str, MeasurementPoint]

    def point(self, label: str) -> MeasurementPoint:
        return self.points[label]


@dataclass(frozen=True, slots=True)
class SampleAnalysis:
    index: int
    serial_number: str
    channels: tuple[ChannelAnalysis, ...]

    @property
    def is_issue(self) -> bool:
        return any(channel.status != "Normal" for channel in self.channels)

    @property
    def is_pure_normal(self) -> bool:
        return all(channel.status == "Normal" for channel in self.channels)

    @property
    def is_defect(self) -> bool:
        defect_statuses = {"No Signal", "Curved Out", "Nan"}
        return any(channel.status in defect_statuses for channel in self.channels)

    def primary_defect_status(self, priority_order: tuple[str, ...]) -> str:
        statuses = [channel.status for channel in self.channels if channel.status != "Normal"]
        for defect_type in priority_order:
            if defect_type in statuses:
                return defect_type
        return statuses[0] if statuses else "Normal"


@dataclass(slots=True)
class ChannelStatistics:
    mic_name: str
    pass_count: int = 0
    fail_count: int = 0
    values_by_label: dict[str, list[float]] = field(default_factory=lambda: {"200Hz": [], "1kHz": [], "4kHz": []})

    def record(self, status: str, point_values: dict[str, float]) -> None:
        for label in self.values_by_label:
            self.values_by_label[label].append(point_values.get(label, np.nan))
        if status == "Normal":
            self.pass_count += 1
            return
        self.fail_count += 1

    def summary_metrics(self, stats_indices: tuple[int, ...], total_qty: int, label: str = "1kHz") -> tuple[float, float, float, float, float]:
        values = np.array(self.values_by_label[label])[list(stats_indices)]
        values = values[~np.isnan(values)]
        if len(values) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        yield_rate = (self.pass_count / total_qty) if total_qty > 0 else 0.0
        return values.min(), values.max(), values.mean(), values.std(), yield_rate


@dataclass(frozen=True, slots=True)
class DefectSummary:
    counts: dict[str, int]
    total_failure_samples: int

    def rate_for(self, defect_type: str, total_qty: int) -> float:
        if total_qty <= 0:
            return 0.0
        return self.counts[defect_type] / total_qty * 100

    def total_rate(self, total_qty: int) -> float:
        if total_qty <= 0:
            return 0.0
        return self.total_failure_samples / total_qty * 100


@dataclass(frozen=True, slots=True)
class AnalysisReport:
    detection: ProductDetection
    product_spec: ProductSpec
    uploaded_log: UploadedLog
    samples: tuple[SampleAnalysis, ...]
    issue_indices: tuple[int, ...]
    stats_indices: tuple[int, ...]
    plotting_normal_indices: tuple[int, ...]
    channel_statistics: tuple[ChannelStatistics, ...]
    defect_summary: DefectSummary

    @property
    def total_qty(self) -> int:
        return len(self.uploaded_log.test_data)

    @property
    def total_fail(self) -> int:
        return len(self.issue_indices)

    @property
    def total_pass(self) -> int:
        return self.total_qty - self.total_fail

    @property
    def yield_percentage(self) -> float:
        if self.total_qty <= 0:
            return 0.0
        return self.total_pass / self.total_qty * 100

    def sample_by_index(self, index: int) -> SampleAnalysis:
        return self.samples[index]

    def selected_samples(self, indices: list[int] | tuple[int, ...]) -> tuple[SampleAnalysis, ...]:
        return tuple(self.sample_by_index(index) for index in indices)

    def channel_statistics_by_name(self) -> dict[str, ChannelStatistics]:
        return {stat.mic_name: stat for stat in self.channel_statistics}
