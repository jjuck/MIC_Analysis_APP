from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.limits import LIMIT_POLICY
from config.models import ChannelSpec, LimitPolicy, ProductSpec
from core.models import (
    AnalysisReport,
    ChannelAnalysis,
    ChannelStatistics,
    DefectSummary,
    MeasurementPoint,
    ProductDetection,
    SampleAnalysis,
    UploadedLog,
)
from core.parser import clean_sn, get_freq_values


@dataclass(frozen=True, slots=True)
class ChannelContext:
    channel_spec: ChannelSpec
    cols: pd.Index
    freqs: np.ndarray
    point_columns: dict[str, str]
    other_indices: tuple[int, ...]
    center_1k_col: str


class ChannelContextFactory:
    def __init__(self, limit_policy: LimitPolicy):
        self._limit_policy = limit_policy

    def build(self, df: pd.DataFrame, product_spec: ProductSpec) -> tuple[ChannelContext, ...]:
        contexts: list[ChannelContext] = []
        for channel in product_spec.channels:
            cols = df.columns[channel.column_range]
            freqs = np.array(get_freq_values(cols))
            point_indices = [int(np.argmin(np.abs(freqs - point))) for point in self._limit_policy.check_points]
            other_indices = tuple(i for i in range(len(cols)) if i not in point_indices)
            point_columns = {
                "200Hz": cols[point_indices[0]],
                "1kHz": cols[point_indices[1]],
                "4kHz": cols[point_indices[2]],
            }
            contexts.append(
                ChannelContext(
                    channel_spec=channel,
                    cols=cols,
                    freqs=freqs,
                    point_columns=point_columns,
                    other_indices=other_indices,
                    center_1k_col=cols[point_indices[1]],
                )
            )
        return tuple(contexts)


class ChannelClassifier:
    def __init__(self, limit_policy: LimitPolicy):
        self._limit_policy = limit_policy

    def classify(self, row: pd.Series, context: ChannelContext, limit_low: pd.Series, limit_high: pd.Series) -> str:
        values = pd.to_numeric(row[context.cols], errors="coerce")
        if values.isna().all():
            return "Nan"

        low = pd.to_numeric(limit_low[context.cols], errors="coerce")
        high = pd.to_numeric(limit_high[context.cols], errors="coerce")
        is_fail = values.isna() | (values < low) | (values > high)
        if not is_fail.any():
            return "Normal"

        if (values < self._limit_policy.no_signal_limit_for(context.channel_spec.mic_type)).any():
            return "No Signal"

        if not context.other_indices:
            return "Margin Out"

        if not is_fail.iloc[list(context.other_indices)].any():
            return "Margin Out"

        return "Curved Out"

    def build_channel_analysis(self, row: pd.Series, context: ChannelContext, limit_low: pd.Series, limit_high: pd.Series) -> ChannelAnalysis:
        points: dict[str, MeasurementPoint] = {}
        for label, col in context.point_columns.items():
            value = pd.to_numeric(row[col], errors="coerce")
            low = pd.to_numeric(limit_low[col], errors="coerce")
            high = pd.to_numeric(limit_high[col], errors="coerce")
            is_fail = (value < low or value > high) if not pd.isna(value) else False
            points[label] = MeasurementPoint(
                label=label,
                display_value=f"{value:.3f}" if not pd.isna(value) else "-",
                is_fail=is_fail,
            )

        if context.channel_spec.has_thd:
            thd_value = pd.to_numeric(row.iloc[context.channel_spec.thd_column_index], errors="coerce")
            thd_limit = self._limit_policy.thd_limit_for(context.channel_spec.mic_type)
            is_fail = (thd_value < 0 or thd_value > thd_limit) if not pd.isna(thd_value) else False
            points["THD"] = MeasurementPoint(
                label="THD",
                display_value=f"{thd_value:.3f}" if not pd.isna(thd_value) else "-",
                is_fail=is_fail,
            )
        else:
            points["THD"] = MeasurementPoint(label="THD", display_value="N/A", is_fail=False)

        status = self.classify(row, context, limit_low, limit_high)
        return ChannelAnalysis(mic_name=context.channel_spec.name, status=status, points=points)


class DefectSummaryService:
    def __init__(self, limit_policy: LimitPolicy):
        self._limit_policy = limit_policy

    def build(self, samples: tuple[SampleAnalysis, ...], issue_indices: tuple[int, ...]) -> DefectSummary:
        counts = {defect_type: 0 for defect_type in self._limit_policy.defect_types}
        total_failure_channels = 0

        for idx in issue_indices:
            for channel in samples[idx].channels:
                if channel.status in counts:
                    counts[channel.status] += 1
                    total_failure_channels += 1

        return DefectSummary(counts=counts, total_failure_channels=total_failure_channels)


class AnalysisService:
    def __init__(self, limit_policy: LimitPolicy):
        self._limit_policy = limit_policy
        self._context_factory = ChannelContextFactory(limit_policy)
        self._classifier = ChannelClassifier(limit_policy)
        self._defect_summary_service = DefectSummaryService(limit_policy)

    def analyze(self, df: pd.DataFrame, product_spec: ProductSpec, detection: ProductDetection) -> AnalysisReport:
        uploaded_log = self._build_uploaded_log(df)
        contexts = self._context_factory.build(df, product_spec)
        statistics = {channel.name: ChannelStatistics(channel.name) for channel in product_spec.channels}
        samples: list[SampleAnalysis] = []
        issue_indices: list[int] = []
        stats_indices: list[int] = []
        plotting_normal_indices: list[int] = []

        for idx, row in uploaded_log.test_data.iterrows():
            sample = self._build_sample_analysis(idx, row, contexts, uploaded_log, statistics)
            samples.append(sample)
            if sample.is_issue:
                issue_indices.append(idx)
            if sample.is_pure_normal:
                plotting_normal_indices.append(idx)
            if not sample.is_defect:
                stats_indices.append(idx)

        sample_tuple = tuple(samples)
        issue_index_tuple = tuple(issue_indices)
        defect_summary = self._defect_summary_service.build(sample_tuple, issue_index_tuple)
        return AnalysisReport(
            detection=detection,
            product_spec=product_spec,
            uploaded_log=uploaded_log,
            samples=sample_tuple,
            issue_indices=issue_index_tuple,
            stats_indices=tuple(stats_indices),
            plotting_normal_indices=tuple(plotting_normal_indices),
            channel_statistics=tuple(statistics.values()),
            defect_summary=defect_summary,
        )

    def _build_uploaded_log(self, df: pd.DataFrame) -> UploadedLog:
        sn_column = df.columns[3]
        limit_low = df.iloc[0]
        limit_high = df.iloc[1]
        raw_test_data = df.iloc[2:].dropna(subset=[sn_column])
        test_data = raw_test_data[raw_test_data[sn_column].astype(str).str.contains('/', na=False)].reset_index(drop=True)
        return UploadedLog(
            df=df,
            sn_column=sn_column,
            limit_low=limit_low,
            limit_high=limit_high,
            test_data=test_data,
        )

    def _build_sample_analysis(
        self,
        index: int,
        row: pd.Series,
        contexts: tuple[ChannelContext, ...],
        uploaded_log: UploadedLog,
        statistics: dict[str, ChannelStatistics],
    ) -> SampleAnalysis:
        channel_analyses: list[ChannelAnalysis] = []
        for context in contexts:
            channel_analysis = self._classifier.build_channel_analysis(
                row=row,
                context=context,
                limit_low=uploaded_log.limit_low,
                limit_high=uploaded_log.limit_high,
            )
            value_at_1k = pd.to_numeric(row[context.center_1k_col], errors="coerce")
            statistics[context.channel_spec.name].record(channel_analysis.status, value_at_1k)
            channel_analyses.append(channel_analysis)

        return SampleAnalysis(
            index=index,
            serial_number=clean_sn(row[uploaded_log.sn_column]),
            channels=tuple(channel_analyses),
        )


def analyze_dataframe(df, model_type, product_spec, detected_model, prod_date, detected_pn):
    detection = ProductDetection(model_name=detected_model, prod_date=prod_date, detected_pn=detected_pn)
    report = AnalysisService(LIMIT_POLICY).analyze(df, product_spec, detection)

    class AnalysisResult:
        def __init__(self, analysis_report: AnalysisReport):
            self.detection = analysis_report.detection
            self.report = analysis_report
            self.product_spec = analysis_report.product_spec
            self.df = analysis_report.uploaded_log.df
            self.test_data = analysis_report.uploaded_log.test_data
            self.limit_low = analysis_report.uploaded_log.limit_low
            self.limit_high = analysis_report.uploaded_log.limit_high
            self.sn_col = analysis_report.uploaded_log.sn_column
            self.issue_indices = list(analysis_report.issue_indices)
            self.stats_indices = list(analysis_report.stats_indices)
            self.plotting_normal_indices = list(analysis_report.plotting_normal_indices)
            self.total_qty = analysis_report.total_qty
            self.total_fail = analysis_report.total_fail
            self.total_pass = analysis_report.total_pass
            self.yield_val = analysis_report.yield_percentage
            self.detected_model = analysis_report.detection.model_name
            self.detected_pn = analysis_report.detection.detected_pn
            self.prod_date = analysis_report.detection.prod_date
            self.model_type = model_type
            self.config = product_spec
            self.sample_info = {
                sample.index: {
                    "sn": sample.serial_number,
                    "results": [
                        {
                            "MIC": channel.mic_name,
                            "Status": channel.status,
                            "points": {
                                label: {"val": point.display_value, "fail": point.is_fail}
                                for label, point in channel.points.items()
                            },
                        }
                        for channel in sample.channels
                    ],
                }
                for sample in analysis_report.samples
            }
            self.ch_stats_data = {
                stat.mic_name: {
                    "pass": stat.pass_count,
                    "fail": stat.fail_count,
                    "vals_1k": stat.values_at_1k,
                }
                for stat in analysis_report.channel_statistics
            }

    return AnalysisResult(report)


def get_defect_counts(sample_info, issue_indices):
    counts = {defect_type: 0 for defect_type in LIMIT_POLICY.defect_types}
    total_failure_channels = 0
    for idx in issue_indices:
        for channel_result in sample_info[idx]["results"]:
            status = channel_result.get("Status", "Normal")
            if status in counts:
                counts[status] += 1
                total_failure_channels += 1
    return counts, total_failure_channels
