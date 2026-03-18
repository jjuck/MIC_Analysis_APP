from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.models import LimitPolicy


class ExcelReportBuilder:
    def __init__(
        self,
        analysis_report,
        limit_policy: LimitPolicy,
        show_normal,
        selected_indices,
        create_fr_plot,
        plot_bell_curve_set,
    ):
        self._report = analysis_report
        self._limit_policy = limit_policy
        self._show_normal = show_normal
        self._selected_indices = tuple(selected_indices)
        self._create_fr_plot = create_fr_plot
        self._plot_bell_curve_set = plot_bell_curve_set
        self._channel_statistics = analysis_report.channel_statistics_by_name()
        self._workbook = None

        self._base_blue = {'bold': True, 'bg_color': '#DEEAF6', 'align': 'center', 'valign': 'vcenter', 'border': 1}
        self._base_green = {'bold': True, 'bg_color': '#E2EFDA', 'align': 'center', 'valign': 'vcenter', 'border': 1}
        self._base_thin = {'align': 'center', 'valign': 'vcenter', 'border': 1}
        self._base_yld_val = {'bold': True, 'font_size': 18, 'font_color': '#2E7D32', 'align': 'center', 'valign': 'vcenter', 'border': 1, 'num_format': '0.0%'}
        self._base_red_thin = {'align': 'center', 'valign': 'vcenter', 'border': 1, 'font_color': 'red', 'bold': True}
        self._base_sn_box = {'bold': True, 'bg_color': '#F2F2F2', 'top': 1, 'bottom': 1, 'align': 'left', 'valign': 'vcenter'}

    def build(self):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            self._workbook = writer.book
            self._write_report_sheets()
        return output.getvalue()

    def _get_fmt(self, base_dict, top=None, bottom=None, left=None, right=None):
        props = base_dict.copy()
        if top is not None:
            props['top'] = top
        if bottom is not None:
            props['bottom'] = bottom
        if left is not None:
            props['left'] = left
        if right is not None:
            props['right'] = right
        return self._workbook.add_format(props)

    def _write_report_sheets(self):
        ws1 = self._workbook.add_worksheet('📈 분석 리포트')
        self._write_dashboard(ws1, 86, is_sheet1=True)
        fig_fr = self._create_fr_plot(self._report, self._show_normal, self._selected_indices, for_excel=True)
        buf_f = io.BytesIO()
        fig_fr.savefig(buf_f, format='png', dpi=100)
        plt.close(fig_fr)
        ws1.insert_image('B7', 'fr.png', {'image_data': buf_f, 'x_scale': 0.41, 'y_scale': 0.35, 'x_offset': 10, 'y_offset': 10})
        fig_dist = self._plot_bell_curve_set(self._report, self._selected_indices, for_excel=True)
        buf_d = io.BytesIO()
        fig_dist.savefig(buf_d, format='png', dpi=100)
        plt.close(fig_dist)
        ws1.insert_image('H7', 'dist.png', {'image_data': buf_d, 'x_scale': 0.41, 'y_scale': 0.35, 'x_offset': 10, 'y_offset': 10})

        self._write_summary_section(ws1, 37, is_sheet1=True)
        ws1.merge_range(45, 1, 45, 13, '🔍 DETAILED FAILURE LOG', self._get_fmt(self._base_blue, left=2, right=2))
        c_l, c_r = 46, 46
        for i, idx in enumerate(self._selected_indices[:10]):
            if i % 2 == 0:
                c_l = self._write_failure_unit(ws1, c_l, 1, idx, 86)
            else:
                c_r = self._write_failure_unit(ws1, c_r, 8, idx, 86)

        ws2 = self._workbook.add_worksheet('🔍 결함상세')
        num_p = (len(self._selected_indices) + 1) // 2
        l_f_ws2 = max(23, 15 + (num_p * 8))
        self._write_dashboard(ws2, l_f_ws2)
        self._write_summary_section(ws2, 6)
        ws2.merge_range(14, 1, 14, 13, '🔍 DETAILED FAILURE LOG', self._get_fmt(self._base_blue, left=2, right=2))
        cl2, cr2 = 15, 15
        for i, idx in enumerate(self._selected_indices):
            if i % 2 == 0:
                cl2 = self._write_failure_unit(ws2, cl2, 1, idx, l_f_ws2)
            else:
                cr2 = self._write_failure_unit(ws2, cr2, 8, idx, l_f_ws2)

    def _write_dashboard(self, ws, last_row_idx=37, is_sheet1=False):
        ws.set_column('A:A', 3)
        ws.set_column('B:B', 15)
        ws.set_column('C:C', 22)
        ws.set_column('D:F', 10)
        ws.set_column('G:N', 11)

        ws.merge_range('B2:F2', '📝 PRODUCTION SUMMARY', self._get_fmt(self._base_blue, top=2, left=2, bottom=1, right=1))
        ws.merge_range('G2:N2', '📈 CHANNEL STATISTICS', self._get_fmt(self._base_green, top=2, right=2, bottom=1, left=0))

        sums = [
            ("Model Type", self._report.product_spec.model_name),
            ("Model P/N", self._report.detection.detected_pn),
            ("Prod. Date", self._report.detection.prod_date),
            ("Quantity", str(self._report.total_qty) + " EA"),
        ]
        for i, (k, v) in enumerate(sums):
            r = 2 + i
            ws.write(r, 1, k, self._get_fmt(self._base_blue, left=2, bottom=2 if r == 5 else 1, top=1, right=1))
            ws.write(r, 2, v, self._get_fmt(self._base_thin, bottom=2 if r == 5 else 1, top=1, left=1, right=1))

        ws.write(2, 3, 'PASS', self._get_fmt(self._base_blue, top=1, bottom=1, left=1, right=1))
        ws.merge_range('E3:F3', self._report.total_pass, self._get_fmt(self._base_thin, top=1, bottom=1, left=1, right=1))
        ws.write(3, 3, 'FAIL', self._get_fmt(self._base_blue, top=1, bottom=1, left=1, right=1))
        ws.merge_range('E4:F4', self._report.total_fail, self._get_fmt(self._base_thin, top=1, bottom=1, left=1, right=1))
        ws.merge_range('D5:D6', 'Yield', self._get_fmt(self._base_blue, bottom=2, top=1, left=1, right=1))
        ws.merge_range('E5:F6', self._report.yield_percentage / 100, self._get_fmt(self._base_yld_val, bottom=2, top=1, left=1, right=1))

        ws.write(2, 6, "MIC", self._get_fmt(self._base_green, top=1, bottom=1, left=0, right=1))
        heads = ["Pass", "Fail", "Yield", "Min", "Max", "Avg", "Stdev"]
        for i, h in enumerate(heads):
            ws.write(2, 7 + i, h, self._get_fmt(self._base_green, right=2 if 7 + i == 13 else 1, top=1, bottom=1, left=1))

        for r_idx in range(4):
            r = 3 + r_idx
            is_l = r == 5
            if r_idx < len(self._report.product_spec.channels):
                channel_name = self._report.product_spec.channels[r_idx].name
                stat = self._channel_statistics[channel_name]
                v_min, v_max, v_avg, v_std, yield_rate = stat.summary_metrics(self._report.stats_indices, self._report.total_qty)
                ws.write(r, 6, channel_name, self._get_fmt(self._base_thin, bottom=2 if is_l else 1, top=1, left=0, right=1))
                vals = [stat.pass_count, stat.fail_count, yield_rate, f"{v_min:.3f}", f"{v_max:.3f}", f"{v_avg:.3f}", f"{v_std:.3f}"]
                for i, val in enumerate(vals):
                    ws.write(r, 7 + i, val, self._get_fmt(self._base_thin, right=2 if 7 + i == 13 else 1, bottom=2 if is_l else 1, top=1, left=1))
            else:
                for c in range(6, 14):
                    ws.write_blank(r, c, "", self._get_fmt({'border': 0}, right=2 if c == 13 else 0, bottom=2 if is_l else 0, left=0 if c == 6 else 0))

        for r_f in range(6, last_row_idx - 1):
            if is_sheet1 and r_f == 36:
                for c in range(1, 14):
                    ws.write_blank(r_f, c, "", self._get_fmt({'border': 0}, bottom=2, left=2 if c == 1 else None, right=2 if c == 13 else None))
                continue
            ws.write_blank(r_f, 1, "", self._get_fmt({'border': 0}, left=2))
            ws.write_blank(r_f, 13, "", self._get_fmt({'border': 0}, right=2))

        ws.write_blank(last_row_idx - 1, 1, "", self._get_fmt({'border': 0}, left=2, bottom=2))
        for c_b in range(2, 13):
            ws.write_blank(last_row_idx - 1, c_b, "", self._get_fmt({'border': 0}, bottom=2))
        ws.write_blank(last_row_idx - 1, 13, "", self._get_fmt({'border': 0}, right=2, bottom=2))

    def _write_summary_section(self, ws, start_row, is_sheet1=False):
        top_l = 2 if is_sheet1 else 1
        ws.merge_range(start_row, 1, start_row, 6, '⚠️ 공정 관리 한계', self._get_fmt(self._base_blue, left=2, top=top_l))
        ws.merge_range(start_row + 1, 1, start_row + 2, 1, 'MIC Type', self._get_fmt(self._base_blue, left=2))
        ws.merge_range(start_row + 1, 2, start_row + 2, 2, 'Limit', self._get_fmt(self._base_blue))
        ws.merge_range(start_row + 1, 3, start_row + 1, 5, 'Frequency Response', self._get_fmt(self._base_blue))
        ws.write(start_row + 1, 6, 'THD', self._get_fmt(self._base_blue))
        for ci, h in enumerate(['200Hz', '1kHz', '4kHz', '1kHz']):
            ws.write(start_row + 2, 3 + ci, h, self._get_fmt(self._base_blue))

        for r_idx, row_data in enumerate(self._limit_policy.control_limit_specs):
            r = start_row + 3 + r_idx
            if r_idx % 2 == 0:
                ws.merge_range(r, 1, r + 1, 1, row_data[0], self._get_fmt(self._base_blue, left=2))
            ws.write(r, 2, row_data[1], self._get_fmt(self._base_blue))
            for c_idx, val in enumerate(row_data[2:]):
                ws.write(r, 3 + c_idx, val, self._get_fmt(self._base_thin))

        ws.merge_range(start_row, 8, start_row, 13, '📊 불량 유형 Summary', self._get_fmt(self._base_green, right=2, top=top_l))
        ws.merge_range(start_row + 1, 8, start_row + 1, 9, "Defect Type", self._get_fmt(self._base_green))
        ws.merge_range(start_row + 1, 10, start_row + 1, 11, "Quantity", self._get_fmt(self._base_green))
        ws.merge_range(start_row + 1, 12, start_row + 1, 13, "Rate (%)", self._get_fmt(self._base_green, right=2))

        for i, defect_type in enumerate(self._limit_policy.defect_types):
            r = start_row + 2 + i
            qty = self._report.defect_summary.counts[defect_type]
            rate = self._report.defect_summary.rate_for(defect_type, self._report.total_qty)
            ws.merge_range(r, 8, r, 9, defect_type, self._get_fmt(self._base_thin))
            ws.merge_range(r, 10, r, 11, qty, self._get_fmt(self._base_thin))
            ws.merge_range(r, 12, r, 13, f"{rate:.1f}%", self._get_fmt(self._base_thin, right=2))

        total_rate = self._report.defect_summary.total_rate(self._report.total_qty)
        ws.merge_range(start_row + 6, 8, start_row + 6, 9, "Total", self._get_fmt(self._base_green))
        ws.merge_range(start_row + 6, 10, start_row + 6, 11, self._report.defect_summary.total_failure_channels, self._get_fmt(self._base_thin))
        ws.merge_range(start_row + 6, 12, start_row + 6, 13, f"{total_rate:.1f}%", self._get_fmt(self._base_thin, right=2))

    def _write_failure_unit(self, ws, r_start, c_base, idx, total_last_row):
        sample = self._report.sample_by_index(idx)
        r = r_start
        if c_base == 1:
            ws.merge_range(r, 1, r, 2, sample.serial_number, self._get_fmt(self._base_sn_box, left=2, right=1))
            for c in range(3, 7):
                ws.write_blank(r, c, "", self._get_fmt({'border': 0}, bottom=1))
        else:
            ws.merge_range(r, 8, r, 10, sample.serial_number, self._get_fmt(self._base_sn_box, left=1, right=1))
            for c in range(11, 13):
                ws.write_blank(r, c, "", self._get_fmt({'border': 0}, bottom=1))
            ws.write_blank(r, 13, "", self._get_fmt({'border': 0}, right=2, bottom=1))
        r += 1
        s_r = r
        ws.merge_range(r, c_base, r + 2, c_base, 'MIC', self._get_fmt(self._base_blue, left=2 if c_base == 1 else 1))
        ws.merge_range(r, c_base + 1, r, c_base + 4, 'Parameter', self._get_fmt(self._base_blue))
        ws.write_blank(r, c_base + 5, "", self._get_fmt(self._base_blue, right=2 if c_base == 8 else 1))
        r += 1
        ws.merge_range(r, c_base + 1, r, c_base + 3, 'Frequency Response', self._get_fmt(self._base_blue))
        ws.write(r, c_base + 4, 'THD', self._get_fmt(self._base_blue))
        r += 1
        for ci, h in enumerate(['200Hz', '1kHz', '4kHz', '1kHz']):
            ws.write(r, c_base + 1 + ci, h, self._get_fmt(self._base_blue))
        ws.merge_range(s_r, c_base + 5, r, c_base + 5, 'Status', self._get_fmt(self._base_blue, right=2 if c_base == 8 else 1))
        r += 1
        rows_w = 0
        for channel in sample.channels:
            ws.write(r, c_base, channel.mic_name, self._get_fmt(self._base_thin, left=2 if c_base == 1 else 1))
            for ci, label in enumerate(["200Hz", "1kHz", "4kHz", "THD"]):
                point = channel.point(label)
                ws.write(r, c_base + 1 + ci, point.display_value, self._get_fmt(self._base_red_thin if point.is_fail else self._base_thin))
            ws.write(r, c_base + 5, channel.status, self._get_fmt(self._base_red_thin if channel.status in self._limit_policy.defect_types else self._base_thin, right=2 if c_base == 8 else 1))
            r += 1
            rows_w += 1
        for _ in range(3 - rows_w + 1):
            is_final = r == total_last_row - 1
            b_line = 2 if is_final else 0
            ws.write_blank(r, c_base, "", self._get_fmt({'border': 0}, left=2 if c_base == 1 else 0, bottom=b_line))
            for c in range(c_base + 1, c_base + 5):
                ws.write_blank(r, c, "", self._get_fmt({'border': 0}, bottom=b_line))
            ws.write_blank(r, c_base + 5, "", self._get_fmt({'border': 0}, right=2 if c_base == 8 else 0, bottom=b_line))
            r += 1
        return r


def generate_excel_report(
    analysis_report,
    limit_policy,
    show_normal,
    selected_indices,
    create_fr_plot,
    plot_bell_curve_set,
):
    return ExcelReportBuilder(
        analysis_report=analysis_report,
        limit_policy=limit_policy,
        show_normal=show_normal,
        selected_indices=selected_indices,
        create_fr_plot=create_fr_plot,
        plot_bell_curve_set=plot_bell_curve_set,
    ).build()
