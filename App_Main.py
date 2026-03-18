import base64
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import streamlit as st

from config.limits import LIMIT_POLICY
from config.product_config import PRODUCT_CATALOG
from core.application import AnalysisApplicationService
from core.parser import get_freq_values
from export.excel_report import ExcelReportBuilder

# 1. 페이지 설정 및 UI 상수
st.set_page_config(page_title="MIC Analysis Tool v1.0", page_icon="🎙️", layout="wide")
FIG_WIDTH, PLOT_HEIGHT = 14, 6
FONT_SIZE_TITLE, FONT_SIZE_AXIS = 16, 12
APPLICATION_SERVICE = AnalysisApplicationService(PRODUCT_CATALOG, LIMIT_POLICY)

# --- [상단 헤더] ---
col_head1, col_head2 = st.columns([4, 1], vertical_alignment="center")
with col_head1:
    st.markdown("<h1>🎙️ MIC Analysis Tool v1.0 <span style='font-size: 16px; color: gray; font-weight: normal; margin-left: 10px;'>( Provided by JW Lee, JJ Kim )</span></h1>", unsafe_allow_html=True)
with col_head2:
    if os.path.exists("logo.png"): st.image("logo.png", width=300)
st.markdown("---")

# 2. [유틸리티 함수]

def create_fr_plot(report, show_normal, highlight_indices, for_excel=False):
    product_spec = report.product_spec
    uploaded_log = report.uploaded_log
    df = uploaded_log.df
    current_test_data = uploaded_log.test_data
    limit_low = uploaded_log.limit_low
    limit_high = uploaded_log.limit_high

    num_draw = 3 if for_excel else len(product_spec.channels)
    fig, axes = plt.subplots(num_draw, 1, figsize=(FIG_WIDTH, PLOT_HEIGHT * num_draw))
    if num_draw == 1: axes = [axes]
    if for_excel: fig.patch.set_linewidth(0)
    for i in range(num_draw):
        ax = axes[i]
        if i < len(product_spec.channels):
            ch = product_spec.channels[i]
            cols = df.columns[ch.column_range]
            freqs = get_freq_values(cols)
            ylim, color, unit = ((-30, 0), 'green', 'dbV') if ch.mic_type == 'analog' else ((-45, -25), 'blue', 'dbFS')
            ax.set_xscale('log'); ax.set_xticks([50, 100, 200, 1000, 4000, 10000, 14000])
            ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter()); ax.minorticks_off()
            if show_normal:
                for n in report.plotting_normal_indices:
                    ax.plot(freqs, pd.to_numeric(current_test_data.loc[n, cols], errors='coerce'), color=color, alpha=0.7, lw=1.2)
            for h in highlight_indices: ax.plot(freqs, pd.to_numeric(current_test_data.loc[h, cols], errors='coerce'), color='red', lw=2.5)
            ax.plot(freqs, pd.to_numeric(limit_low[cols], errors='coerce'), 'k--', lw=1.2); ax.plot(freqs, pd.to_numeric(limit_high[cols], errors='coerce'), 'k--', lw=1.2)
            ax.set_title(ch.name, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
            ax.set_ylabel(f'Response ({unit})', fontsize=FONT_SIZE_AXIS); ax.set_ylim(ylim); ax.grid(True, alpha=0.4)
        else: ax.axis('off')
    plt.tight_layout(); return fig

def plot_bell_curve_set(report, sel_idx, for_excel=False):
    product_spec = report.product_spec
    uploaded_log = report.uploaded_log
    df = uploaded_log.df
    test_data = uploaded_log.test_data
    num_draw = 3 if for_excel else len(product_spec.channels)
    fig, axes = plt.subplots(num_draw, 1, figsize=(FIG_WIDTH, PLOT_HEIGHT * num_draw))
    if num_draw == 1: axes = [axes]
    if for_excel: fig.patch.set_linewidth(0)
    for i in range(num_draw):
        ax = axes[i]
        if i < len(product_spec.channels):
            ch = product_spec.channels[i]
            col_idx = np.argmin(np.abs(np.array(get_freq_values(df.columns[ch.column_range])) - 1000))
            v_all = pd.to_numeric(test_data[df.columns[ch.column_range][col_idx]], errors='coerce')
            v_clean = v_all.iloc[list(report.stats_indices)].dropna()
            lcl, ucl = LIMIT_POLICY.cpk_limit_for(ch.mic_type)
            if len(v_clean) >= 2:
                mu, std = v_clean.mean(), v_clean.std()
                cpk = min((ucl-mu)/(3*std), (mu-lcl)/(3*std)) if std > 0 else 0
                x_r = np.linspace(lcl - 2, ucl + 2, 200); p = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_r - mu) / std)**2)
                ax.plot(x_r, p, 'k', lw=2.5, alpha=0.7); ax.fill_between(x_r, p, color='gray', alpha=0.1)
                ax.axvline(lcl, color='blue', ls='--', lw=1.5); ax.axvline(ucl, color='red', ls='--', lw=1.5)
                if sel_idx:
                    sel_v = v_all.iloc[list(sel_idx)].dropna()
                    for v in sel_v:
                        if std > 0:
                            p_v = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((v - mu) / std)**2)
                            ax.scatter(v, p_v, color='red', s=200, zorder=5, edgecolors='white', linewidth=1.5)
                ax.text(0.95, 0.75, "Cpk: " + str(round(cpk, 2)), transform=ax.transAxes, ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=FONT_SIZE_AXIS, fontweight='bold')
                ax.set_title(f"{ch.name} - Distribution", fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15); ax.set_xlim(lcl-2, ucl+2); ax.grid(True, alpha=0.2)
        else: ax.axis('off')
    plt.tight_layout(); return fig

def get_base64_image(img_path):
    if os.path.exists(img_path):
        with open(img_path, "rb") as f: data = f.read()
        return base64.b64encode(data).decode()
    return None

# 4. [메인 프로세스]
uploaded_file = st.sidebar.file_uploader("CSV 로그 파일을 업로드하세요.", type=['csv'])

if uploaded_file:
    df = APPLICATION_SERVICE.read_dataframe(uploaded_file)

    if df is not None:
        detection = APPLICATION_SERVICE.detect_product(df)
        model_list = list(PRODUCT_CATALOG.model_names())
        model_type = st.sidebar.selectbox("제품 모델 선택", options=model_list, index=model_list.index(detection.model_name) if detection.model_name in model_list else 0)
        product_spec = PRODUCT_CATALOG.get(model_type)
        st.sidebar.markdown("---")
        
        st.sidebar.header("✔️ 정상 시료 설정")
        show_normal = st.sidebar.checkbox("정상 시료 FR 표시", value=True)

        report = APPLICATION_SERVICE.analyze(df, model_type, detection)
        test_data = report.uploaded_log.test_data

        if len(test_data) > 0:
            issue_indices = report.issue_indices
            channel_statistics = report.channel_statistics_by_name()
            limit_low = report.uploaded_log.limit_low
            limit_high = report.uploaded_log.limit_high
            detected_pn = report.detection.detected_pn
            prod_date = report.detection.prod_date
            total_qty = report.total_qty
            total_fail = report.total_fail
            total_pass = report.total_pass
            yield_val = report.yield_percentage
            defect_counts = report.defect_summary.counts
            total_f_ch = report.defect_summary.total_failure_channels

            # --- [UI Dashboard] ---
            st.subheader("📝 Production Dashboard")
            d1, d2, d3 = st.columns([1.2, 1.3, 2.5])
            
            with d1: st.markdown(f"**Model P/N:** `{detected_pn}`\n\n**Prod. Date:** `{prod_date}`\n\n**Quantity:** `{len(test_data)} EA`")
            with d2:
                st.markdown(f"""
                <div style="display: flex; gap: 8px;">
                    <div style="background-color: #f8f9fa; padding: 10px 15px; border-radius: 10px; border-left: 5px solid #28a745; flex: 1;">
                        <p style="margin:0; font-size:11px; color:#6c757d; font-weight:bold;">PASS</p>
                        <p style="margin:0; font-size:24px; font-weight:800; color:#28a745;">{total_pass}</p>
                    </div>
                    <div style="background-color: #f8f9fa; padding: 10px 15px; border-radius: 10px; border-left: 5px solid #dc3545; flex: 1;">
                        <p style="margin:0; font-size:11px; color:#6c757d; font-weight:bold;">FAIL</p>
                        <p style="margin:0; font-size:24px; font-weight:800; color:#dc3545;">{total_fail}</p>
                    </div>
                </div>
                <div style="margin-top: 10px;">
                    <p style="margin-bottom: 0px; font-weight: bold; font-size: 20px;">Overall Yield: {yield_val:.1f}%</p>
                    <div style="width: 100%; background-color: #e0e0e0; border-radius: 5px; border: 1px solid #bdc3c7; margin-top: 2px;">
                        <div style="width: {yield_val}%; background-color: #2ecc71; height: 12px; border-radius: 4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with d3:
                s_html = """<table style="width:100%; border-collapse:collapse; border:1px solid #bdc3c7; font-size:13px; text-align:center;">
                <thead style="background-color:#F2F2F2; font-weight:bold;">
                <tr><th rowspan="2" style="border:1px solid #bdc3c7; padding:8px;">MIC</th><th colspan="7" style="border:1px solid #bdc3c7; padding:8px;">Statistics</th></tr>
                <tr><th style="border:1px solid #bdc3c7; padding:5px;">Pass</th><th style="border:1px solid #bdc3c7; padding:5px;">Fail</th><th style="border:1px solid #bdc3c7; padding:5px;">Yield</th><th style="border:1px solid #bdc3c7; padding:5px;">Min</th><th style="border:1px solid #bdc3c7; padding:5px;">Max</th><th style="border:1px solid #bdc3c7; padding:5px;">Avg</th><th style="border:1px solid #bdc3c7; padding:5px;">Stdev</th></tr>
                </thead><tbody>"""
                for ch_n, stat in channel_statistics.items():
                    v_min, v_max, v_avg, v_std, yld = stat.summary_metrics(report.stats_indices, total_qty)
                    s_html += f"<tr><td style='border:1px solid #bdc3c7; padding:5px; font-weight:bold; background-color:#F9F9F9;'>{ch_n}</td>"
                    s_html += f"<td style='border:1px solid #bdc3c7; padding:5px;'>{stat.pass_count}</td><td style='border:1px solid #bdc3c7; padding:5px;'>{stat.fail_count}</td>"
                    s_html += f"<td style='border:1px solid #bdc3c7; padding:5px;'>{yld}</td><td style='border:1px solid #bdc3c7; padding:5px;'>{v_min:.3f}</td>"
                    s_html += f"<td style='border:1px solid #bdc3c7; padding:5px;'>{v_max:.3f}</td><td style='border:1px solid #bdc3c7; padding:5px;'>{v_avg:.3f}</td><td style='border:1px solid #bdc3c7; padding:5px;'>{v_std:.3f}</td></tr>"
                s_html += "</tbody></table>"
                st.markdown(s_html, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # [탭 확장: 불량 유형 요약 탭 추가]
            tab_fr, tab_dist, tab_detail, tab_summary = st.tabs(["📈 주파수 응답 (FR)", "📉 정규분포 (Cpk)", "🔍 결함 시료 상세", "📊 불량 유형 요약"])
            
            # 사이드바 체크박스 (전체 선택 기능)
            st.sidebar.markdown("---"); st.sidebar.header("❌️ 결함 시료 선택")
            def on_all_select_change():
                for i in issue_indices: st.session_state[f"ch_{i}"] = st.session_state.all_sel_trigger
            st.sidebar.checkbox("전체 선택", key="all_sel_trigger", on_change=on_all_select_change)
            
            sel_idx = []
            for i in issue_indices:
                sample = report.sample_by_index(i)
                if st.sidebar.checkbox(f"SN: {sample.serial_number}", key=f"ch_{i}"): sel_idx.append(i)

            with tab_fr: st.pyplot(create_fr_plot(report, show_normal, sel_idx))
            with tab_dist: st.pyplot(plot_bell_curve_set(report, sel_idx))
            
            with tab_detail:
                # 공정 관리 한계 테이블 (s_html과 동일 스타일 적용)
                st.markdown("⚠️ **공정 관리 한계 (Process Control Limit)**", unsafe_allow_html=True)
                ref_html = """
                <table style="width:100%; border-collapse:collapse; border:1px solid #bdc3c7; font-size:13px; text-align:center; margin-bottom:25px;">
                    <thead style="background-color:#F2F2F2; font-weight:bold;">
                        <tr><th rowspan="2" style="border:1px solid #bdc3c7; padding:8px;">MIC Type</th><th rowspan="2" style="border:1px solid #bdc3c7; padding:8px;">Limit</th><th colspan="3" style="border:1px solid #bdc3c7; padding:5px;">Frequency Response</th><th style="border:1px solid #bdc3c7; padding:5px;">THD</th></tr>
                        <tr><th style="border:1px solid #bdc3c7; padding:5px;">200Hz</th><th style="border:1px solid #bdc3c7; padding:5px;">1kHz</th><th style="border:1px solid #bdc3c7; padding:5px;">4kHz</th><th style="border:1px solid #bdc3c7; padding:5px;">1kHz</th></tr>
                    </thead>
                    <tbody>
                        <tr><td rowspan="2" style="border:1px solid #bdc3c7; padding:5px; font-weight:bold; background-color:#F9F9F9;">Digital MIC</td><td style="border:1px solid #bdc3c7; padding:5px; background-color:#F9F9F9;">UCL</td><td style="border:1px solid #bdc3c7; padding:5px;">-35</td><td style="border:1px solid #bdc3c7; padding:5px;">-36</td><td style="border:1px solid #bdc3c7; padding:5px;">-35</td><td style="border:1px solid #bdc3c7; padding:5px;">0.5</td></tr>
                        <tr><td style="border:1px solid #bdc3c7; padding:5px; background-color:#F9F9F9;">LCL</td><td style="border:1px solid #bdc3c7; padding:5px;">-39</td><td style="border:1px solid #bdc3c7; padding:5px;">-38</td><td style="border:1px solid #bdc3c7; padding:5px;">-39</td><td style="border:1px solid #bdc3c7; padding:5px;">-</td></tr>
                        <tr><td rowspan="2" style="border:1px solid #bdc3c7; padding:5px; font-weight:bold; background-color:#F9F9F9;">Analog MIC</td><td style="border:1px solid #bdc3c7; padding:5px; background-color:#F9F9F9;">UCL</td><td style="border:1px solid #bdc3c7; padding:5px;">-14.5</td><td style="border:1px solid #bdc3c7; padding:5px;">-9</td><td style="border:1px solid #bdc3c7; padding:5px;">-8</td><td style="border:1px solid #bdc3c7; padding:5px;">1.0</td></tr>
                        <tr><td style="border:1px solid #bdc3c7; padding:5px; background-color:#F9F9F9;">LCL</td><td style="border:1px solid #bdc3c7; padding:5px;">-18.5</td><td style="border:1px solid #bdc3c7; padding:5px;">-11</td><td style="border:1px solid #bdc3c7; padding:5px;">-12</td><td style="border:1px solid #bdc3c7; padding:5px;">-</td></tr>
                    </tbody>
                </table>
                """
                st.markdown(ref_html, unsafe_allow_html=True)
                
                if sel_idx:
                    for idx in sel_idx:
                        sample = report.sample_by_index(idx)
                        st.markdown(f"📄 **SN: {sample.serial_number}**", unsafe_allow_html=True)
                        p_html = """<table style="width:100%; border-collapse:collapse; border:1px solid #bdc3c7; font-size:13px; text-align:center; margin-bottom:20px;">
                        <thead style="background-color:#F2F2F2; font-weight:bold;">
                        <tr><th rowspan="3" style="border:1px solid #bdc3c7; padding:8px;">MIC</th><th colspan="5" style="border:1px solid #bdc3c7; padding:8px;">Parameter</th></tr>
                        <tr><th colspan="3" style="border:1px solid #bdc3c7; padding:5px;">Frequency Response</th><th style="border:1px solid #bdc3c7; padding:5px;">THD</th><th rowspan="2" style="border:1px solid #bdc3c7; padding:5px;">Status</th></tr>
                        <tr><th style="border:1px solid #bdc3c7; padding:5px;">200Hz</th><th style="border:1px solid #bdc3c7; padding:5px;">1kHz</th><th style="border:1px solid #bdc3c7; padding:5px;">4kHz</th><th style="border:1px solid #bdc3c7; padding:5px;">1kHz</th></tr>
                        </thead><tbody>"""
                        for channel in sample.channels:
                            p_html += f"<tr><td style='border:1px solid #bdc3c7; padding:5px; font-weight:bold; background-color:#F9F9F9;'>{channel.mic_name}</td>"
                            for label in ["200Hz", "1kHz", "4kHz", "THD"]:
                                point = channel.point(label)
                                color = "color:red; font-weight:bold;" if point.is_fail else ""
                                p_html += f"<td style='border:1px solid #bdc3c7; padding:5px; {color}'>{point.display_value}</td>"
                            status_style = "color:red; font-weight:bold;" if channel.status in LIMIT_POLICY.defect_types else ""
                            p_html += f"<td style='border:1px solid #bdc3c7; padding:5px; {status_style}'>{channel.status}</td></tr>"
                        p_html += "</tbody></table>"
                        st.markdown(p_html, unsafe_allow_html=True)
                else: st.warning("사이드바에서 결함 시료를 선택하여 상세 데이터를 확인하세요.")

            # --- [네 번째 탭 구현: 리스트 방식 + 한 줄 합치기(Minify) + 전체 대비 비율 계산] ---
            with tab_summary:
                st.subheader("📊 전체 결함 시료 불량 유형 요약")

                # HTML 조립 (들여쓰기 제거 및 한 줄 처리)
                html_parts = []
                html_parts.append("<table style='width:100%; border-collapse:collapse; border:1px solid #bdc3c7; font-size:13px; text-align:center;'>")
                html_parts.append("<thead style='background-color:#F2F2F2; font-weight:bold;'><tr>")
                html_parts.append("<th style='border:1px solid #bdc3c7; padding:8px;'>불량 유형 (Defect Type)</th>")
                html_parts.append("<th style='border:1px solid #bdc3c7; padding:8px;'>수량 (Quantity)</th>")
                html_parts.append("<th style='border:1px solid #bdc3c7; padding:8px;'>비율 (Rate %)</th>")
                html_parts.append("</tr></thead><tbody>")
                
                for t in LIMIT_POLICY.defect_types:
                    qty = defect_counts[t]
                    # [수정] 전체 수량(total_qty) 대비 비율 계산
                    rate = (qty / total_qty * 100) if total_qty > 0 else 0
                    html_parts.append("<tr>")
                    html_parts.append(f"<td style='border:1px solid #bdc3c7; padding:5px; font-weight:bold; background-color:#F9F9F9;'>{t}</td>")
                    html_parts.append(f"<td style='border:1px solid #bdc3c7; padding:5px;'>{qty} EA</td>")
                    html_parts.append(f"<td style='border:1px solid #bdc3c7; padding:5px;'>{rate:.1f}%</td>")
                    html_parts.append("</tr>")
                
                # Total 행 (전체 불량률)
                total_rate = (total_f_ch / total_qty * 100) if total_qty > 0 else 0
                html_parts.append("<tr style='background-color:#E2EFDA; font-weight:bold;'>")
                html_parts.append("<td style='border:1px solid #bdc3c7; padding:5px;'>Total Failure</td>")
                html_parts.append(f"<td style='border:1px solid #bdc3c7; padding:5px;'>{total_f_ch} EA</td>")
                html_parts.append(f"<td style='border:1px solid #bdc3c7; padding:5px;'>{total_rate:.1f}%</td>")
                html_parts.append("</tr></tbody></table>")
                
                st.markdown("".join(html_parts), unsafe_allow_html=True)
                st.caption(f"※ 비율(Rate)은 전체 검사 수량({total_qty} EA) 대비 발생 비율입니다.")

            st.sidebar.markdown("---") 
            img_b64 = get_base64_image("excel_icon.png")
            if img_b64:
                st.sidebar.markdown(f'<div style="display: flex; align-items: center; margin-bottom: 10px;"><img src="data:image/png;base64,{img_b64}" width="38" style="margin-right: 12px;"><span style="font-size: 24px; font-weight: 700; color: #31333f;">Excel Export</span></div>', unsafe_allow_html=True)
            else: st.sidebar.header("📊 Excel Export")
            excel_data = ExcelReportBuilder(report, LIMIT_POLICY, show_normal, sel_idx, create_fr_plot, plot_bell_curve_set).build()
            st.sidebar.download_button(label="📥 Download Report", data=excel_data, file_name=f"Report_{detected_pn}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
else: st.info("사이드바에서 CSV 로그 파일을 업로드하세요.")
