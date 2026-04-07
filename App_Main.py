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
st.set_page_config(page_title="MIC Analysis Tool v1.1", page_icon="🎙️", layout="wide")
FIG_WIDTH, PLOT_HEIGHT = 14, 6
FONT_SIZE_TITLE, FONT_SIZE_AXIS = 16, 12
APPLICATION_SERVICE = AnalysisApplicationService(PRODUCT_CATALOG, LIMIT_POLICY)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@500;700;800&display=swap');

    :root {
        --mic-bg: #f6f7fb;
        --mic-surface: #eef1f7;
        --mic-card: rgba(255,255,255,0.92);
        --mic-card-soft: rgba(242,244,250,0.86);
        --mic-text: #223042;
        --mic-muted: #68768b;
        --mic-line: rgba(145,158,181,0.18);
        --mic-primary: #2559bd;
        --mic-primary-dim: #123f96;
        --mic-green: #1e8b5b;
        --mic-red: #b64040;
        --mic-shadow: 0 18px 40px rgba(30, 52, 84, 0.08);
        --mic-radius-xl: 24px;
        --mic-radius-lg: 18px;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(37,89,189,0.08), transparent 28%),
            linear-gradient(180deg, #fbfcff 0%, var(--mic-bg) 52%, #f3f5fa 100%);
        color: var(--mic-text);
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    [data-testid="stToolbar"] { right: 1rem; }
    [data-testid="stAppViewContainer"] > .main {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #edf1f7 0%, #e7ebf3 100%);
        border-right: 1px solid rgba(145,158,181,0.18);
    }

    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        border: 1px dashed rgba(37,89,189,0.25);
        background: rgba(255,255,255,0.72);
        border-radius: 20px;
        padding: 1rem;
    }

    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] .stDownloadButton > button {
        border-radius: 16px;
        border: none;
        background: linear-gradient(135deg, var(--mic-primary) 0%, var(--mic-primary-dim) 100%);
        color: white;
        font-weight: 700;
        box-shadow: 0 12px 24px rgba(37,89,189,0.22);
    }

    [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(255,255,255,0.65);
        border-radius: 20px;
        padding: 0.45rem;
        box-shadow: inset 0 0 0 1px rgba(145,158,181,0.12);
    }

    [data-baseweb="tab"] {
        border-radius: 14px !important;
        height: 44px !important;
        padding: 0 1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em;
    }

    [data-baseweb="tab-highlight"] {
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(37,89,189,0.12), rgba(37,89,189,0.18));
    }

    .mic-hero {
        display: flex;
        justify-content: space-between;
        align-items: stretch;
        gap: 1.25rem;
        padding: 1.35rem 1.5rem;
        border-radius: 26px;
        background: linear-gradient(135deg, rgba(255,255,255,0.94) 0%, rgba(247,249,255,0.88) 100%);
        box-shadow: var(--mic-shadow);
        border: 1px solid rgba(145,158,181,0.12);
        overflow: hidden;
        position: relative;
        margin-bottom: 1rem;
    }

    .mic-hero:after {
        content: '';
        position: absolute;
        right: -70px;
        top: -40px;
        width: 240px;
        height: 240px;
        background: linear-gradient(135deg, rgba(37,89,189,0.08), rgba(37,89,189,0.02));
        transform: rotate(24deg);
    }

    .mic-hero__title {
        font-family: 'Manrope', sans-serif;
        font-size: 2.15rem;
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin: 0;
        color: #14253b;
    }

    .mic-hero__meta {
        margin-top: 0.4rem;
        color: var(--mic-muted);
        font-size: 0.98rem;
        font-weight: 500;
    }

    .mic-badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.7rem;
        margin-top: 1rem;
        position: relative;
        z-index: 1;
    }

    .mic-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.55rem 0.9rem;
        border-radius: 999px;
        background: rgba(37,89,189,0.08);
        color: var(--mic-primary);
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .mic-badge--muted {
        background: rgba(108,118,139,0.09);
        color: var(--mic-muted);
    }

    .mic-shell {
        padding: 1.2rem;
        border-radius: var(--mic-radius-xl);
        background: linear-gradient(180deg, rgba(243,246,252,0.9), rgba(239,243,249,0.86));
        border: 1px solid rgba(145,158,181,0.12);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.8);
        margin-bottom: 1rem;
    }

    .mic-card {
        height: 100%;
        padding: 1.2rem 1.25rem;
        border-radius: var(--mic-radius-lg);
        background: var(--mic-card);
        border: 1px solid rgba(145,158,181,0.12);
        box-shadow: var(--mic-shadow);
    }

    .mic-card--soft {
        background: var(--mic-card-soft);
    }

    .mic-label {
        margin: 0 0 0.4rem 0;
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--mic-muted);
    }

    .mic-metric {
        font-family: 'Manrope', sans-serif;
        font-size: 2.2rem;
        line-height: 1;
        font-weight: 800;
        letter-spacing: -0.05em;
        color: #18283d;
    }

    .mic-metric--primary { color: var(--mic-primary); }
    .mic-metric--green { color: var(--mic-green); }
    .mic-metric--red { color: var(--mic-red); }

    .mic-submeta {
        font-size: 0.88rem;
        color: var(--mic-muted);
        font-weight: 500;
        margin-top: 0.2rem;
    }

    .mic-progress {
        margin-top: 1rem;
        width: 100%;
        height: 12px;
        border-radius: 999px;
        background: rgba(37,89,189,0.08);
        overflow: hidden;
    }

    .mic-progress > span {
        display: block;
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(135deg, #3a6fdb, #1e8b5b);
    }

    .mic-data-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0 10px;
        font-size: 13px;
        text-align: center;
    }

    .mic-data-table thead th {
        padding: 12px 10px;
        font-size: 0.69rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--mic-muted);
        background: rgba(237,241,248,0.72);
    }

    .mic-data-table tbody td {
        padding: 14px 10px;
        background: rgba(255,255,255,0.86);
        color: var(--mic-text);
    }

    .mic-data-table tbody tr td:first-child {
        border-top-left-radius: 16px;
        border-bottom-left-radius: 16px;
        font-weight: 700;
        text-align: left;
        padding-left: 16px;
    }

    .mic-data-table tbody tr td:last-child {
        border-top-right-radius: 16px;
        border-bottom-right-radius: 16px;
    }

    .mic-yield-chip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 76px;
        padding: 0.32rem 0.7rem;
        border-radius: 999px;
        background: rgba(30,139,91,0.1);
        color: var(--mic-green);
        font-weight: 800;
    }

    .mic-section-title {
        font-family: 'Manrope', sans-serif;
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0 0 1rem 0;
        color: #1b2e47;
    }

    .mic-note {
        color: var(--mic-muted);
        font-size: 0.82rem;
        margin-top: 0.55rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- [상단 헤더] ---
col_head1, col_head2 = st.columns([4.3, 1], vertical_alignment="center")
with col_head1:
    st.markdown(
        """
        <div class="mic-hero">
            <div style="position:relative; z-index:1;">
                <p class="mic-label" style="margin-bottom:0.5rem;">MIC Precision Console</p>
                <h1 class="mic-hero__title">🎙️ MIC Analysis Tool v1.1</h1>
                <div class="mic-hero__meta">High-clarity production analytics for MIC validation & defect triage</div>
                <div class="mic-badge-row">
                    <span class="mic-badge">Precision Dashboard</span>
                    <span class="mic-badge mic-badge--muted">Provided by JW Lee, JJ Kim</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_head2:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=260)

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


def build_report_filename(report):
    return f"{report.detection.report_basename}.xlsx"

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
            total_failure_samples = report.defect_summary.total_failure_samples

            # --- [UI Dashboard] ---
            st.markdown("<div class='mic-section-title'>📝 Production Dashboard</div>", unsafe_allow_html=True)
            d1, d2, d3 = st.columns([1.2, 1.3, 2.5])
            
            with d1:
                st.markdown(
                    f"""
                    <div class="mic-shell" style="height:100%;">
                        <div class="mic-card mic-card--soft">
                            <p class="mic-label">Production Meta</p>
                            <div class="mic-submeta"><strong style="color:#1b2e47;">Model P/N</strong><br>{detected_pn}</div>
                            <div class="mic-submeta" style="margin-top:0.85rem;"><strong style="color:#1b2e47;">Prod. Date</strong><br>{prod_date}</div>
                            <div class="mic-submeta" style="margin-top:0.85rem;"><strong style="color:#1b2e47;">Quantity</strong><br>{len(test_data)} EA</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with d2:
                st.markdown(f"""
                <div class="mic-shell" style="height:100%;">
                    <div class="mic-card">
                        <p class="mic-label">Overall Yield</p>
                        <div class="mic-metric mic-metric--primary">{yield_val:.1f}%</div>
                        <div class="mic-progress"><span style="width:{yield_val}%;"></span></div>
                        <div style="display:flex; gap:12px; margin-top:1rem;">
                            <div class="mic-card mic-card--soft" style="flex:1; padding:0.9rem 1rem; box-shadow:none;">
                                <p class="mic-label">Pass</p>
                                <div class="mic-metric mic-metric--green" style="font-size:1.8rem;">{total_pass}</div>
                            </div>
                            <div class="mic-card mic-card--soft" style="flex:1; padding:0.9rem 1rem; box-shadow:none;">
                                <p class="mic-label">Fail</p>
                                <div class="mic-metric mic-metric--red" style="font-size:1.8rem;">{total_fail}</div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with d3:
                s_html = """<div class='mic-shell'><div class='mic-card'><table class='mic-data-table'>
                <thead>
                <tr><th rowspan="2">MIC</th><th colspan="7">Precision Metrics Summary</th></tr>
                <tr><th>Pass</th><th>Fail</th><th>Yield</th><th>Min</th><th>Max</th><th>Avg</th><th>Stdev</th></tr>
                </thead><tbody>"""
                for ch_n, stat in channel_statistics.items():
                    v_min, v_max, v_avg, v_std, yld = stat.summary_metrics(report.stats_indices, total_qty)
                    yld_pct = f"{yld * 100:.1f}%"
                    s_html += f"<tr><td>{ch_n}</td>"
                    s_html += f"<td>{stat.pass_count}</td><td>{stat.fail_count}</td>"
                    s_html += f"<td><span class='mic-yield-chip'>{yld_pct}</span></td><td>{v_min:.3f}</td>"
                    s_html += f"<td>{v_max:.3f}</td><td style='color:var(--mic-primary); font-weight:800;'>{v_avg:.3f}</td><td>{v_std:.3f}</td></tr>"
                s_html += "</tbody></table></div></div>"
                st.markdown(s_html, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # [탭 확장: 불량 유형 요약 탭 추가]
            tab_fr, tab_dist, tab_detail, tab_summary = st.tabs(["📈 주파수 응답 (FR)", "📉 정규분포 (Cpk)", "🔍 결함 시료 상세", "📊 기타 Yield & 통계"])
            
            # 사이드바 체크박스 (전체 선택 기능)
            st.sidebar.header("❌ 결함 시료 선택")
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
                st.markdown("<div class='mic-section-title' style='font-size:1.1rem;'>⚠️ Process Control Limit</div>", unsafe_allow_html=True)
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
                st.markdown("<div style='margin-top:8px; margin-bottom:8px; font-weight:700; font-size:18px;'>불량 유형 요약</div>", unsafe_allow_html=True)

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
                total_rate = (total_failure_samples / total_qty * 100) if total_qty > 0 else 0
                html_parts.append("<tr style='background-color:#E2EFDA; font-weight:bold;'>")
                html_parts.append("<td style='border:1px solid #bdc3c7; padding:5px;'>Total Failure</td>")
                html_parts.append(f"<td style='border:1px solid #bdc3c7; padding:5px;'>{total_failure_samples} EA</td>")
                html_parts.append(f"<td style='border:1px solid #bdc3c7; padding:5px;'>{total_rate:.1f}%</td>")
                html_parts.append("</tr></tbody></table>")
                
                st.markdown("".join(html_parts), unsafe_allow_html=True)
                st.caption(f"※ 비율(Rate)은 전체 검사 수량({total_qty} EA) 대비 발생 비율입니다.")

                def build_frequency_stats_table(title, freq_label, stats_map):
                    table_parts = []
                    table_parts.append(f"<div style='margin-top:24px; margin-bottom:8px; font-weight:700; font-size:18px;'>{title}</div>")
                    table_parts.append("<table style='width:100%; border-collapse:collapse; border:1px solid #bdc3c7; font-size:13px; text-align:center;'>")
                    table_parts.append("<thead style='background-color:#F2F2F2; font-weight:bold;'><tr>")
                    table_parts.append("<th style='border:1px solid #bdc3c7; padding:8px;'>MIC</th>")
                    table_parts.append("<th style='border:1px solid #bdc3c7; padding:8px;'>Yield</th>")
                    table_parts.append("<th style='border:1px solid #bdc3c7; padding:8px;'>Min</th>")
                    table_parts.append("<th style='border:1px solid #bdc3c7; padding:8px;'>Max</th>")
                    table_parts.append("<th style='border:1px solid #bdc3c7; padding:8px;'>Avg</th>")
                    table_parts.append("<th style='border:1px solid #bdc3c7; padding:8px;'>Stdev</th>")
                    table_parts.append("</tr></thead><tbody>")

                    for mic_name, stat in stats_map.items():
                        v_min, v_max, v_avg, v_std, yld = stat.summary_metrics(report.stats_indices, total_qty, label=freq_label)
                        table_parts.append("<tr>")
                        table_parts.append(f"<td style='border:1px solid #bdc3c7; padding:5px; font-weight:bold; background-color:#F9F9F9;'>{mic_name}</td>")
                        table_parts.append(f"<td style='border:1px solid #bdc3c7; padding:5px;'>{yld * 100:.1f}%</td>")
                        table_parts.append(f"<td style='border:1px solid #bdc3c7; padding:5px;'>{v_min:.3f}</td>")
                        table_parts.append(f"<td style='border:1px solid #bdc3c7; padding:5px;'>{v_max:.3f}</td>")
                        table_parts.append(f"<td style='border:1px solid #bdc3c7; padding:5px;'>{v_avg:.3f}</td>")
                        table_parts.append(f"<td style='border:1px solid #bdc3c7; padding:5px;'>{v_std:.3f}</td>")
                        table_parts.append("</tr>")

                    table_parts.append("</tbody></table>")
                    return "".join(table_parts)

                st.markdown(build_frequency_stats_table("200Hz 통계", "200Hz", channel_statistics), unsafe_allow_html=True)
                st.markdown(build_frequency_stats_table("4kHz 통계", "4kHz", channel_statistics), unsafe_allow_html=True)

                digital_channels = [channel for channel in report.product_spec.channels if channel.mic_type == "digital"]
                if digital_channels:
                    digital_stats_parts = []
                    digital_stats_parts.append("<div style='margin-top:24px; margin-bottom:8px; font-weight:700; font-size:18px;'>Digital MIC 통합 통계</div>")
                    digital_stats_parts.append("<table style='width:100%; border-collapse:collapse; border:1px solid #bdc3c7; font-size:13px; text-align:center;'>")
                    digital_stats_parts.append("<thead style='background-color:#F2F2F2; font-weight:bold;'><tr>")
                    digital_stats_parts.append("<th style='border:1px solid #bdc3c7; padding:8px;'>Group</th>")
                    digital_stats_parts.append("<th style='border:1px solid #bdc3c7; padding:8px;'>Freq.</th>")
                    digital_stats_parts.append("<th style='border:1px solid #bdc3c7; padding:8px;'>Yield</th>")
                    digital_stats_parts.append("<th style='border:1px solid #bdc3c7; padding:8px;'>Min</th>")
                    digital_stats_parts.append("<th style='border:1px solid #bdc3c7; padding:8px;'>Max</th>")
                    digital_stats_parts.append("<th style='border:1px solid #bdc3c7; padding:8px;'>Avg</th>")
                    digital_stats_parts.append("<th style='border:1px solid #bdc3c7; padding:8px;'>Stdev</th>")
                    digital_stats_parts.append("</tr></thead><tbody>")

                    digital_yield = (
                        sum(
                            all(channel.status == "Normal" for channel in sample.channels if channel.mic_name in {c.name for c in digital_channels})
                            for sample in report.samples
                        ) / total_qty
                    ) if total_qty > 0 else 0.0

                    for freq_label in ["200Hz", "1kHz", "4kHz"]:
                        combined_values = []
                        digital_names = {channel.name for channel in digital_channels}
                        for sample_index in report.stats_indices:
                            sample = report.sample_by_index(sample_index)
                            sample_values = []
                            for channel in sample.channels:
                                if channel.mic_name not in digital_names:
                                    continue
                                point = channel.point(freq_label)
                                try:
                                    sample_values.append(float(point.display_value))
                                except ValueError:
                                    continue

                            if sample_values:
                                combined_values.append(float(np.mean(sample_values)))

                        if combined_values:
                            combined_array = np.array(combined_values)
                            v_min = combined_array.min()
                            v_max = combined_array.max()
                            v_avg = combined_array.mean()
                            v_std = combined_array.std()
                        else:
                            v_min = v_max = v_avg = v_std = 0.0

                        digital_stats_parts.append("<tr>")
                        digital_stats_parts.append("<td style='border:1px solid #bdc3c7; padding:5px; font-weight:bold; background-color:#F9F9F9;'>Digital MIC Total</td>")
                        digital_stats_parts.append(f"<td style='border:1px solid #bdc3c7; padding:5px;'>{freq_label}</td>")
                        digital_stats_parts.append(f"<td style='border:1px solid #bdc3c7; padding:5px;'>{digital_yield * 100:.1f}%</td>")
                        digital_stats_parts.append(f"<td style='border:1px solid #bdc3c7; padding:5px;'>{v_min:.3f}</td>")
                        digital_stats_parts.append(f"<td style='border:1px solid #bdc3c7; padding:5px;'>{v_max:.3f}</td>")
                        digital_stats_parts.append(f"<td style='border:1px solid #bdc3c7; padding:5px;'>{v_avg:.3f}</td>")
                        digital_stats_parts.append(f"<td style='border:1px solid #bdc3c7; padding:5px;'>{v_std:.3f}</td>")
                        digital_stats_parts.append("</tr>")

                    digital_stats_parts.append("</tbody></table>")
                    st.markdown("".join(digital_stats_parts), unsafe_allow_html=True)

            st.sidebar.markdown("<div class='mic-label' style='margin-top:1.5rem;'>Export</div>", unsafe_allow_html=True)
            img_b64 = get_base64_image("excel_icon.png")
            if img_b64:
                st.sidebar.markdown(f'<div class="mic-card mic-card--soft" style="margin-bottom:12px; display:flex; align-items:center; gap:12px;"><img src="data:image/png;base64,{img_b64}" width="34"><div><div class="mic-label" style="margin-bottom:4px;">Report Output</div><div style="font-family:Manrope,sans-serif; font-size:1.15rem; font-weight:800; color:#1b2e47;">Excel Export</div></div></div>', unsafe_allow_html=True)
            else: st.sidebar.header("📊 Excel Export")
            excel_data = ExcelReportBuilder(report, LIMIT_POLICY, show_normal, sel_idx, create_fr_plot, plot_bell_curve_set).build()
            st.sidebar.download_button(label="📥 Download Report", data=excel_data, file_name=build_report_filename(report), mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
else: st.info("사이드바에서 CSV 로그 파일을 업로드하세요.")
