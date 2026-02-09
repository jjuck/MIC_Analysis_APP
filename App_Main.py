import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import io
import chardet
import os
import base64

# 1. 페이지 설정 및 UI 상수
st.set_page_config(page_title="MIC Analysis Tool", page_icon="🎙️", layout="wide")
FIG_WIDTH, PLOT_HEIGHT = 14, 6
FONT_SIZE_TITLE, FONT_SIZE_AXIS = 16, 12

# --- [미세조정 설정값] ---
X_OFFSET, Y_OFFSET = 10, 10 
# -----------------------

# --- [상단 헤더] ---
col_head1, col_head2 = st.columns([4, 1], vertical_alignment="center")
with col_head1:
    st.markdown("<h1>🎙️ MIC Analysis Tool <span style='font-size: 16px; color: gray; font-weight: normal; margin-left: 10px;'>( 제작 : JW Lee, 자문 : JJ Kim )</span></h1>", unsafe_allow_html=True)
with col_head2:
    if os.path.exists("logo.png"): st.image("logo.png", width=300)
st.markdown("---")

# 2. 제품군 설정
PRODUCT_CONFIGS = {
    "RH": {"pn": ["96575N1100", "96575GJ100"], "channels": [{"name": "Digital MIC1", "type": "digital", "range": range(51, 101), "thd_idx": 15}, {"name": "Digital MIC2", "type": "digital", "range": range(103, 153), "thd_idx": 18}, {"name": "Digital MIC3", "type": "digital", "range": range(155, 205), "thd_idx": 21}]},
    "3903(LH Ecall)": {"pn": ["96575N1050", "96575GJ000"], "channels": [{"name": "Ecall MIC (Analog)", "type": "analog", "range": range(6, 47), "thd_idx": 69}, {"name": "Digital MIC1", "type": "digital", "range": range(107, 157), "thd_idx": 217}, {"name": "Digital MIC2", "type": "digital", "range": range(159, 209), "thd_idx": 220}]},
    "3203(LH non Ecall)": {"pn": ["96575N1000", "96575GJ010"], "channels": [{"name": "Digital MIC1", "type": "digital", "range": range(6, 56), "thd_idx": 116}, {"name": "Digital MIC2", "type": "digital", "range": range(58, 108), "thd_idx": 119}]},
    "LITE(LH)": {"pn": ["96575NR000", "96575GJ200"], "channels": [{"name": "Analog MIC", "type": "analog", "range": range(6, 47), "thd_idx": 95}]},
    "LITE(RH)": {"pn": ["96575NR100", "96575GJ300"], "channels": [{"name": "Analog MIC", "type": "analog", "range": range(6, 47), "thd_idx": 95}]}
}

# 3. [유틸리티 함수]
def clean_sn(val):
    if pd.isna(val): return ""
    return str(val).replace('"', '').replace("'", "").replace('\t', '').strip()

def get_freq_values(cols):
    return [float(str(c).split('.')[0]) for c in cols]

def detect_info(df):
    model, prod_date, matched_pn = None, "Unknown", "Unknown"
    try:
        for i in range(2, min(15, len(df))):
            sn_raw = clean_sn(df.iloc[i, 3])
            if '/' in sn_raw:
                parts = sn_raw.split('/'); pn_part = parts[0]
                for m, info in PRODUCT_CONFIGS.items():
                    if pn_part in info["pn"]: model, matched_pn = m, pn_part; break
                if len(parts[1]) >= 8: prod_date = f"20{parts[1][2:4]}/{parts[1][4:6]}/{parts[1][6:8]}"
                if model: break
    except: pass
    return model, prod_date, matched_pn

def classify_sample(row, cols, freqs, limit_low, limit_high):
    val = pd.to_numeric(row[cols], errors='coerce')
    l_low = pd.to_numeric(limit_low[cols], errors='coerce')
    l_high = pd.to_numeric(limit_high[cols], errors='coerce')
    is_fail = val.isna() | (val < l_low) | (val > l_high)
    if not is_fail.any(): return "Normal"
    check_pts = [200, 1000, 4000]
    pt_idx = [np.argmin(np.abs(np.array(freqs) - p)) for p in check_pts]
    other_idx = [i for i in range(len(cols)) if i not in pt_idx]
    if not is_fail.iloc[other_idx].any(): return "Margin Out"
    return "Defect"

def get_row_summary_data(row, ch_info, all_cols):
    cols = all_cols[ch_info["range"]]; freqs = get_freq_values(cols)
    data = {"Channel": ch_info["name"]}
    for t in [200, 1000, 4000]:
        try:
            idx = np.argmin(np.abs(np.array(freqs) - t))
            val = pd.to_numeric(row[cols[idx]], errors='coerce')
            data[f"{t}Hz"] = f"{val:.3f}" if not pd.isna(val) else "-"
        except: data[f"{t}Hz"] = "-"
    thd_idx = ch_info.get("thd_idx")
    if thd_idx is not None:
        thd_val = pd.to_numeric(row[all_cols[thd_idx]], errors='coerce')
        data["THD"] = f"{thd_val:.3f}" if not pd.isna(thd_val) else "-"
    else: data["THD"] = "N/A"
    return data

def create_fr_plot(config, df, current_test_data, limit_low, limit_high, show_normal, plotting_normal_indices, highlight_indices, for_excel=False):
    num_draw = 3 if for_excel else len(config["channels"])
    fig, axes = plt.subplots(num_draw, 1, figsize=(FIG_WIDTH, PLOT_HEIGHT * num_draw))
    if num_draw == 1: axes = [axes]
    if for_excel: fig.patch.set_linewidth(0)
    for i in range(num_draw):
        ax = axes[i]
        if i < len(config["channels"]):
            ch = config["channels"][i]; cols = df.columns[ch["range"]]; freqs = get_freq_values(cols)
            ylim, color, unit = ((-30, 0), 'green', 'dbV') if ch["type"] == 'analog' else ((-45, -25), 'blue', 'dbFS')
            ax.set_xscale('log'); ax.set_xticks([50, 100, 200, 1000, 4000, 10000, 14000])
            ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter()); ax.minorticks_off()
            if show_normal:
                for n in plotting_normal_indices: ax.plot(freqs, pd.to_numeric(current_test_data.loc[n, cols], errors='coerce'), color=color, alpha=0.7, lw=1.2)
            for h in highlight_indices: ax.plot(freqs, pd.to_numeric(current_test_data.loc[h, cols], errors='coerce'), color='red', lw=2.5)
            ax.plot(freqs, pd.to_numeric(limit_low[cols], errors='coerce'), 'k--', lw=1.2); ax.plot(freqs, pd.to_numeric(limit_high[cols], errors='coerce'), 'k--', lw=1.2)
            ax.set_title(ch["name"], fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
            ax.set_ylabel(f'Response ({unit})', fontsize=FONT_SIZE_AXIS); ax.set_ylim(ylim); ax.grid(True, alpha=0.4)
        else: ax.axis('off')
    plt.tight_layout(); return fig

def plot_bell_curve_set(config, df, test_data, stats_indices, sel_idx, for_excel=False):
    num_draw = 3 if for_excel else len(config["channels"])
    fig, axes = plt.subplots(num_draw, 1, figsize=(FIG_WIDTH, PLOT_HEIGHT * num_draw))
    if num_draw == 1: axes = [axes]
    if for_excel: fig.patch.set_linewidth(0)
    for i in range(num_draw):
        ax = axes[i]
        if i < len(config["channels"]):
            ch = config["channels"][i]; col_idx = np.argmin(np.abs(np.array(get_freq_values(df.columns[ch["range"]])) - 1000))
            v_all = pd.to_numeric(test_data[df.columns[ch["range"]][col_idx]], errors='coerce')
            v_clean = v_all.iloc[stats_indices].dropna()
            lcl, ucl = (-11, -9) if ch["type"] == 'analog' else (-38, -36)
            if len(v_clean) >= 2:
                mu, std = v_clean.mean(), v_clean.std()
                cpk = min((ucl-mu)/(3*std), (mu-lcl)/(3*std)) if std > 0 else 0
                x_r = np.linspace(lcl - 2, ucl + 2, 200); p = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_r - mu) / std)**2)
                ax.plot(x_r, p, 'k', lw=2.5, alpha=0.7); ax.fill_between(x_r, p, color='gray', alpha=0.1)
                ax.axvline(lcl, color='blue', ls='--', lw=1.5); ax.axvline(ucl, color='red', ls='--', lw=1.5)
                if sel_idx:
                    sel_v = v_all.iloc[sel_idx].dropna()
                    for v in sel_v:
                        if std > 0:
                            p_v = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((v - mu) / std)**2)
                            ax.scatter(v, p_v, color='red', s=200, zorder=5, edgecolors='white', linewidth=1.5)
                ax.text(0.95, 0.75, "Cpk: " + str(round(cpk, 2)), transform=ax.transAxes, ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=FONT_SIZE_AXIS, fontweight='bold')
                ax.set_title(f"{ch['name']} - Distribution", fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15); ax.set_xlim(lcl-2, ucl+2); ax.grid(True, alpha=0.2)
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
    raw_bytes = uploaded_file.read()
    det_enc = chardet.detect(raw_bytes)['encoding']
    df = pd.read_csv(io.StringIO(raw_bytes.decode(det_enc if det_enc else 'utf-8-sig', errors='replace')), low_memory=False)
    df.columns = [str(c) for c in df.columns]

    if df is not None:
        detected_model, prod_date, detected_pn = detect_info(df)
        model_list = list(PRODUCT_CONFIGS.keys())
        model_type = st.sidebar.selectbox("제품 모델 선택", options=model_list, index=model_list.index(detected_model) if detected_model in model_list else 0)
        config = PRODUCT_CONFIGS[model_type]; st.sidebar.markdown("---")
        
        st.sidebar.header("✔️ 정상 시료 설정")
        show_normal = st.sidebar.checkbox("정상 시료 FR 표시", value=True)

        sn_col, limit_low, limit_high = df.columns[3], df.iloc[0], df.iloc[1]
        raw_test_data = df.iloc[2:].dropna(subset=[sn_col])
        test_data = raw_test_data[raw_test_data[sn_col].astype(str).str.contains('/', na=False)].reset_index(drop=True)

        if len(test_data) > 0:
            sample_info, issue_indices, stats_indices, plotting_normal_indices = {}, [], [], []
            ch_stats_data = {ch["name"]: {"pass": 0, "fail": 0, "vals_1k": []} for ch in config["channels"]}

            for idx, row in test_data.iterrows():
                row_res = []
                is_pure_normal, is_fail, is_defect = True, False, False
                for ch in config["channels"]:
                    cols = df.columns[ch["range"]]; freqs = get_freq_values(cols)
                    status = classify_sample(row, cols, freqs, limit_low, limit_high)
                    if status == "Defect": is_defect = True
                    if status != "Normal": is_pure_normal, is_fail = False, True
                    val_1k = pd.to_numeric(row[cols[np.argmin(np.abs(np.array(freqs) - 1000))]], errors='coerce')
                    ch_stats_data[ch["name"]]["vals_1k"].append(val_1k)
                    if status == "Normal": ch_stats_data[ch["name"]]["pass"] += 1
                    else: ch_stats_data[ch["name"]]["fail"] += 1
                    res = get_row_summary_data(row, ch, df.columns); res["Status"] = status; row_res.append(res)
                
                sample_info[idx] = {"table": pd.DataFrame(row_res), "sn": clean_sn(row[sn_col])}
                if is_fail: issue_indices.append(idx)
                if is_pure_normal: plotting_normal_indices.append(idx)
                if not is_defect: stats_indices.append(idx)

            st.sidebar.markdown("---"); st.sidebar.header("❌️ 결함 시료 선택")
            sel_idx = [i for i in issue_indices if st.sidebar.checkbox(f"SN: {sample_info[i]['sn']}", key=f"ch_{i}")]

            # --- [UI Dashboard] ---
            st.subheader("📝 Production Dashboard")
            d1, d2, d3 = st.columns([1.2, 1.3, 2.5])
            total_qty, total_fail = len(test_data), len(issue_indices); total_pass = total_qty - total_fail; yield_val = total_pass / total_qty * 100
            
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
                <tr><th rowspan="2" style="border:1px solid #bdc3c7; padding:8px;">Channel</th><th colspan="7" style="border:1px solid #bdc3c7; padding:8px;">Statistics</th></tr>
                <tr><th style="border:1px solid #bdc3c7; padding:5px;">Pass</th><th style="border:1px solid #bdc3c7; padding:5px;">Fail</th><th style="border:1px solid #bdc3c7; padding:5px;">Yield</th><th style="border:1px solid #bdc3c7; padding:5px;">Min</th><th style="border:1px solid #bdc3c7; padding:5px;">Max</th><th style="border:1px solid #bdc3c7; padding:5px;">Avg</th><th style="border:1px solid #bdc3c7; padding:5px;">Stdev</th></tr>
                </thead><tbody>"""
                for ch_n, stat in ch_stats_data.items():
                    v = np.array(stat["vals_1k"])[stats_indices]; v = v[~np.isnan(v)]
                    v_min, v_max, v_avg, v_std = (v.min(), v.max(), v.mean(), v.std()) if len(v) > 0 else (0,0,0,0)
                    yld = f"{(stat['pass']/total_qty)*100:.1f}%"
                    s_html += f"<tr><td style='border:1px solid #bdc3c7; padding:5px; font-weight:bold; background-color:#F9F9F9;'>{ch_n}</td>"
                    s_html += f"<td style='border:1px solid #bdc3c7; padding:5px;'>{stat['pass']}</td><td style='border:1px solid #bdc3c7; padding:5px;'>{stat['fail']}</td>"
                    s_html += f"<td style='border:1px solid #bdc3c7; padding:5px;'>{yld}</td><td style='border:1px solid #bdc3c7; padding:5px;'>{v_min:.2f}</td>"
                    s_html += f"<td style='border:1px solid #bdc3c7; padding:5px;'>{v_max:.2f}</td><td style='border:1px solid #bdc3c7; padding:5px;'>{v_avg:.2f}</td><td style='border:1px solid #bdc3c7; padding:5px;'>{v_std:.2f}</td></tr>"
                s_html += "</tbody></table>"
                st.markdown(s_html, unsafe_allow_html=True)
            
            st.markdown("---"); tab_fr, tab_dist, tab_detail = st.tabs(["📈 주파수 응답 (FR)", "📉 정규분포 (Cpk)", "🔍 결함 시료 상세"])
            with tab_fr: st.pyplot(create_fr_plot(config, df, test_data, limit_low, limit_high, show_normal, plotting_normal_indices, sel_idx))
            with tab_dist: st.pyplot(plot_bell_curve_set(config, df, test_data, stats_indices, sel_idx))
            with tab_detail:
                if sel_idx:
                    for idx in sel_idx:
                        st.markdown(f"📄 **SN: {sample_info[idx]['sn']}**", unsafe_allow_html=True)
                        p_html = """<table style="width:100%; border-collapse:collapse; border:1px solid #bdc3c7; font-size:13px; text-align:center; margin-bottom:20px;">
                        <thead style="background-color:#F2F2F2; font-weight:bold;">
                        <tr><th rowspan="3" style="border:1px solid #bdc3c7; padding:8px;">Channel</th><th colspan="5" style="border:1px solid #bdc3c7; padding:8px;">Parameter</th></tr>
                        <tr><th colspan="3" style="border:1px solid #bdc3c7; padding:5px;">Frequency Response</th><th style="border:1px solid #bdc3c7; padding:5px;">THD</th><th rowspan="2" style="border:1px solid #bdc3c7; padding:5px;">Status</th></tr>
                        <tr><th style="border:1px solid #bdc3c7; padding:5px;">200Hz</th><th style="border:1px solid #bdc3c7; padding:5px;">1000Hz</th><th style="border:1px solid #bdc3c7; padding:5px;">4000Hz</th><th style="border:1px solid #bdc3c7; padding:5px;">1kHz</th></tr>
                        </thead><tbody>"""
                        for _, r in sample_info[idx]['table'].iterrows():
                            p_html += f"<tr><td style='border:1px solid #bdc3c7; padding:5px; font-weight:bold; background-color:#F9F9F9;'>{r['Channel']}</td>"
                            p_html += f"<td style='border:1px solid #bdc3c7; padding:5px;'>{r['200Hz']}</td><td style='border:1px solid #bdc3c7; padding:5px;'>{r['1000Hz']}</td><td style='border:1px solid #bdc3c7; padding:5px;'>{r['4000Hz']}</td>"
                            p_html += f"<td style='border:1px solid #bdc3c7; padding:5px;'>{r['THD']}</td><td style='border:1px solid #bdc3c7; padding:5px;'>{r['Status']}</td></tr>"
                        p_html += "</tbody></table>"
                        st.markdown(p_html, unsafe_allow_html=True)
                else: st.warning("결함 시료를 선택하세요.")

            # --- [엑셀 Export 로직: Status 3행 병합 & 하단 코너 마감 수정본] ---
            def generate_excel():
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    workbook = writer.book
                    
                    # 1. 서식 라이브러리
                    base_blue = {'bold': True, 'bg_color': '#DEEAF6', 'align': 'center', 'valign': 'vcenter', 'border': 1}
                    base_green = {'bold': True, 'bg_color': '#E2EFDA', 'align': 'center', 'valign': 'vcenter', 'border': 1}
                    base_thin = {'align': 'center', 'valign': 'vcenter', 'border': 1}
                    base_yld_val = {'bold': True, 'font_size': 18, 'font_color': '#2E7D32', 'align': 'center', 'valign': 'vcenter', 'border': 1, 'num_format': '0.0%'}

                    def get_fmt(base_dict, top=None, bottom=None, left=None, right=None):
                        props = base_dict.copy()
                        if top is not None: props['top'] = top
                        if bottom is not None: props['bottom'] = bottom
                        if left is not None: props['left'] = left
                        if right is not None: props['right'] = right
                        return workbook.add_format(props)

                    # --- Sheet 1: 분석 리포트 (기존 유지) ---
                    def write_dashboard(ws, last_row_idx=37):
                        ws.set_column('A:A', 3); ws.set_column('B:B', 15); ws.set_column('C:C', 22); ws.set_column('D:F', 10); ws.set_column('G:N', 11)
                        ws.merge_range('B2:F2', '📝 PRODUCTION SUMMARY', get_fmt(base_blue, top=2, left=2, bottom=1, right=1))
                        ws.merge_range('G2:N2', '📈 CHANNEL STATISTICS', get_fmt(base_green, top=2, right=2, bottom=1, left=1))
                        
                        sums = [("Model Type", model_type), ("Model P/N", detected_pn), ("Prod. Date", prod_date), ("Quantity", str(total_qty) + " EA")]
                        for i, (k, v) in enumerate(sums):
                            r = 2 + i
                            ws.write(r, 1, k, get_fmt(base_blue, left=2, bottom=2 if r==5 else 1, top=1, right=1))
                            ws.write(r, 2, v, get_fmt(base_thin, bottom=2 if r==5 else 1, top=1, left=1, right=1))
                        
                        ws.write(2, 3, 'PASS', get_fmt(base_blue, top=1, bottom=1, left=1, right=1))
                        ws.merge_range('E3:F3', total_pass, get_fmt(base_thin, top=1, bottom=1, left=1, right=1))
                        ws.write(3, 3, 'FAIL', get_fmt(base_blue, top=1, bottom=1, left=1, right=1))
                        ws.merge_range('E4:F4', total_fail, get_fmt(base_thin, top=1, bottom=1, left=1, right=1))
                        ws.merge_range('D5:D6', 'Yield', get_fmt(base_blue, bottom=2, top=1, left=1, right=1))
                        ws.merge_range('E5:F6', yield_val/100, get_fmt(base_yld_val, bottom=2, top=1, left=1, right=1))

                        ws.write(2, 6, "MIC", get_fmt(base_green, top=1, bottom=1, left=1, right=1))
                        heads = ["Pass", "Fail", "Yield", "Min", "Max", "Avg", "Stdev"]
                        for i, h in enumerate(heads):
                            ws.write(2, 7+i, h, get_fmt(base_green, right=2 if 7+i==13 else 1, top=1, bottom=1, left=1))
                        
                        for r_idx in range(4):
                            r = 3 + r_idx; is_l = (r == 5)
                            if r_idx < len(config["channels"]):
                                ch_n = config["channels"][r_idx]["name"]; stat = ch_stats_data[ch_n]
                                v = np.array(stat["vals_1k"])[stats_indices]; v = v[~np.isnan(v)]
                                v_min, v_max, v_avg, v_std = (v.min(), v.max(), v.mean(), v.std()) if len(v) > 0 else (0,0,0,0)
                                ws.write(r, 6, ch_n, get_fmt(base_thin, bottom=2 if is_l else 1, top=1, left=1, right=1))
                                vals = [stat['pass'], stat['fail'], f"{(stat['pass']/total_qty)*100:.1f}%", f"{v_min:.2f}", f"{v_max:.2f}", f"{v_avg:.2f}", f"{v_std:.2f}"]
                                for i, val in enumerate(vals): ws.write(r, 7+i, val, get_fmt(base_thin, right=2 if 7+i==13 else 1, bottom=2 if is_l else 1, top=1, left=1))
                            else:
                                for c in range(6, 14): ws.write_blank(r, c, "", get_fmt({'border':0}, right=2 if c==13 else 0, bottom=2 if is_l else 0, left=1 if c==6 else 0))

                        # 하단 프레임 기둥 (기본)
                        for r_f in range(6, last_row_idx - 1):
                            ws.write_blank(r_f, 1, "", get_fmt({'border':0}, left=2))
                            ws.write_blank(r_f, 13, "", get_fmt({'border':0}, right=2))
                        
                        # 최하단 바닥 마감 (기본)
                        ws.write_blank(last_row_idx-1, 1, "", get_fmt({'border':0}, left=2, bottom=2))
                        for c_b in range(2, 13): ws.write_blank(last_row_idx-1, c_b, "", get_fmt({'border':0}, bottom=2))
                        ws.write_blank(last_row_idx-1, 13, "", get_fmt({'border':0}, right=2, bottom=2))

                    ws1 = workbook.add_worksheet('📈 분석 리포트'); write_dashboard(ws1, 37)
                    fig_fr = create_fr_plot(config, df, test_data, limit_low, limit_high, show_normal, plotting_normal_indices, sel_idx, for_excel=True)
                    buf_f = io.BytesIO(); fig_fr.savefig(buf_f, format='png', dpi=100); plt.close(fig_fr)
                    ws1.insert_image('B7', 'fr.png', {'image_data': buf_f, 'x_scale': 0.41, 'y_scale': 0.35, 'x_offset': 10, 'y_offset': 10})
                    fig_dist = plot_bell_curve_set(config, df, test_data, stats_indices, sel_idx, for_excel=True)
                    buf_d = io.BytesIO(); fig_dist.savefig(buf_d, format='png', dpi=100); plt.close(fig_dist)
                    ws1.insert_image('H7', 'dist.png', {'image_data': buf_d, 'x_scale': 0.41, 'y_scale': 0.35, 'x_offset': 10, 'y_offset': 10})

                    # --- Sheet 2: 결함상세 (8행 고정 & Status 3단 병합 & 코너 마감) ---
                    ws2 = workbook.add_worksheet('🔍 결함상세')
                    total_defect_rows = len(sel_idx) * 8 if sel_idx else 0
                    l_f_row = max(37, 9 + total_defect_rows)
                    write_dashboard(ws2, l_f_row)
                    ws2.merge_range('B8:N8', '🔍 DETAILED FAILURE LOG', get_fmt(base_blue, left=2, right=2, top=1, bottom=1))
                    
                    curr_r = 9
                    if sel_idx:
                        for i in sel_idx:
                            # 1. SN 바 (Row 1)
                            ws2.merge_range(curr_r, 1, curr_r, 13, sample_info[i]['sn'], get_fmt({'bold':True, 'bg_color':'#F2F2F2', 'border':1}, left=2, right=2, top=1, bottom=1))
                            curr_r += 1
                            
                            # 2. 헤더 Tier 1 (Row 2) -> Status 병합 시작행(curr_r)
                            status_start_row = curr_r
                            ws2.merge_range(curr_r, 1, curr_r+2, 1, 'MIC', get_fmt(base_blue, left=2, top=1, bottom=1)) # MIC: 3행 병합
                            ws2.merge_range(curr_r, 2, curr_r, 5, 'Parameter', get_fmt(base_blue, top=1, bottom=1, left=1, right=1))
                            # G열(6) 건너뜀 (Status 자리)
                            for c in range(7, 13): ws2.write_blank(curr_r, c, "", get_fmt({'border':0}))
                            ws2.write_blank(curr_r, 13, "", get_fmt({'border':0}, right=2))
                            curr_r += 1
                            
                            # 헤더 Tier 2 (Row 3)
                            ws2.merge_range(curr_r, 2, curr_r, 4, 'Frequency Response', get_fmt(base_blue, top=1, bottom=1, left=1, right=1))
                            ws2.write(curr_r, 5, 'THD', get_fmt(base_blue, top=1, bottom=1, left=1, right=1))
                            # G열(6) 건너뜀
                            for c in range(7, 13): ws2.write_blank(curr_r, c, "", get_fmt({'border':0}))
                            ws2.write_blank(curr_r, 13, "", get_fmt({'border':0}, right=2))
                            curr_r += 1
                            
                            # 헤더 Tier 3 (Row 4) -> Status 병합 끝행(curr_r)
                            status_end_row = curr_r
                            t3_h = ['200Hz', '1kHz', '4kHz', '1kHz']
                            for c_idx, h in enumerate(t3_h): ws2.write(curr_r, 2+c_idx, h, get_fmt(base_blue, top=1, bottom=1, left=1, right=1))
                            # G열(6) 건너뜀
                            for c in range(7, 13): ws2.write_blank(curr_r, c, "", get_fmt({'border':0}))
                            ws2.write_blank(curr_r, 13, "", get_fmt({'border':0}, right=2))
                            
                            # [Status 병합] G열(6) 3행 병합 (덮어쓰기 방지 위해 마지막에 수행)
                            ws2.merge_range(status_start_row, 6, status_end_row, 6, 'Status', get_fmt(base_blue, top=1, bottom=1, left=1, right=1))
                            curr_r += 1
                            
                            # 3. 데이터 행 (Row 5~7 가변)
                            rows_written = 0
                            for _, r_v in sample_info[i]['table'].iterrows():
                                ws2.write(curr_r, 1, r_v['Channel'], get_fmt(base_thin, left=2, top=1, bottom=1, right=1))
                                d_row = [r_v['200Hz'], r_v['1000Hz'], r_v['4000Hz'], r_v['THD'], r_v['Status']]
                                for c_idx, val in enumerate(d_row): ws2.write(curr_r, 2+c_idx, val, get_fmt(base_thin, top=1, bottom=1, left=1, right=1))
                                for c in range(7, 13): ws2.write_blank(curr_r, c, "", get_fmt({'border':0}))
                                ws2.write_blank(curr_r, 13, "", get_fmt({'border':0}, right=2))
                                curr_r += 1
                                rows_written += 1
                            
                            # 4. 8행 높이 맞추기 패딩
                            padding = 3 - rows_written
                            for _ in range(padding):
                                ws2.write_blank(curr_r, 1, "", get_fmt({'border':0}, left=2))
                                ws2.write_blank(curr_r, 13, "", get_fmt({'border':0}, right=2))
                                curr_r += 1
                            
                            # 5. 공백 1행 (Row 8) - 하단 마감 체크
                            is_final_row = (curr_r == l_f_row - 1) # 여기가 전체 시트의 마지막 줄인가?
                            
                            # 좌측 B열 마감
                            b_fmt = get_fmt({'border':0}, left=2, bottom=2) if is_final_row else get_fmt({'border':0}, left=2)
                            ws2.write_blank(curr_r, 1, "", b_fmt)
                            
                            # 중앙 바닥 마감 (마지막 줄일 때만)
                            if is_final_row:
                                for c in range(2, 13): ws2.write_blank(curr_r, c, "", get_fmt({'border':0}, bottom=2))
                            
                            # 우측 N열 마감
                            n_fmt = get_fmt({'border':0}, right=2, bottom=2) if is_final_row else get_fmt({'border':0}, right=2)
                            ws2.write_blank(curr_r, 13, "", n_fmt)
                            
                            curr_r += 1
                return output.getvalue()

            st.sidebar.markdown("---")
            img_b64 = get_base64_image("excel_icon.png")
            if img_b64:
                st.sidebar.markdown(f'<div style="display: flex; align-items: center; margin-bottom: 10px;"><img src="data:image/png;base64,{img_b64}" width="38" style="margin-right: 12px;"><span style="font-size: 24px; font-weight: 700; color: #31333f;">Excel Export</span></div>', unsafe_allow_html=True)
            else: st.sidebar.header("📊 Excel Export")
            st.sidebar.download_button(label="📥 Download Report", data=generate_excel(), file_name=f"Report_{detected_pn}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
else: st.info("사이드바에서 CSV 로그 파일을 업로드하세요.")