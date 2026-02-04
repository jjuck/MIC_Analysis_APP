import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import io
import chardet
import os

# 1. 페이지 설정 및 UI 일관성 상수
st.set_page_config(page_title="MIC Analysis Tool", page_icon="🎙️", layout="wide")

FIG_WIDTH = 12       
PLOT_HEIGHT = 6      
FONT_SIZE_TITLE = 16 
FONT_SIZE_AXIS = 12  

# --- [상단 헤더: 고정] ---
col_head1, col_head2 = st.columns([4, 1], vertical_alignment="center")
with col_head1:
    st.markdown("<h1>🎙️ MIC Analysis Tool <span style='font-size: 16px; color: gray; font-weight: normal; margin-left: 10px;'>( 제작 : JW Lee, 자문 : JJ Kim )</span></h1>", unsafe_allow_html=True)
with col_head2:
    if os.path.exists("logo.png"): st.image("logo.png", width=300)
st.markdown("---")

# 2. 제품군 설정
PRODUCT_CONFIGS = {
    "RH": {"pn": ["96575N1100", "96575GJ100"], "channels": [{"name": "Digital Ch1", "type": "digital", "range": range(51, 101), "thd_idx": 15}, {"name": "Digital Ch2", "type": "digital", "range": range(103, 153), "thd_idx": 18}, {"name": "Digital Ch3", "type": "digital", "range": range(155, 205), "thd_idx": 21}]},
    "3903(LH Ecall)": {"pn": ["96575N1050", "96575GJ000"], "channels": [{"name": "Ecall Mic (Analog)", "type": "analog", "range": range(6, 47), "thd_idx": 69}, {"name": "Digital Ch1", "type": "digital", "range": range(107, 157), "thd_idx": 217}, {"name": "Digital Ch2", "type": "digital", "range": range(159, 209), "thd_idx": 220}]},
    "3203(LH non Ecall)": {"pn": ["96575N1000", "96575GJ010"], "channels": [{"name": "Digital Ch1", "type": "digital", "range": range(6, 56), "thd_idx": 116}, {"name": "Digital Ch2", "type": "digital", "range": range(58, 108), "thd_idx": 119}]},
    "LITE(LH)": {"pn": ["96575NR000", "96575GJ200"], "channels": [{"name": "Analog Mic", "type": "analog", "range": range(6, 47), "thd_idx": 95}]},
    "LITE(RH)": {"pn": ["96575NR100", "96575GJ300"], "channels": [{"name": "Analog Mic", "type": "analog", "range": range(6, 47), "thd_idx": 95}]}
}

# 3. [함수 정의 영역]
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
                parts = sn_raw.split('/')
                pn_part = parts[0]
                for m, info in PRODUCT_CONFIGS.items():
                    if pn_part in info["pn"]:
                        model, matched_pn = m, pn_part
                        break
                sn_part = parts[1]
                if len(sn_part) >= 8:
                    prod_date = f"20{sn_part[2:4]}/{sn_part[4:6]}/{sn_part[6:8]}"
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
    cols = all_cols[ch_info["range"]]
    freqs = get_freq_values(cols)
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
        data["THD (1kHz, %)"] = f"{thd_val:.3f}" if not pd.isna(thd_val) else "-"
    else: data["THD (1kHz, %)"] = "N/A"
    return data

def create_fr_plot(config, df, current_test_data, limit_low, limit_high, show_normal, plotting_normal_indices, highlight_indices):
    num_ch = len(config["channels"])
    fig, axes = plt.subplots(num_ch, 1, figsize=(FIG_WIDTH, PLOT_HEIGHT * num_ch))
    if num_ch == 1: axes = [axes]
    for i, ch in enumerate(config["channels"]):
        ax, cols = axes[i], df.columns[ch["range"]]
        freqs = get_freq_values(cols)
        ylim, color, unit = ((-30, 0), 'green', 'dbV') if ch["type"] == 'analog' else ((-45, -25), 'blue', 'dbFS')
        ax.set_xscale('log')
        target_ticks = [50, 100, 200, 1000, 4000, 10000, 14000]
        ax.set_xticks(target_ticks)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.minorticks_off()
        if show_normal:
            for n in plotting_normal_indices: ax.plot(freqs, pd.to_numeric(current_test_data.loc[n, cols], errors='coerce'), color=color, alpha=0.7, lw=1.2)
        for h in highlight_indices: ax.plot(freqs, pd.to_numeric(current_test_data.loc[h, cols], errors='coerce'), color='red', lw=2.5)
        ax.plot(freqs, pd.to_numeric(limit_low[cols], errors='coerce'), 'k--', lw=1.2)
        ax.plot(freqs, pd.to_numeric(limit_high[cols], errors='coerce'), 'k--', lw=1.2)
        ax.set_title(ch["name"], fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        ax.set_ylabel(f'Response ({unit})', fontsize=FONT_SIZE_AXIS)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_AXIS)
        ax.set_ylim(ylim); ax.grid(True, which='major', linestyle='-', alpha=0.4, color='#bdc3c7')
    plt.tight_layout(); return fig

def plot_bell_curve(ax, data_series, stats_indices, selected_indices, title, mic_type):
    v_raw = pd.to_numeric(data_series.iloc[stats_indices], errors='coerce')
    v_clean = v_raw.dropna()
    lcl, ucl = (-11, -9) if mic_type == 'analog' else (-38, -36)
    if len(v_clean) < 2:
        ax.set_title(f"{title} (Insufficient Data)", fontsize=10); return
    mu, std = v_clean.mean(), v_clean.std()
    cpk = min((ucl-mu)/(3*std), (mu-lcl)/(3*std)) if std > 0 else 0
    x_range = np.linspace(lcl - 2, ucl + 2, 200)
    p = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu) / std)**2)
    ax.plot(x_range, p, 'k', lw=2.5, alpha=0.7); ax.fill_between(x_range, p, color='gray', alpha=0.1)
    ax.axvline(lcl, color='blue', ls='--', lw=1.5, label=f'LSL ({lcl})')
    ax.axvline(ucl, color='red', ls='--', lw=1.5, label=f'USL ({ucl})')
    if selected_indices:
        sel_vals = pd.to_numeric(data_series.iloc[selected_indices], errors='coerce').dropna()
        for v in sel_vals:
            y_pos = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((v - mu) / std)**2)
            ax.scatter(v, y_pos, color='red', s=100, edgecolors='white', zorder=5)
    
    # [수정] Sample N 문구 제거
    stats_txt = f"Mean: {mu:.2f}\nStd: {std:.3f}\nCpk: {cpk:.2f}"
    ax.text(0.95, 0.75, stats_txt, transform=ax.transAxes, ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=FONT_SIZE_AXIS, fontweight='bold')
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.tick_params(labelsize=FONT_SIZE_AXIS); ax.set_xlim(lcl - 2, ucl + 2)
    ax.legend(loc='upper right', fontsize=FONT_SIZE_AXIS - 2); ax.grid(True, alpha=0.2)

# 4. [메인 프로세스]
st.sidebar.header("🛠️ 모델 및 데이터 설정")
uploaded_file = st.sidebar.file_uploader("CSV 로그 파일을 업로드하세요.", type=['csv'])

if uploaded_file:
    raw_bytes = uploaded_file.read()
    det_enc = chardet.detect(raw_bytes)['encoding']
    enc_list = [det_enc, 'utf-8-sig', 'gbk', 'cp949', 'utf-8']
    df = None
    for e in enc_list:
        try:
            df = pd.read_csv(io.StringIO(raw_bytes.decode(e, errors='replace')), low_memory=False)
            df.columns = [str(c) for c in df.columns]
            break
        except: continue

    if df is not None:
        detected_model, prod_date, detected_pn = detect_info(df)
        model_list = list(PRODUCT_CONFIGS.keys())
        model_type = st.sidebar.selectbox("제품 모델 선택", options=model_list, index=model_list.index(detected_model) if detected_model else 0)
        st.sidebar.markdown("---")
        
        # [수정] 정상 시료 설정 문구 복원
        st.sidebar.header("✔️ 정상 시료 설정")
        show_normal = st.sidebar.checkbox("정상 시료 FR 표시", value=True)
        
        config = PRODUCT_CONFIGS[model_type]
        sn_col, limit_low, limit_high = df.columns[3], df.iloc[0], df.iloc[1]
        raw_test_data = df.iloc[2:].dropna(subset=[sn_col])
        test_data = raw_test_data[raw_test_data[sn_col].astype(str).str.contains('/', na=False)].reset_index(drop=True)

        if len(test_data) > 0:
            sample_info, issue_indices, stats_indices, plotting_normal_indices = {}, [], [], []
            ch_stats = {ch["name"]: {"pass": 0, "fail": 0, "vals_1k": []} for ch in config["channels"]}

            for idx, row in test_data.iterrows():
                is_any_defect, is_any_fail, is_pure_normal = False, False, True
                row_table = []
                for ch in config["channels"]:
                    cols = df.columns[ch["range"]]
                    freqs = get_freq_values(cols)
                    status = classify_sample(row, cols, freqs, limit_low, limit_high)
                    if status == "Defect": is_any_defect = True
                    if status != "Normal": is_any_fail, is_pure_normal = True, False
                    
                    idx_1k = np.argmin(np.abs(np.array(freqs) - 1000))
                    val_1k = pd.to_numeric(row[cols[idx_1k]], errors='coerce')
                    ch_stats[ch["name"]]["vals_1k"].append(val_1k)
                    if status == "Normal": ch_stats[ch["name"]]["pass"] += 1
                    else: ch_stats[ch["name"]]["fail"] += 1
                    
                    summary = get_row_summary_data(row, ch, df.columns)
                    summary["Status"] = status
                    row_table.append(summary)

                sample_info[idx] = {"table": pd.DataFrame(row_table), "sn": clean_sn(row[sn_col])}
                if is_any_fail: issue_indices.append(idx)
                if not is_any_defect: stats_indices.append(idx)
                if is_pure_normal: plotting_normal_indices.append(idx)

            # --- [상단 고정: Dashboard] ---
            # [수정] 이모티콘 변경: 🚀 -> 📝
            st.subheader("📝 Production Dashboard")
            d1, d2, d3 = st.columns([1.2, 1.3, 2.5])
            with d1: st.markdown(f"**Model P/N:** `{detected_pn}`\n\n**Prod. Date:** `{prod_date}`\n\n**Quantity:** `{len(test_data)} EA`")
            with d2:
                total_qty, total_fail = len(test_data), len(issue_indices)
                total_pass, yield_val = total_qty - total_fail, ((total_qty - total_fail) / total_qty * 100)
                st.markdown(f"""<div style="display: flex; gap: 8px; margin-bottom: 5px;"><div style="background-color: #f8f9fa; padding: 10px 15px; border-radius: 10px; border-left: 5px solid #28a745; flex: 1;"><p style="margin:0; font-size:11px; color:#6c757d; font-weight:bold;">PASS</p><p style="margin:0; font-size:20px; font-weight:800; color:#28a745;">{total_pass}</p></div><div style="background-color: #f8f9fa; padding: 10px 15px; border-radius: 10px; border-left: 5px solid #dc3545; flex: 1;"><p style="margin:0; font-size:11px; color:#6c757d; font-weight:bold;">FAIL</p><p style="margin:0; font-size:20px; font-weight:800; color:#dc3545;">{total_fail}</p></div></div>""", unsafe_allow_html=True)
                
                # [수정] Overall Yield 폰트 크기 증대 (14px -> 20px)
                st.markdown(f"<p style='margin-bottom:-10px; font-weight:bold; font-size:20px;'>Overall Yield: {yield_val:.1f}%</p>", unsafe_allow_html=True)
                st.markdown(f"""<div style="width: 100%; background-color: #e0e0e0; border-radius: 5px; border: 1px solid #bdc3c7; margin-top: 10px;"><div style="width: {yield_val}%; background-color: #2ecc71; height: 12px; border-radius: 4px;"></div></div>""", unsafe_allow_html=True)
            with d3:
                ch_summary = []
                for ch_name, stat in ch_stats.items():
                    v_stats = np.array(stat["vals_1k"])[stats_indices]
                    v = v_stats[~np.isnan(v_stats)]
                    v_min, v_max, v_avg, v_std = (v.min(), v.max(), v.mean(), v.std()) if len(v) > 0 else (0,0,0,0)
                    ch_summary.append({"Channel": ch_name, "Pass": stat["pass"], "Fail": stat["fail"], "Yield": f"{(stat['pass']/total_qty)*100:.1f}%", "Min": f"{v_min:.2f}", "Max": f"{v_max:.2f}", "Avg": f"{v_avg:.2f}", "Stdev": f"{v_std:.2f}"})
                st.dataframe(pd.DataFrame(ch_summary), hide_index=True, use_container_width=True)
            
            st.markdown("---")

            # --- [탭 시스템: 시각화 및 상세 분석] ---
            # [수정] 이모티콘 변경: 📊 -> 📈
            tab_fr, tab_dist, tab_detail = st.tabs(["📈 주파수 응답 (FR)", "📉 정규분포 (Cpk)", "🔍 결함 시료 상세"])

            st.sidebar.markdown("---")
            st.sidebar.header("❌️ 결함 시료 선택")
            sel_idx = [i for i in issue_indices if st.sidebar.checkbox(f"SN: {sample_info[i]['sn']}", key=f"ch_{i}")]

            with tab_fr:
                st.subheader(f"📈 {model_type} Frequency Response")
                st.pyplot(create_fr_plot(config, df, test_data, limit_low, limit_high, show_normal, plotting_normal_indices, sel_idx))

            with tab_dist:
                st.subheader("📉 1kHz Sensitivity Distribution")
                fig_d, axes_d = plt.subplots(len(config["channels"]), 1, figsize=(FIG_WIDTH, PLOT_HEIGHT * len(config["channels"])))
                if len(config["channels"]) == 1: axes_d = [axes_d]
                for i, ch in enumerate(config["channels"]):
                    col_idx = np.argmin(np.abs(np.array(get_freq_values(df.columns[ch["range"]])) - 1000))
                    plot_bell_curve(axes_d[i], test_data[df.columns[ch["range"]][col_idx]], stats_indices, sel_idx, f"{ch['name']} - Distribution", ch["type"])
                st.pyplot(fig_d)

            with tab_detail:
                if sel_idx:
                    st.info("🔍 선택된 결함 시료의 주파수 포인트별 상세 데이터입니다.")
                    for i in sel_idx:
                        st.write(f"📄 **Serial Number: {sample_info[i]['sn']}**")
                        st.table(sample_info[i]["table"][["Channel", "200Hz", "1000Hz", "4000Hz", "THD (1kHz, %)", "Status"]].set_index("Channel"))
                else:
                    st.warning("사이드바에서 분석할 결함 시료를 선택해 주세요.")

        else: st.warning("⚠️ 분석 가능한 유효 데이터를 찾을 수 없습니다.")
else: st.info("사이드바에서 CSV 로그 파일을 업로드하세요.")