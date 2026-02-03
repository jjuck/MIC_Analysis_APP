import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import io
import chardet
import os

# 1. 페이지 설정
st.set_page_config(page_title="MIC Analysis Tool", page_icon="🎙️", layout="wide")

# --- [상단 헤더: 제목 - 로고 수직 중단 정렬] ---
col1, col2 = st.columns([4, 1], vertical_alignment="center")
with col1:
    st.markdown(
        """
        <h1 style='display: inline; margin: 0;'>🎙️ MIC Analysis Tool 
            <span style='font-size: 16px; color: gray; font-weight: normal; margin-left: 10px;'>
                ( 제작 : JW Lee, 자문 : JJ Kim )
            </span>
        </h1>
        """, 
        unsafe_allow_html=True
    )
with col2:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=300)
st.markdown("---")

# 2. 제품군 및 P/N 매핑 정보
PRODUCT_CONFIGS = {
    "RH": {"pn": ["96575N1100", "96575GJ100"], "channels": [{"name": "Digital Ch1", "type": "digital", "range": range(51, 101), "thd_idx": 15}, {"name": "Digital Ch2", "type": "digital", "range": range(103, 153), "thd_idx": 18}, {"name": "Digital Ch3", "type": "digital", "range": range(155, 205), "thd_idx": 21}]},
    "3903(LH Ecall)": {"pn": ["96575N1050", "96575GJ000"], "channels": [{"name": "Ecall Mic (Analog)", "type": "analog", "range": range(6, 47), "thd_idx": 69}, {"name": "Digital Ch1", "type": "digital", "range": range(107, 157), "thd_idx": 217}, {"name": "Digital Ch2", "type": "digital", "range": range(159, 209), "thd_idx": 220}]},
    "3203(LH non Ecall)": {"pn": ["96575N1000", "96575GJ010"], "channels": [{"name": "Digital Ch1", "type": "digital", "range": range(6, 56), "thd_idx": 116}, {"name": "Digital Ch2", "type": "digital", "range": range(58, 108), "thd_idx": 119}]},
    "LITE(LH)": {"pn": ["96575NR000", "96575GJ200"], "channels": [{"name": "Analog Mic", "type": "analog", "range": range(6, 47), "thd_idx": 95}]},
    "LITE(RH)": {"pn": ["96575NR100", "96575GJ300"], "channels": [{"name": "Analog Mic", "type": "analog", "range": range(6, 47), "thd_idx": 95}]}
}

# 3. 데이터 파싱 및 정보 추출 함수
def detect_info(df):
    model, prod_date, matched_pn = None, "Unknown", "Unknown"
    try:
        for i in range(2, min(12, len(df))):
            sn_raw = str(df.iloc[i, 3]).strip()
            if '/' in sn_raw:
                parts = sn_raw.split('/')
                pn_part, sn_part = parts[0], parts[1]
                for m, info in PRODUCT_CONFIGS.items():
                    if pn_part in info["pn"]:
                        model, matched_pn = m, pn_part
                        break
                if len(sn_part) >= 8:
                    y, m, d = sn_part[2:4], sn_part[4:6], sn_part[6:8]
                    prod_date = f"20{y}/{m}/{d}"
                if model: break
    except: pass
    return model, prod_date, matched_pn

# --- [사이드바 구성] ---
st.sidebar.header("🛠️ 모델 및 데이터 설정")
uploaded_file = st.sidebar.file_uploader("CSV 로그 파일을 업로드하세요.", type=['csv'])

model_list = list(PRODUCT_CONFIGS.keys())
df, detected_model, prod_date, detected_pn = None, None, "Unknown", "Unknown"

if uploaded_file:
    raw_bytes = uploaded_file.read()
    det = chardet.detect(raw_bytes)
    encoding = det['encoding'] if det['encoding'] else 'utf-8'
    df = pd.read_csv(io.StringIO(raw_bytes.decode(encoding, errors='replace')), low_memory=False)
    detected_model, prod_date, detected_pn = detect_info(df)

default_idx = model_list.index(detected_model) if detected_model else 0
if detected_model: st.sidebar.success(f"✅ 모델 자동 인식됨: {detected_model}")
model_type = st.sidebar.selectbox("제품 모델 선택", options=model_list, index=default_idx)

st.sidebar.markdown("---")
st.sidebar.header("🔍 시각화 옵션")
show_detail_table = st.sidebar.checkbox("상세 테이블 표시", value=True)
show_fr_plot = st.sidebar.checkbox("주파수 응답(FR) 그래프 표시", value=True)
show_dist_plot = st.sidebar.checkbox("정규분포 그래프 표시", value=False)
st.sidebar.markdown("---")
st.sidebar.header("✔️ 정상 시료 설정")
show_normal = st.sidebar.checkbox("정상 시료 FR 표시", value=True)

# ---------------------------------------------------------

# 4. 과거 버전 유틸리티 및 시각화 함수 (요청에 따라 롤백)
def get_freq_values(cols): return [float(str(c).split('.')[0]) for c in cols]

def get_channel_status(row, cols, mic_type, l_low_row, l_high_row):
    val = pd.to_numeric(row[cols], errors='coerce')
    l_low, l_high = pd.to_numeric(l_low_row[cols], errors='coerce'), pd.to_numeric(l_high_row[cols], errors='coerce')
    out_min, out_max = (-30, 0) if mic_type == 'analog' else (-45, -25)
    if ((val < out_min) | (val > out_max)).any(): return "Outlier"
    if ((val < l_low) | (val > l_high)).any(): return "Spec Out"
    return "OK"

def get_row_summary_data(row, ch_info, all_cols):
    cols = all_cols[ch_info["range"]]
    freqs = get_freq_values(cols)
    data = {"Channel": ch_info["name"]}
    for t in [200, 1000, 4000]:
        try:
            idx = np.argmin(np.abs(np.array(freqs) - t))
            data[f"{t}Hz"] = f"{float(row[cols[idx]]):.3f}"
        except: data[f"{t}Hz"] = "-"
    th_key = "THD (1kHz, %)"
    if ch_info["thd_idx"] is not None:
        try: data[th_key] = f"{float(row[all_cols[ch_info["thd_idx"]]]):.3f}"
        except: data[th_key] = "-"
    else: data[th_key] = "N/A"
    return data

def plot_bell_curve(ax, data_series, normal_indices, selected_indices, title, mic_type):
    target_indices = list(normal_indices) + list(selected_indices)
    plot_data = pd.to_numeric(data_series.iloc[target_indices], errors='coerce').dropna()
    if mic_type == 'analog':
        lcl, ucl = -11, -9
        clean_data = plot_data[(plot_data > -20) & (plot_data < 0)]
    else:
        lcl, ucl = -38, -36
        clean_data = plot_data[(plot_data > -45) & (plot_data < -25)]
    if len(clean_data) < 2: return
    mu, std = clean_data.mean(), clean_data.std()
    x_min, x_max = lcl - 2, ucl + 2
    x = np.linspace(x_min, x_max, 200)
    if std > 0:
        p = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std)**2)
        ax.plot(x, p, 'k', linewidth=2, alpha=0.6)
        ax.fill_between(x, p, color='gray', alpha=0.1)
    ax.axvline(lcl, color='blue', ls='--', lw=1.5, label=f'LCL ({lcl})')
    ax.axvline(ucl, color='red', ls='--', lw=1.5, label=f'UCL ({ucl})')
    if selected_indices and std > 0:
        sel_vals = pd.to_numeric(data_series.iloc[selected_indices], errors='coerce').dropna()
        for v in sel_vals:
            y_pos = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((v - mu) / std)**2)
            ax.scatter(v, y_pos, color='red', s=100, edgecolors='white', zorder=5)
    ax.set_title(title, fontweight='bold'); ax.set_xlim(x_min, x_max); ax.legend(fontsize=9)

def create_fr_plot(config, df, current_test_data, limit_low, limit_high, show_normal, normal_indices, highlight_indices):
    num_ch = len(config["channels"])
    fig, axes = plt.subplots(num_ch, 1, figsize=(10, 5 * num_ch))
    if num_ch == 1: axes = [axes]
    for i, ch in enumerate(config["channels"]):
        ax, cols = axes[i], df.columns[ch["range"]]
        freqs = get_freq_values(cols)
        ylim, color, unit = ((-30, 0), 'green', 'dbV') if ch["type"] == 'analog' else ((-45, -25), 'blue', 'dbFS')
        if show_normal:
            for n in normal_indices: ax.plot(freqs, pd.to_numeric(current_test_data.loc[n, cols], errors='coerce'), color=color, alpha=0.7, lw=1.2)
        for h in highlight_indices: ax.plot(freqs, pd.to_numeric(current_test_data.loc[h, cols], errors='coerce'), color='red', lw=2.5)
        ax.plot(freqs, pd.to_numeric(limit_low[cols], errors='coerce'), 'k--', lw=1.2)
        ax.plot(freqs, pd.to_numeric(limit_high[cols], errors='coerce'), 'k--', lw=1.2)
        ax.set_xscale('log'); ax.set_ylim(ylim); ax.set_title(ch["name"], fontweight='bold'); ax.set_ylabel(f'Response ({unit})'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# 5. 메인 분석 프로세스
if df is not None:
    config = PRODUCT_CONFIGS[model_type]
    sn_col_name = df.columns[3]
    limit_low, limit_high = df.iloc[0], df.iloc[1]
    raw_test_data = df.iloc[2:].dropna(subset=[sn_col_name])
    test_data = raw_test_data[raw_test_data[sn_col_name].str.contains('/', na=False)].reset_index(drop=True)

    if len(test_data) > 0:
        sample_info, issue_indices, normal_indices = {}, [], []
        ch_stats = {ch["name"]: {"pass": 0, "fail": 0, "vals_1k": []} for ch in config["channels"]}

        for idx, row in test_data.iterrows():
            is_sample_issue, row_table = False, []
            for ch in config["channels"]:
                cols = df.columns[ch["range"]]
                status = get_channel_status(row, cols, ch["type"], limit_low, limit_high)
                
                # 1kHz 산포용 데이터
                freqs = get_freq_values(cols)
                idx_1k = np.argmin(np.abs(np.array(freqs) - 1000))
                val_1k = pd.to_numeric(row[cols[idx_1k]], errors='coerce')
                ch_stats[ch["name"]]["vals_1k"].append(val_1k)

                if status == "OK": ch_stats[ch["name"]]["pass"] += 1
                else:
                    is_sample_issue = True
                    ch_stats[ch["name"]]["fail"] += 1
                
                row_table.append(get_row_summary_data(row, ch, df.columns))
                row_table[-1]["Status"] = status

            sample_info[idx] = {"table": pd.DataFrame(row_table), "sn": str(row[sn_col_name]).strip()}
            if is_sample_issue: issue_indices.append(idx)
            else: normal_indices.append(idx)

        # --- [대시보드 섹션] ---
        st.subheader("🚀 Production Dashboard")
        d_col1, d_col2, d_col3 = st.columns([1.2, 1.3, 2.5])
        
        with d_col1:
            st.markdown(f"**Model P/N:** `{detected_pn}`")
            st.markdown(f"**Prod. Date:** `{prod_date}`")
            st.markdown(f"**Quantity:** `{len(test_data)} EA`")

        with d_col2:
            total_qty = len(test_data)
            total_fail = len(issue_indices)
            total_pass = total_qty - total_fail
            yield_val = (total_pass / total_qty) * 100
            
            # Status Pill Cards
            st.markdown(f"""
                <div style="display: flex; gap: 8px; margin-bottom: 5px;">
                    <div style="background-color: #f8f9fa; padding: 8px 15px; border-radius: 10px; border-left: 5px solid #28a745; flex: 1;">
                        <p style="margin:0; font-size:11px; color:#6c757d; font-weight:bold;">PASS</p>
                        <p style="margin:0; font-size:20px; font-weight:800; color:#28a745;">{total_pass}</p>
                    </div>
                    <div style="background-color: #f8f9fa; padding: 8px 15px; border-radius: 10px; border-left: 5px solid #dc3545; flex: 1;">
                        <p style="margin:0; font-size:11px; color:#6c757d; font-weight:bold;">FAIL</p>
                        <p style="margin:0; font-size:20px; font-weight:800; color:#dc3545;">{total_fail}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # 수율 바 최적화
            st.markdown(f"<p style='margin-bottom:-10px; font-weight:bold; font-size:14px;'>Overall Yield: {yield_val:.1f}%</p>", unsafe_allow_html=True)
            st.markdown(f"""
                <div style="width: 100%; background-color: #e0e0e0; border-radius: 5px; border: 1px solid #bdc3c7; margin-top: 5px;">
                    <div style="width: {yield_val}%; background-color: #2ecc71; height: 12px; border-radius: 4px;"></div>
                </div>
                """, unsafe_allow_html=True)

        with d_col3:
            # 채널별 요약 테이블 (Min, Max, Avg, Stdev 포함)
            ch_summary = []
            for ch_name, stat in ch_stats.items():
                v = np.array(stat["vals_1k"]); v = v[~np.isnan(v)]
                ch_summary.append({
                    "Channel": ch_name, "Pass": stat["pass"], "Fail": stat["fail"],
                    "Yield": f"{(stat['pass']/total_qty)*100:.1f}%",
                    "Min": f"{v.min():.2f}", "Max": f"{v.max():.2f}",
                    "Avg": f"{v.mean():.2f}", "Stdev": f"{v.std():.2f}"
                })
            st.dataframe(pd.DataFrame(ch_summary), hide_index=True, use_container_width=True)
        st.markdown("---")

        # --- [시각화 결과 출력: 순서 및 스타일 롤백] ---
        st.sidebar.markdown("---")
        st.sidebar.header("❌️ 결함 시료 선택")
        selected_indices = [i for i in issue_indices if st.sidebar.checkbox(f"SN: {sample_info[i]['sn']}", key=f"check_{i}")]

        # 1. 상세 테이블
        if selected_indices and show_detail_table:
            st.info("🔍 **선택 시료 상세 분석 테이블**")
            for idx in selected_indices:
                st.write(f"📄 **SN: {sample_info[idx]['sn']}**")
                # 요청하신 과거 버전 컬럼 리스트 적용
                st.table(sample_info[idx]["table"][["Channel", "200Hz", "1000Hz", "4000Hz", "THD (1kHz, %)", "Status"]].set_index("Channel"))

        # 2. FR 그래프
        if show_fr_plot:
            st.subheader(f"📊 {model_type} 주파수 응답(FR) 분석")
            st.pyplot(create_fr_plot(config, df, test_data, limit_low, limit_high, show_normal, normal_indices, selected_indices))

        # 3. 정규분포 그래프
        if show_dist_plot:
            st.info("📉 **1kHz Sensitivity 정규분포 분석**")
            fig_d, axes_d = plt.subplots(len(config["channels"]), 1, figsize=(8, 4 * len(config["channels"])))
            if len(config["channels"]) == 1: axes_d = [axes_d]
            for i, ch in enumerate(config["channels"]):
                cols = df.columns[ch["range"]]
                idx_1k = np.argmin(np.abs(np.array(get_freq_values(cols)) - 1000))
                plot_bell_curve(axes_d[i], test_data[cols[idx_1k]], normal_indices, selected_indices, f"{ch['name']} - Distribution", ch["type"])
            st.pyplot(fig_d)
else:
    st.info("사이드바에서 CSV 파일을 업로드해 주세요.")