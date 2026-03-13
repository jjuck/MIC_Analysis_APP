import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import io
import chardet
import os

# 페이지 설정
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

# 1. 제품군 및 P/N 매핑 정보
PRODUCT_CONFIGS = {
    "RH": {"pn": ["96575N1100", "96575GJ100"], "channels": [{"name": "Digital Ch1", "type": "digital", "range": range(51, 101), "thd_idx": 15}, {"name": "Digital Ch2", "type": "digital", "range": range(103, 153), "thd_idx": 18}, {"name": "Digital Ch3", "type": "digital", "range": range(155, 205), "thd_idx": 21}]},
    "3903(LH Ecall)": {"pn": ["96575N1050", "96575GJ000"], "channels": [{"name": "Ecall Mic (Analog)", "type": "analog", "range": range(6, 47), "thd_idx": 69}, {"name": "Digital Ch1", "type": "digital", "range": range(107, 157), "thd_idx": 217}, {"name": "Digital Ch2", "type": "digital", "range": range(159, 209), "thd_idx": 220}]},
    "3203(LH non Ecall)": {"pn": ["96575N1000", "96575GJ010"], "channels": [{"name": "Digital Ch1", "type": "digital", "range": range(6, 56), "thd_idx": 116}, {"name": "Digital Ch2", "type": "digital", "range": range(58, 108), "thd_idx": 119}]},
    "LITE(LH)": {"pn": ["96575NR000", "96575GJ200"], "channels": [{"name": "Analog Mic", "type": "analog", "range": range(6, 47), "thd_idx": 95}]},
    "LITE(RH)": {"pn": ["96575NR100", "96575GJ300"], "channels": [{"name": "Analog Mic", "type": "analog", "range": range(6, 47), "thd_idx": 95}]}
}

# 모델 감지 및 날짜 추출 함수
def detect_model_and_date(df):
    model, prod_date = None, "Unknown"
    try:
        for i in range(2, min(12, len(df))):
            sn_raw = str(df.iloc[i, 3]).strip()
            if '/' in sn_raw:
                parts = sn_raw.split('/')
                pn_part = parts[0]
                sn_part = parts[1]
                # 모델 매칭
                for m, info in PRODUCT_CONFIGS.items():
                    if pn_part in info["pn"]:
                        model = m
                        break
                # 날짜 추출 (ex: 6A260107... -> 2026/01/07)
                if len(sn_part) >= 8:
                    y, m, d = sn_part[2:4], sn_part[4:6], sn_part[6:8]
                    prod_date = f"20{y}/{m}/{d}"
                if model: break
    except: pass
    return model, prod_date

# --- [사이드바 구성] ---
st.sidebar.header("🛠️ 모델 및 데이터 설정")
uploaded_file = st.sidebar.file_uploader("CSV 로그 파일을 업로드하세요.", type=['csv'])

model_list = list(PRODUCT_CONFIGS.keys())
df, detected_model, prod_date = None, None, "Unknown"

if uploaded_file:
    raw_bytes = uploaded_file.read()
    det = chardet.detect(raw_bytes)
    encoding = det['encoding'] if det['encoding'] else 'utf-8'
    df = pd.read_csv(io.StringIO(raw_bytes.decode(encoding, errors='replace')), low_memory=False)
    detected_model, prod_date = detect_model_and_date(df)

default_idx = model_list.index(detected_model) if detected_model else 0
if detected_model:
    st.sidebar.success(f"✅ 모델 인식: {detected_model}")
model_type = st.sidebar.selectbox("제품 모델 선택", options=model_list, index=default_idx)

st.sidebar.markdown("---")
st.sidebar.header("🔍 시각화 옵션")
show_detail_table = st.sidebar.checkbox("상세 테이블 표시", value=True)
show_fr_plot = st.sidebar.checkbox("주파수 응답(FR) 그래프 표시", value=True)
show_dist_plot = st.sidebar.checkbox("정규분포 그래프 표시", value=False)

st.sidebar.markdown("---")
st.sidebar.header("✔️ 정상 시료 설정")
show_normal = st.sidebar.checkbox("정상 시료 FR 표시", value=True)

# --- [분석 함수들] ---
def get_freq_values(cols): return [float(str(c).split('.')[0]) for c in cols]

def get_channel_status(row, cols, mic_type, l_low_row, l_high_row):
    val = pd.to_numeric(row[cols], errors='coerce')
    l_low, l_high = pd.to_numeric(l_low_row[cols], errors='coerce'), pd.to_numeric(l_high_row[cols], errors='coerce')
    out_min, out_max = (-30, 0) if mic_type == 'analog' else (-45, -25)
    if ((val < out_min) | (val > out_max)).any(): return "Outlier"
    if ((val < l_low) | (val > l_high)).any(): return "Spec Out"
    return "OK"

# [그래프 관련 함수 생략 - 이전과 동일]
def plot_bell_curve(ax, data_series, normal_indices, selected_indices, title, mic_type):
    target_indices = list(normal_indices) + list(selected_indices)
    plot_data = pd.to_numeric(data_series.iloc[target_indices], errors='coerce').dropna()
    if mic_type == 'analog': lcl, ucl = -11, -9
    else: lcl, ucl = -38, -36
    if len(plot_data) < 2: return
    mu, std = plot_data.mean(), plot_data.std()
    x = np.linspace(lcl - 2, ucl + 2, 200)
    if std > 0:
        p = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std)**2)
        ax.plot(x, p, 'k', linewidth=2, alpha=0.6)
        ax.fill_between(x, p, color='gray', alpha=0.1)
    ax.axvline(lcl, color='blue', ls='--', lw=1.5); ax.axvline(ucl, color='red', ls='--', lw=1.5)
    ax.set_title(title, fontweight='bold')

def create_fr_plot(config, df, current_test_data, limit_low, limit_high, show_normal, normal_indices, highlight_indices):
    num_ch = len(config["channels"])
    fig, axes = plt.subplots(num_ch, 1, figsize=(10, 4 * num_ch))
    if num_ch == 1: axes = [axes]
    for i, ch in enumerate(config["channels"]):
        ax, cols = axes[i], df.columns[ch["range"]]
        freqs = get_freq_values(cols)
        ylim, color = ((-30, 0), 'green') if ch["type"] == 'analog' else ((-45, -25), 'blue')
        if show_normal:
            for n in normal_indices: ax.plot(freqs, pd.to_numeric(current_test_data.loc[n, cols], errors='coerce'), color=color, alpha=0.3, lw=0.8)
        for h in highlight_indices: ax.plot(freqs, pd.to_numeric(current_test_data.loc[h, cols], errors='coerce'), color='red', lw=2)
        ax.plot(freqs, pd.to_numeric(limit_low[cols], errors='coerce'), 'k--', lw=1)
        ax.plot(freqs, pd.to_numeric(limit_high[cols], errors='coerce'), 'k--', lw=1)
        ax.set_xscale('log'); ax.set_ylim(ylim); ax.set_title(ch["name"], fontweight='bold'); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig

# --- [메인 프로세스] ---
if df is not None:
    config = PRODUCT_CONFIGS[model_type]
    sn_col_name = df.columns[3]
    limit_low, limit_high = df.iloc[0], df.iloc[1]
    raw_test_data = df.iloc[2:].dropna(subset=[sn_col_name])
    test_data = raw_test_data[raw_test_data[sn_col_name].str.contains('/', na=False)].reset_index(drop=True)

    if len(test_data) > 0:
        # 데이터 집계
        sample_info, issue_indices, normal_indices = {}, [], []
        ch_stats = {ch["name"]: {"fail": 0, "vals_1k": []} for ch in config["channels"]}

        for idx, row in test_data.iterrows():
            is_sample_issue, row_table = False, []
            for ch in config["channels"]:
                cols = df.columns[ch["range"]]
                status = get_channel_status(row, cols, ch["type"], limit_low, limit_high)
                
                # 1kHz 감도 데이터 수집
                freqs = get_freq_values(cols)
                idx_1k = np.argmin(np.abs(np.array(freqs) - 1000))
                val_1k = pd.to_numeric(row[cols[idx_1k]], errors='coerce')
                ch_stats[ch["name"]]["vals_1k"].append(val_1k)

                if status != "OK":
                    is_sample_issue = True
                    ch_stats[ch["name"]]["fail"] += 1
                
                # 요약 데이터 생성 (테이블용)
                thd_val = row[df.columns[ch["thd_idx"]]] if ch["thd_idx"] else "-"
                row_table.append({"Channel": ch["name"], "1000Hz": f"{val_1k:.2f}", "THD": f"{thd_val:.2f}", "Status": status})

            sample_info[idx] = {"table": pd.DataFrame(row_table), "sn": str(row[sn_col_name]).strip()}
            if is_sample_issue: issue_indices.append(idx)
            else: normal_indices.append(idx)

        # --- [대시보드 영역] ---
        st.subheader("🚀 Production Dashboard")
        d_col1, d_col2, d_col3 = st.columns([1.5, 1, 2.5])
        
        with d_col1:
            st.markdown(f"**Model:** `{model_type}`")
            st.markdown(f"**Date:** `{prod_date}`")
        
        with d_col2:
            total = len(test_data)
            p_count = total - len(issue_indices)
            y_rate = (p_count / total) * 100
            st.metric("Total Yield", f"{y_rate:.1f}%", delta=f"{p_count}/{total} Pass")

        with d_col3:
            # 채널별 상세 통계 테이블 생성
            ch_summary = []
            for ch_name, stat in ch_stats.items():
                v = np.array(stat["vals_1k"])
                v = v[~np.isnan(v)]
                ch_summary.append({
                    "Channel": ch_name,
                    "Pass Rate": f"{((total-stat['fail'])/total)*100:.1f}%",
                    "Min": f"{v.min():.2f}",
                    "Max": f"{v.max():.2f}",
                    "Avg": f"{v.mean():.2f}",
                    "Stdev": f"{v.std():.2f}"
                })
            st.dataframe(pd.DataFrame(ch_summary), hide_index=True, use_container_width=True)
        st.markdown("---")

        # --- [시각화 영역] ---
        st.sidebar.markdown("---")
        st.sidebar.header("❌️ 결함 시료 선택")
        selected_indices = [i for i in issue_indices if st.sidebar.checkbox(f"SN: {sample_info[i]['sn']}", key=f"check_{i}")]

        if selected_indices and show_detail_table:
            st.info("🔍 선택 시료 상세 분석 테이블")
            for idx in selected_indices:
                st.write(f"📄 **SN: {sample_info[idx]['sn']}**")
                st.table(sample_info[idx]["table"].set_index("Channel"))

        if show_fr_plot:
            st.pyplot(create_fr_plot(config, df, test_data, limit_low, limit_high, show_normal, normal_indices, selected_indices))

        if show_dist_plot:
            st.info("📉 1kHz Sensitivity 정규분포")
            fig_d, axes_d = plt.subplots(len(config["channels"]), 1, figsize=(8, 3 * len(config["channels"])))
            if len(config["channels"]) == 1: axes_d = [axes_d]
            for i, ch in enumerate(config["channels"]):
                plot_bell_curve(axes_d[i], pd.Series(ch_stats[ch["name"]]["vals_1k"]), normal_indices, selected_indices, ch["name"], ch["type"])
            st.pyplot(fig_d)
else:
    st.info("사이드바에서 CSV 파일을 업로드해 주세요.")