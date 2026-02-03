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
def detect_info(df):
    model, prod_date, matched_pn = None, "Unknown", "Unknown"
    try:
        for i in range(2, min(12, len(df))):
            sn_raw = str(df.iloc[i, 3]).strip()
            if '/' in sn_raw:
                parts = sn_raw.split('/')
                pn_part = parts[0]
                sn_part = parts[1]
                for m, info in PRODUCT_CONFIGS.items():
                    if pn_part in info["pn"]:
                        model = m
                        matched_pn = pn_part
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
if detected_model:
    st.sidebar.success(f"✅ 모델 인식: {detected_model}")
model_type = st.sidebar.selectbox("제품 모델 선택", options=model_list, index=default_idx)

# --- [시각화 옵션] ---
st.sidebar.markdown("---")
st.sidebar.header("🔍 시각화 옵션")
show_detail_table = st.sidebar.checkbox("상세 테이블 표시", value=True)
show_fr_plot = st.sidebar.checkbox("주파수 응답(FR) 그래프 표시", value=True)
show_dist_plot = st.sidebar.checkbox("정규분포 그래프 표시", value=False)
st.sidebar.markdown("---")
st.sidebar.header("✔️ 정상 시료 설정")
show_normal = st.sidebar.checkbox("정상 시료 FR 표시", value=True)

# --- [유틸리티 및 그래프 함수] ---
def get_freq_values(cols): return [float(str(c).split('.')[0]) for c in cols]
def get_channel_status(row, cols, mic_type, l_low_row, l_high_row):
    val = pd.to_numeric(row[cols], errors='coerce')
    l_low, l_high = pd.to_numeric(l_low_row[cols], errors='coerce'), pd.to_numeric(l_high_row[cols], errors='coerce')
    out_min, out_max = (-30, 0) if mic_type == 'analog' else (-45, -25)
    if ((val < out_min) | (val > out_max)).any(): return "Outlier"
    if ((val < l_low) | (val > l_high)).any(): return "Spec Out"
    return "OK"

# 그래프 생성 함수 (create_fr_plot, plot_bell_curve 등 기존 유지)
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

# --- [메인 분석 프로세스] ---
if df is not None:
    config = PRODUCT_CONFIGS[model_type]
    sn_col_name = df.columns[3]
    limit_low, limit_high = df.iloc[0], df.iloc[1]
    raw_test_data = df.iloc[2:].dropna(subset=[sn_col_name])
    test_data = raw_test_data[raw_test_data[sn_col_name].str.contains('/', na=False)].reset_index(drop=True)

    if len(test_data) > 0:
        sample_info, issue_indices, normal_indices = {}, [], []
        # 채널별 통계 초기화 (Pass/Fail 추가)
        ch_stats = {ch["name"]: {"pass": 0, "fail": 0, "vals_1k": []} for ch in config["channels"]}

        for idx, row in test_data.iterrows():
            is_sample_issue, row_table = False, []
            for ch in config["channels"]:
                cols = df.columns[ch["range"]]
                status = get_channel_status(row, cols, ch["type"], limit_low, limit_high)
                
                # 1kHz 데이터 추출
                freqs = get_freq_values(cols)
                idx_1k = np.argmin(np.abs(np.array(freqs) - 1000))
                val_1k = pd.to_numeric(row[cols[idx_1k]], errors='coerce')
                ch_stats[ch["name"]]["vals_1k"].append(val_1k)

                if status == "OK":
                    ch_stats[ch["name"]]["pass"] += 1
                else:
                    is_sample_issue = True
                    ch_stats[ch["name"]]["fail"] += 1
                
                thd_val = row[df.columns[ch["thd_idx"]]] if ch["thd_idx"] else "-"
                row_table.append({"Channel": ch["name"], "1000Hz": f"{val_1k:.2f}", "THD": f"{thd_val:.2f}", "Status": status})

            sample_info[idx] = {"table": pd.DataFrame(row_table), "sn": str(row[sn_col_name]).strip()}
            if is_sample_issue: issue_indices.append(idx)
            else: normal_indices.append(idx)

        # --- [대시보드 요약 섹션] ---
        st.subheader("🚀 Production Dashboard")
        d_col1, d_col2, d_col3 = st.columns([1.2, 1, 2.8])
        
        with d_col1:
            st.markdown(f"**Model P/N:** `{detected_pn}`")
            st.markdown(f"**Prod. Date:** `{prod_date}`")
        
        with d_col2:
            total_qty = len(test_data)
            total_fail = len(issue_indices)
            total_pass = total_qty - total_fail
            yield_val = (total_pass / total_qty) * 100
            
            # [개선] 전문적인 배지 스타일 UI
            st.markdown(f"""
                <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 10px;">
                    <div style="background-color: #f0f2f6; padding: 5px 15px; border-radius: 10px; border-left: 5px solid #00c853;">
                        <p style="margin:0; font-size:12px; color:#5e636e;">PASS</p>
                        <p style="margin:0; font-size:20px; font-weight:bold; color:#00c853;">{total_pass}</p>
                    </div>
                    <div style="background-color: #f0f2f6; padding: 5px 15px; border-radius: 10px; border-left: 5px solid #ff1744;">
                        <p style="margin:0; font-size:12px; color:#5e636e;">FAIL</p>
                        <p style="margin:0; font-size:20px; font-weight:bold; color:#ff1744;">{total_fail}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # 수율 표시 (metric 유지 혹은 커스텀 스타일)
            st.metric("Overall Yield", f"{yield_val:.1f}%")
            # 수율 수치 아래에 바로 추가 가능
            st.progress(yield_val / 100)

        with d_col3:
            # 채널별 통계 테이블 구성 (Pass/Fail 수량 포함)
            ch_summary = []
            for ch_name, stat in ch_stats.items():
                v = np.array(stat["vals_1k"])
                v = v[~np.isnan(v)]
                ch_summary.append({
                    "Channel": ch_name,
                    "Pass": stat["pass"],
                    "Fail": stat["fail"],
                    "Yield(%)": f"{(stat['pass']/(stat['pass']+stat['fail']))*100:.1f}%",
                    "Min": f"{v.min():.2f}",
                    "Max": f"{v.max():.2f}",
                    "Avg": f"{v.mean():.2f}",
                    "Stdev": f"{v.std():.2f}"
                })
            st.dataframe(pd.DataFrame(ch_summary), hide_index=True, use_container_width=True)
        st.markdown("---")

        # --- [시각화 섹션] ---
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
else:
    st.info("사이드바에서 CSV 파일을 업로드해 주세요.")