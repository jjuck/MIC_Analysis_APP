import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# 페이지 설정
st.set_page_config(page_title="MIC LOG Analyzer", layout="wide")

st.title("🎙️ MIC LOG 분석 자동화 툴")
st.markdown("---")

# 1. 제품군 설정 정보 (백엔드 매핑 인덱스 유지, UI 명칭 정제)
PRODUCT_CONFIGS = {
    "3903": {
        "channels": [
            {"name": "Ecall Mic (Analog)", "type": "analog", "range": range(6, 47), "thd_idx": None},
            {"name": "Digital Ch1", "type": "digital", "range": range(107, 157), "thd_idx": 217}, # HJ 매핑
            {"name": "Digital Ch2", "type": "digital", "range": range(159, 209), "thd_idx": 220}, # HM 매핑
        ]
    },
    "3203": {
        "channels": [
            {"name": "Digital Ch1", "type": "digital", "range": range(6, 56), "thd_idx": 116}, # DM 매핑
            {"name": "Digital Ch2", "type": "digital", "range": range(58, 108), "thd_idx": 119}, # DP 매핑
        ]
    },
    "RH": {
        "channels": [
            {"name": "Digital Ch1", "type": "digital", "range": range(51, 101), "thd_idx": 15}, # P 매핑
            {"name": "Digital Ch2", "type": "digital", "range": range(103, 153), "thd_idx": 18}, # S 매핑
            {"name": "Digital Ch3", "type": "digital", "range": range(155, 205), "thd_idx": 21}, # V 매핑
        ]
    }
}

# 2. 사이드바 설정
st.sidebar.header("🛠️ 모델 및 데이터 설정")
model_type = st.sidebar.selectbox("제품 모델을 선택하세요.", options=["3903", "3203", "RH"])
uploaded_file = st.sidebar.file_uploader(f"[{model_type}] CSV 파일을 업로드하세요.", type=['csv'])

st.sidebar.markdown("---")
st.sidebar.header("🔍 시각화 필터")
show_normal = st.sidebar.checkbox("정상 시료 표시 (Normal)", value=True)

# 3. 유틸리티 함수
def get_freq_values(cols):
    return [float(str(c).split('.')[0]) for c in cols]

def get_channel_status(row, cols, mic_type, l_low_row, l_high_row):
    val = pd.to_numeric(row[cols], errors='coerce')
    l_low = pd.to_numeric(l_low_row[cols], errors='coerce')
    l_high = pd.to_numeric(l_high_row[cols], errors='coerce')
    out_min, out_max = (-30, 0) if mic_type == 'analog' else (-45, -25)
    
    if ((val < out_min) | (val > out_max)).any(): return "Outlier"
    if ((val < l_low) | (val > l_high)).any(): return "Spec Out"
    return "OK"

def get_row_summary_data(row, ch_info, all_cols):
    cols = all_cols[ch_info["range"]]
    freqs = get_freq_values(cols)
    targets = [200, 1000, 4000]
    data = {"Channel": ch_info["name"]}
    
    for t in targets:
        try:
            idx = np.argmin(np.abs(np.array(freqs) - t))
            val = float(row[cols[idx]])
            data[f"{t}Hz"] = f"{val:.3f}"
        except:
            data[f"{t}Hz"] = "-"
            
    if ch_info["thd_idx"] is not None:
        try:
            thd_val = float(row[all_cols[ch_info["thd_idx"]]])
            data["THD (%)"] = f"{thd_val:.3f}"
        except:
            data["THD (%)"] = "-"
    else:
        data["THD (%)"] = "N/A"
        
    return data

# 4. 분석 프로세스
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, low_memory=False)
    config = PRODUCT_CONFIGS[model_type]
    sn_col = 'Unnamed: 3'
    limit_low, limit_high = df.iloc[0], df.iloc[1]
    
    test_data = df.iloc[2:].copy()
    test_data = test_data.dropna(subset=[sn_col]).reset_index(drop=True)
    
    sample_info = {}
    issue_indices = []
    normal_indices = []

    for idx, row in test_data.iterrows():
        ch_status_list = []
        is_issue = False
        row_table_data = []

        for ch in config["channels"]:
            status = get_channel_status(row, df.columns[ch["range"]], ch["type"], limit_low, limit_high)
            ch_status_list.append(f"{ch['name']}: **{status}**")
            
            if status != "OK": is_issue = True
            
            summary = get_row_summary_data(row, ch, df.columns)
            summary["Status"] = status
            row_table_data.append(summary)
            
        sn = str(row[sn_col]).strip()
        summary_text = f"📄 **SN: {sn}** ｜ " + " , ".join(ch_status_list)
        sample_info[idx] = {"summary": summary_text, "table": pd.DataFrame(row_table_data)}
        
        if is_issue: issue_indices.append(idx)
        else: normal_indices.append(idx)

    # 사이드바 선택 리스트
    selected_indices = []
    if issue_indices:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📍 결함 시료 선택 (Outlier 포함)")
        for i in issue_indices:
            sn = str(test_data.loc[i, sn_col]).strip()
            if st.sidebar.checkbox(f"SN: {sn}", key=f"check_{i}"):
                selected_indices.append(i)

    # 상세 데이터 테이블 출력 (정제된 명칭 사용)
    if selected_indices:
        st.info("🔍 **선택 시료 상세 분석 (소수점 3자리 및 THD 포함)**")
        for idx in selected_indices:
            st.markdown(sample_info[idx]["summary"])
            df_display = sample_info[idx]["table"][["Channel", "200Hz", "1000Hz", "4000Hz", "THD (%)", "Status"]]
            st.table(df_display.set_index("Channel"))

    # 5. 시각화 (그래프 스타일 및 축 설정)
    def plot_mic_fr(ax, ch_info):
        cols = df.columns[ch_info["range"]]
        freqs = get_freq_values(cols)
        ylim, color, unit = ((-30, 0), 'green', 'dbV') if ch_info["type"] == 'analog' else ((-45, -25), 'blue', 'dbFS')
        
        if show_normal and normal_indices:
            for i in normal_indices:
                y = pd.to_numeric(test_data.loc[i, cols], errors='coerce')
                ax.plot(freqs, y, color=color, alpha=0.7, linewidth=1.5)
        
        for i in selected_indices:
            y = pd.to_numeric(test_data.loc[i, cols], errors='coerce')
            ax.plot(freqs, y, color='red', alpha=1.0, linewidth=2.5)

        ax.plot(freqs, pd.to_numeric(limit_low[cols], errors='coerce'), 'k--', linewidth=1.2, label='Limit')
        ax.plot(freqs, pd.to_numeric(limit_high[cols], errors='coerce'), 'k--', linewidth=1.2)
        
        ax.set_xscale('log')
        ax.set_ylim(ylim)
        ax.set_xticks([100, 200, 1000, 4000, 10000])
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax.set_title(ch_info["name"], fontsize=14, fontweight='bold')
        ax.set_ylabel(f'Response ({unit})')
        ax.grid(True, which="both", ls="-", alpha=0.3)

    # 그래프 출력
    num_channels = len(config["channels"])
    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 6 * num_channels))
    if num_channels == 1: axes = [axes]
    for i, ch in enumerate(config["channels"]):
        plot_mic_fr(axes[i], ch)
    plt.tight_layout()
    st.pyplot(fig)

else:
    st.info("사이드바에서 제품 모델을 확인하고 CSV 파일을 업로드해 주세요.")