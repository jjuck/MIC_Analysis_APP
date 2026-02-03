import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# 페이지 설정
st.set_page_config(page_title="MIC LOG Analyzer", layout="wide")

st.title("🎙️ MIC LOG 분석 자동화 툴 (Detail Data Table)")
st.markdown("---")

# 1. 제품군 설정 정보
PRODUCT_CONFIGS = {
    "3903": {
        "channels": [
            {"name": "Ecall FR (Analog)", "type": "analog", "range": range(6, 47)},
            {"name": "Digital Ch1 (Mic3)", "type": "digital", "range": range(107, 157)},
            {"name": "Digital Ch2 (Mic4)", "type": "digital", "range": range(159, 209)},
        ]
    },
    "3203": {
        "channels": [
            {"name": "Digital Ch1 (Mic1)", "type": "digital", "range": range(6, 56)},
            {"name": "Digital Ch2 (Mic2)", "type": "digital", "range": range(58, 108)},
        ]
    }
}

# 2. 사이드바 설정
st.sidebar.header("🛠️ 모델 및 데이터 설정")
model_type = st.sidebar.selectbox("제품 모델을 선택하세요.", options=["3903", "3203"])
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

# 특정 주파수 수치 추출 함수
def get_specific_freq_data(row, cols, freqs, target_list):
    data = {}
    for target in target_list:
        try:
            # 주파수 리스트에서 가장 가까운 값의 인덱스 찾기
            idx = np.argmin(np.abs(np.array(freqs) - target))
            val = row[cols[idx]]
            data[f"{target}Hz"] = round(float(val), 3)
        except:
            data[f"{target}Hz"] = "-"
    return data

# 4. 분석 프로세스
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, low_memory=False)
    config = PRODUCT_CONFIGS[model_type]
    sn_col = 'Unnamed: 3'
    limit_low, limit_high = df.iloc[0], df.iloc[1]
    
    test_data = df.iloc[2:].copy()
    test_data = test_data.dropna(subset=[sn_col]).reset_index(drop=True)
    
    # 시료 분류 및 정보 수집
    sample_info = {} # 상세 정보 저장용
    issue_indices = [] # Spec Out + Outlier 모두 포함
    normal_indices = []

    for idx, row in test_data.iterrows():
        ch_status_list = []
        is_issue = False
        row_table_data = []

        for ch in config["channels"]:
            cols = df.columns[ch["range"]]
            freqs = get_freq_values(cols)
            status = get_channel_status(row, cols, ch["type"], limit_low, limit_high)
            
            ch_name_simple = ch["name"].split('(')[0].strip()
            ch_status_list.append(f"{ch_name_simple}: **{status}**")
            
            if status != "OK": is_issue = True
            
            # 테이블용 수치 데이터 추출 (200, 1k, 4k)
            freq_vals = get_specific_freq_data(row, cols, freqs, [200, 1000, 4000])
            freq_vals["Channel"] = ch_name_simple
            freq_vals["Status"] = status
            row_table_data.append(freq_vals)
            
        sn = str(row[sn_col]).strip()
        summary_text = f"📄 **SN: {sn}** ｜ " + " , ".join(ch_status_list)
        
        sample_info[idx] = {"summary": summary_text, "table": pd.DataFrame(row_table_data)}
        
        if is_issue: issue_indices.append(idx)
        else: normal_indices.append(idx)

    # 사이드바: 결함 시료(Outlier + Spec Out) 선택
    selected_indices = []
    if issue_indices:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📍 결함 시료 선택 (Outlier/SpecOut)")
        for i in issue_indices:
            sn = str(test_data.loc[i, sn_col]).strip()
            if st.sidebar.checkbox(f"SN: {sn}", key=f"check_{i}"):
                selected_indices.append(i)

    # 상단 요약
    c1, c2 = st.columns(2)
    c1.metric("총 시료 수", len(test_data))
    c2.metric("결함 시료 수 (Outlier 포함)", len(issue_indices))

    # [수정 기능] 선택 시료 상세 요약 및 수치 테이블 출력
    if selected_indices:
        st.info("🔍 **선택 시료 상세 분석 (200Hz, 1kHz, 4kHz 수치)**")
        for idx in selected_indices:
            st.markdown(sample_info[idx]["summary"])
            st.table(sample_info[idx]["table"].set_index("Channel")) # 테이블 출력
    elif not issue_indices:
        st.success("✅ 모든 시료가 정상(OK)입니다.")

    # 5. 시각화 함수
    def plot_mic_fr(ax, ch_info):
        cols = df.columns[ch_info["range"]]
        freqs = get_freq_values(cols)
        ylim, color, unit = ((-30, 0), 'green', 'dbV') if ch_info["type"] == 'analog' else ((-45, -25), 'blue', 'dbFS')
        
        # 정상 시료
        if show_normal and normal_indices:
            for i in normal_indices:
                y = pd.to_numeric(test_data.loc[i, cols], errors='coerce')
                ax.plot(freqs, y, color=color, alpha=0.7, linewidth=1.5)
        
        # 선택된 결함 시료 (빨간색 강조)
        for i in selected_indices:
            sn = str(test_data.loc[i, sn_col]).strip()
            y = pd.to_numeric(test_data.loc[i, cols], errors='coerce')
            ax.plot(freqs, y, color='red', alpha=1.0, linewidth=2.5, label=f"Issue: {sn}")

        ax.plot(freqs, pd.to_numeric(limit_low[cols], errors='coerce'), 'k--', linewidth=1.2, label='Limit')
        ax.plot(freqs, pd.to_numeric(limit_high[cols], errors='coerce'), 'k--', linewidth=1.2)
        
        ax.set_xscale('log')
        ax.set_ylim(ylim)
        ax.set_xticks([100, 200, 1000, 4000, 10000])
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax.set_title(ch_info["name"], fontsize=14, fontweight='bold')
        ax.grid(True, which="both", ls="-", alpha=0.3)
        if selected_indices: ax.legend(fontsize='x-small', loc='lower right')

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