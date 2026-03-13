import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# 페이지 설정
st.set_page_config(page_title="MIC LOG Analyzer", layout="wide")

st.title("🎙️ MIC LOG 분석 자동화 툴 (Multi-Model)")
st.markdown("---")

# 1. 제품군 설정 정보 (Config)
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

# 2. 사이드바: 모델 선택 및 파일 업로드
st.sidebar.header("🛠️ 모델 및 데이터 설정")
model_type = st.sidebar.selectbox("제품 모델을 선택하세요.", options=["3903", "3203"])
uploaded_file = st.sidebar.file_uploader(f"[{model_type}] CSV 파일을 업로드하세요.", type=['csv'])

st.sidebar.markdown("---")
st.sidebar.header("🔍 필터 설정")
show_normal = st.sidebar.checkbox("정상 시료 표시 (Normal)", value=True)
show_specout = st.sidebar.checkbox("Spec Out 시료 표시", value=True)

# 3. 유틸리티 함수
def get_freq_values(cols):
    # 컬럼 이름에서 숫자(주파수)만 추출
    return [float(str(c).split('.')[0]) for c in cols]

def check_channel_status(row, cols, mic_type, l_low_row, l_high_row):
    val = pd.to_numeric(row[cols], errors='coerce')
    l_low = pd.to_numeric(l_low_row[cols], errors='coerce')
    l_high = pd.to_numeric(l_high_row[cols], errors='coerce')
    
    # 타입별 Outlier 기준 설정
    out_min, out_max = (-30, 0) if mic_type == 'analog' else (-45, -25)
    
    is_out = ((val < out_min) | (val > out_max)).any()
    is_spec = ((val < l_low) | (val > l_high)).any()
    return is_out, is_spec

# 4. 분석 및 시각화 프로세스
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, low_memory=False)
    config = PRODUCT_CONFIGS[model_type]
    
    sn_col = 'Unnamed: 3'
    limit_low = df.iloc[0]
    limit_high = df.iloc[1]
    
    # 데이터 정제: 시리얼 번호 없는 행 드랍
    test_data = df.iloc[2:].copy()
    test_data = test_data.dropna(subset=[sn_col]).reset_index(drop=True)
    
    # 시료 분류
    normal_indices = []
    specout_indices = []
    outlier_sns = []

    for idx, row in test_data.iterrows():
        is_any_out = False
        is_any_spec = False
        
        for ch in config["channels"]:
            cols = df.columns[ch["range"]]
            out_flag, spec_flag = check_channel_status(row, cols, ch["type"], limit_low, limit_high)
            if out_flag: is_any_out = True
            if spec_flag: is_any_spec = True
            
        sn = str(row[sn_col]).strip()
        if is_any_out:
            outlier_sns.append(sn)
        elif is_any_spec:
            specout_indices.append(idx)
        else:
            normal_indices.append(idx)

    # 대시보드 요약
    c1, c2, c3 = st.columns(3)
    c1.metric("총 시료 수", len(test_data))
    c2.metric("Spec Out 수", len(specout_indices))
    c3.metric("Outlier 수", len(outlier_sns))

    if outlier_sns:
        with st.expander("⚠️ Outlier 시리얼 리스트"):
            st.write(", ".join(outlier_sns))

    # 5. 시각화 함수
    def plot_mic_fr(ax, ch_info):
        cols = df.columns[ch_info["range"]]
        freqs = get_freq_values(cols)
        
        if ch_info["type"] == 'analog':
            ylim, color, unit = (-30, 0), 'green', 'dbV'
        else:
            ylim, color, unit = (-45, -25), 'blue', 'dbFS'
            
        major_ticks = [100, 200, 1000, 4000, 10000]

        # 배경: 정상 데이터
        if show_normal and normal_indices:
            for i in normal_indices:
                y = pd.to_numeric(test_data.loc[i, cols], errors='coerce')
                ax.plot(freqs, y, color=color, alpha=0.05, linewidth=0.5)
        
        # 강조: Spec Out 데이터
        if show_specout and specout_indices:
            for i in specout_indices:
                sn = str(test_data.loc[i, sn_col]).strip()
                y = pd.to_numeric(test_data.loc[i, cols], errors='coerce')
                ax.plot(freqs, y, label=f"SpecOut: {sn}", linewidth=1.5)

        # 리미트 가이드라인
        ax.plot(freqs, pd.to_numeric(limit_low[cols], errors='coerce'), 'r--', label='Lower Limit', alpha=0.7)
        ax.plot(freqs, pd.to_numeric(limit_high[cols], errors='coerce'), 'r--', label='Upper Limit', alpha=0.7)
        
        ax.set_xscale('log')
        ax.set_ylim(ylim)
        ax.set_xticks(major_ticks)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax.set_title(ch_info["name"], fontsize=14, fontweight='bold')
        ax.set_ylabel(f'Response ({unit})')
        ax.grid(True, which="both", ls="-", alpha=0.3)
        if show_specout and specout_indices:
            ax.legend(fontsize='x-small', loc='lower right')

    # 그래프 출력
    st.subheader(f"📊 {model_type} 주파수 응답 분석 결과")
    num_channels = len(config["channels"])
    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 6 * num_channels))
    
    # 채널이 1개일 때 axes가 배열이 아닌 점 처리
    if num_channels == 1: axes = [axes]
    
    for i, ch in enumerate(config["channels"]):
        plot_mic_fr(axes[i], ch)
        
    plt.tight_layout()
    st.pyplot(fig)

else:
    st.info("사이드바에서 제품 모델을 확인하고 CSV 파일을 업로드해 주세요.")