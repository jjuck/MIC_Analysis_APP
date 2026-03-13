import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# 페이지 설정
st.set_page_config(page_title="MIC LOG Analyzer", layout="wide")

st.title("🎙️ MIC LOG 분석 자동화 툴 (Advanced Reporting)")
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
    """채널별 상태를 문자열로 반환"""
    val = pd.to_numeric(row[cols], errors='coerce')
    l_low = pd.to_numeric(l_low_row[cols], errors='coerce')
    l_high = pd.to_numeric(l_high_row[cols], errors='coerce')
    
    out_min, out_max = (-30, 0) if mic_type == 'analog' else (-45, -25)
    
    # 1. Outlier 체크
    if ((val < out_min) | (val > out_max)).any():
        return "Outlier"
    
    # 2. Spec Out 체크
    if ((val < l_low) | (val > l_high)).any():
        return "Spec Out"
    
    return "OK"

# 4. 분석 프로세스
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, low_memory=False)
    config = PRODUCT_CONFIGS[model_type]
    
    sn_col = 'Unnamed: 3'
    limit_low = df.iloc[0]
    limit_high = df.iloc[1]
    
    # 데이터 정제
    test_data = df.iloc[2:].copy()
    test_data = test_data.dropna(subset=[sn_col]).reset_index(drop=True)
    
    # 시료별 채널 상태 분석
    sample_summary = {} # {index: "요약문구"}
    specout_indices = []
    normal_indices = []
    outlier_sns = []

    for idx, row in test_data.iterrows():
        ch_results = []
        is_any_out = False
        is_any_spec = False
        
        for ch in config["channels"]:
            status = get_channel_status(row, df.columns[ch["range"]], ch["type"], limit_low, limit_high)
            # 깔끔한 출력을 위해 채널명에서 괄호 부분 제거
            ch_name_simple = ch["name"].split('(')[0].strip()
            ch_results.append(f"{ch_name_simple}: **{status}**")
            
            if status == "Outlier": is_any_out = True
            if status == "Spec Out": is_any_spec = True
            
        sn = str(row[sn_col]).strip()
        summary_text = f"📄 **SN: {sn}** ｜ " + " , ".join(ch_results)
        sample_summary[idx] = summary_text
        
        if is_any_out:
            outlier_sns.append(sn)
        elif is_any_spec:
            specout_indices.append(idx)
        else:
            normal_indices.append(idx)

    # 사이드바: Spec Out 시료 개별 선택
    selected_spec_indices = []
    if specout_indices:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📍 Spec Out 시료 선택")
        for i in specout_indices:
            sn = str(test_data.loc[i, sn_col]).strip()
            if st.sidebar.checkbox(f"SN: {sn}", key=f"check_{i}"):
                selected_spec_indices.append(i)

    # 상단 요약 지표
    c1, c2, c3 = st.columns(3)
    c1.metric("총 시료 수", len(test_data))
    c2.metric("Spec Out 수", len(specout_indices))
    c3.metric("Outlier 수", len(outlier_sns))

    # [신규 기능] 선택 시료 상세 요약 출력
    if selected_spec_indices:
        st.info("💡 **선택 시료 상세 상태**")
        for idx in selected_spec_indices:
            st.write(sample_summary[idx])
    elif not outlier_sns and not specout_indices:
        st.success("✅ 모든 시료가 정상(OK)입니다.")

    # 5. 시각화 함수
    def plot_mic_fr(ax, ch_info):
        cols = df.columns[ch_info["range"]]
        freqs = get_freq_values(cols)
        
        if ch_info["type"] == 'analog':
            ylim, color, unit = (-30, 0), 'green', 'dbV'
        else:
            ylim, color, unit = (-45, -25), 'blue', 'dbFS'
            
        major_ticks = [100, 200, 1000, 4000, 10000]

        # 정상 시료 (기존 요청 스타일)
        if show_normal and normal_indices:
            for i in normal_indices:
                y = pd.to_numeric(test_data.loc[i, cols], errors='coerce')
                ax.plot(freqs, y, color=color, alpha=0.7, linewidth=1.5)
        
        # 선택된 Spec Out 시료 (기존 요청 스타일: 빨간색, 굵게)
        for i in selected_spec_indices:
            sn = str(test_data.loc[i, sn_col]).strip()
            y = pd.to_numeric(test_data.loc[i, cols], errors='coerce')
            ax.plot(freqs, y, color='red', alpha=1.0, linewidth=2.5, label=f"SpecOut: {sn}")

        # 임계값: 검은 점선
        ax.plot(freqs, pd.to_numeric(limit_low[cols], errors='coerce'), 'k--', linewidth=1.2, label='Limit')
        ax.plot(freqs, pd.to_numeric(limit_high[cols], errors='coerce'), 'k--', linewidth=1.2)
        
        ax.set_xscale('log')
        ax.set_ylim(ylim)
        ax.set_xticks(major_ticks)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax.set_title(ch_info["name"], fontsize=14, fontweight='bold')
        ax.set_ylabel(f'Response ({unit})')
        ax.grid(True, which="both", ls="-", alpha=0.3)
        if selected_spec_indices:
            ax.legend(fontsize='x-small', loc='lower right')

    # 그래프 출력
    st.subheader(f"📊 {model_type} 분석 결과 시각화")
    num_channels = len(config["channels"])
    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 6 * num_channels))
    
    if num_channels == 1: axes = [axes]
    for i, ch in enumerate(config["channels"]):
        plot_mic_fr(axes[i], ch)
        
    plt.tight_layout()
    st.pyplot(fig)

else:
    st.info("사이드바에서 제품 모델을 확인하고 CSV 파일을 업로드해 주세요.")