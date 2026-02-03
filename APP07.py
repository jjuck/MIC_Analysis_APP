import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import io
import chardet  # Notepad++와 유사한 지능형 인코딩 감지 라이브러리

# 페이지 설정
st.set_page_config(page_title="MIC LOG Analyzer", layout="wide")

st.title("🎙️ MIC LOG 분석 자동화 툴")
st.markdown("---")

# 1. 제품군 설정 정보
PRODUCT_CONFIGS = {
    "3903": {
        "channels": [
            {"name": "Ecall Mic (Analog)", "type": "analog", "range": range(6, 47), "thd_idx": None},
            {"name": "Digital Ch1", "type": "digital", "range": range(107, 157), "thd_idx": 217}, 
            {"name": "Digital Ch2", "type": "digital", "range": range(159, 209), "thd_idx": 220}, 
        ]
    },
    "3203": {
        "channels": [
            {"name": "Digital Ch1", "type": "digital", "range": range(6, 56), "thd_idx": 116}, 
            {"name": "Digital Ch2", "type": "digital", "range": range(58, 108), "thd_idx": 119}, 
        ]
    },
    "RH": {
        "channels": [
            {"name": "Digital Ch1", "type": "digital", "range": range(51, 101), "thd_idx": 15}, 
            {"name": "Digital Ch2", "type": "digital", "range": range(103, 153), "thd_idx": 18}, 
            {"name": "Digital Ch3", "type": "digital", "range": range(155, 205), "thd_idx": 21}, 
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
            data[f"{t}Hz"] = round(val, 3)
        except:
            data[f"{t}Hz"] = None
    if ch_info["thd_idx"] is not None:
        try:
            thd_val = float(row[all_cols[ch_info["thd_idx"]]])
            data["THD (%)"] = round(thd_val, 3)
        except:
            data["THD (%)"] = None
    return data

# 4. 분석 프로세스
if uploaded_file is not None:
    # [지능형 인코딩 변환 알고리즘]
    raw_data = uploaded_file.read()
    
    # 1. 바이트 패턴 분석을 통한 인코딩 감지
    detection = chardet.detect(raw_data)
    detected_enc = detection['encoding']
    confidence = detection['confidence']
    
    try:
        # 감지된 인코딩으로 디코딩 (실패 시 UTF-8 강제 시도)
        if detected_enc is not None and confidence > 0.5:
            decoded_content = raw_data.decode(detected_enc)
        else:
            decoded_content = raw_data.decode('utf-8', errors='replace')
    except:
        # 최후의 수단: charset-normalizer 사용 (더 강력한 엔진)
        from charset_normalizer import from_bytes
        decoded_content = str(from_bytes(raw_data).best())

    # UTF-8로 정규화된 텍스트 데이터를 로드
    df = pd.read_csv(io.StringIO(decoded_content), low_memory=False)

    config = PRODUCT_CONFIGS[model_type]
    sn_col = 'Unnamed: 3'
    limit_low, limit_high = df.iloc[0], df.iloc[1]
    
    test_data = df.iloc[2:].copy()
    test_data = test_data.dropna(subset=[sn_col]).reset_index(drop=True)
    
    sample_info, issue_indices, normal_indices, full_report_list = {}, [], [], []

    for idx, row in test_data.iterrows():
        ch_status_list, is_issue = [], False
        sn = str(row[sn_col]).strip()
        row_table_data = []

        for ch in config["channels"]:
            status = get_channel_status(row, df.columns[ch["range"]], ch["type"], limit_low, limit_high)
            ch_status_list.append(f"{ch['name']}: **{status}**")
            if status != "OK": is_issue = True
            
            summary = get_row_summary_data(row, ch, df.columns)
            summary.update({"Status": status, "Serial Number": sn})
            row_table_data.append(summary)
            full_report_list.append(summary)
            
        summary_text = f"📄 **SN: {sn}** ｜ " + " , ".join(ch_status_list)
        sample_info[idx] = {"summary": summary_text, "table": pd.DataFrame(row_table_data)}
        if is_issue: issue_indices.append(idx)
        else: normal_indices.append(idx)

    # 요약 메트릭 및 엑셀 다운로드
    c1, c2, c3 = st.columns([1,1,1])
    c1.metric("총 시료 수", len(test_data))
    c2.metric("결함 시료 수", len(issue_indices))
    
    if full_report_list:
        full_report_df = pd.DataFrame(full_report_list)[["Serial Number", "Channel", "200Hz", "1000Hz", "4000Hz", "THD (%)", "Status"]]
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            full_report_df.to_excel(writer, index=False, sheet_name='Report')
        c3.download_button(label="📥 분석 결과 엑셀 내보내기", data=output.getvalue(), 
                           file_name=f"Analysis_{model_type}.xlsx", mime="application/vnd.ms-excel")

    # 사이드바 결함 시료 선택
    selected_indices = []
    if issue_indices:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📍 결함 시료 선택")
        for i in issue_indices:
            sn = str(test_data.loc[i, sn_col]).strip()
            if st.sidebar.checkbox(f"SN: {sn}", key=f"check_{i}"):
                selected_indices.append(i)

    # 상세 정보 출력
    if selected_indices:
        st.info("🔍 **선택 시료 상세 분석**")
        for idx in selected_indices:
            st.markdown(sample_info[idx]["summary"])
            st.table(sample_info[idx]["table"][["Channel", "200Hz", "1000Hz", "4000Hz", "THD (%)", "Status"]].set_index("Channel"))

    # 5. 시각화 (그래프)
    def plot_mic_fr(ax, ch_info):
        cols = df.columns[ch_info["range"]]
        freqs = get_freq_values(cols)
        ylim, color, unit = ((-30, 0), 'green', 'dbV') if ch_info["type"] == 'analog' else ((-45, -25), 'blue', 'dbFS')
        
        if show_normal:
            for i in normal_indices:
                ax.plot(freqs, pd.to_numeric(test_data.loc[i, cols], errors='coerce'), color=color, alpha=0.7, linewidth=1.5)
        for i in selected_indices:
            ax.plot(freqs, pd.to_numeric(test_data.loc[i, cols], errors='coerce'), color='red', alpha=1.0, linewidth=2.5)
            
        ax.plot(freqs, pd.to_numeric(limit_low[cols], errors='coerce'), 'k--', linewidth=1.2)
        ax.plot(freqs, pd.to_numeric(limit_high[cols], errors='coerce'), 'k--', linewidth=1.2)
        ax.set_xscale('log')
        ax.set_ylim(ylim)
        ax.set_xticks([100, 200, 1000, 4000, 10000])
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax.set_title(ch_info["name"], fontsize=14, fontweight='bold')
        ax.grid(True, which="both", ls="-", alpha=0.3)

    st.subheader(f"📊 {model_type} 그래프 분석")
    fig, axes = plt.subplots(len(config["channels"]), 1, figsize=(12, 6 * len(config["channels"])))
    if len(config["channels"]) == 1: axes = [axes]
    for i, ch in enumerate(config["channels"]): plot_mic_fr(axes[i], ch)
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("사이드바에서 제품 모델을 확인하고 CSV 파일을 업로드해 주세요.")