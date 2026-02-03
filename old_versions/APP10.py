import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import io
import chardet

# 페이지 설정
st.set_page_config(page_title="MIC LOG Analyzer", layout="wide")

st.title("🎙️ MIC LOG 분석 자동화 툴")
st.markdown("---")

# 1. 제품군 설정 정보
PRODUCT_CONFIGS = {
    "3903": {
        "channels": [
            {"name": "Ecall Mic (Analog)", "type": "analog", "range": range(6, 47), "thd_idx": 69},
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
    },
    "RH_본사": {
        "channels": [
            {"name": "Digital Ch1", "type": "digital", "range": range(51, 92), "thd_idx": 15}, 
            {"name": "Digital Ch2", "type": "digital", "range": range(94, 135), "thd_idx": 18}, 
            {"name": "Digital Ch3", "type": "digital", "range": range(137, 178), "thd_idx": 21}, 
        ]
    },
    "LITE": { 
        "channels": [
            {"name": "Analog Mic", "type": "analog", "range": range(6, 47), "thd_idx": 95}, 
        ]
    }
}

# 2. 사이드바 설정
st.sidebar.header("🛠️ 모델 및 데이터 설정")
model_type = st.sidebar.selectbox("제품 모델을 선택하세요.", options=["3903", "3203", "RH", "RH_본사", "LITE"])
uploaded_file = st.sidebar.file_uploader(f"[{model_type}] CSV 파일을 업로드하세요.", type=['csv'])

st.sidebar.markdown("---")
st.sidebar.header("🔍 시각화 옵션")
show_normal = st.sidebar.checkbox("정상 시료 FR 표시", value=True)
show_detail_table = st.sidebar.checkbox("선택 시료 상세 테이블 표시", value=True)
show_dist_plot = st.sidebar.checkbox("수치별 분포(종 모양) 그래프 표시", value=False)

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
        except: data[f"{t}Hz"] = "-"
    
    th_key = "THD (1kHz, %)"
    if ch_info["thd_idx"] is not None:
        try:
            thd_val = float(row[all_cols[ch_info["thd_idx"]]])
            data[th_key] = f"{thd_val:.3f}"
        except: data[th_key] = "-"
    else: data[th_key] = "N/A"
    return data

# [수정] 분포 그래프: THD 제거, LCL/UCL 추가, 수치 표기 제거
def plot_bell_curve(ax, data_series, selected_values, title):
    clean_data = data_series.dropna()
    if len(clean_data) < 2: return
    
    mu, std = clean_data.mean(), clean_data.std()
    x = np.linspace(mu - 4*std, mu + 4*std, 100)
    p = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std)**2)
    
    ax.plot(x, p, 'k', linewidth=2, alpha=0.6)
    ax.fill_between(x, p, color='gray', alpha=0.1)
    
    # 관리 한계선 (LCL: -38, UCL: -36) 추가
    ax.axvline(-38, color='blue', linestyle='--', linewidth=1.2, label='LCL (-38)')
    ax.axvline(-36, color='red', linestyle='--', linewidth=1.2, label='UCL (-36)')
    
    # 선택 시료 위치 표시 (수치 어노테이션 제거)
    for val in selected_values:
        y_pos = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((val - mu) / std)**2)
        ax.scatter(val, y_pos, color='red', s=80, edgecolors='white', zorder=5)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("Sensitivity (dBV/dbFS)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', ls=':', alpha=0.5)

# [수정] FR 그래프 그리기 함수 (기존 색상 및 스타일 복구)
def create_fr_plot(config, df, current_test_data, limit_low, limit_high, show_normal, normal_indices, highlight_indices):
    num_channels = len(config["channels"])
    fig, axes = plt.subplots(num_channels, 1, figsize=(10, 5 * num_channels))
    if num_channels == 1: axes = [axes]
    
    for i, ch in enumerate(config["channels"]):
        ax = axes[i]
        cols = df.columns[ch["range"]]
        freqs = get_freq_values(cols)
        # 기존 색상 설정 복구: Analog=Green, Digital=Blue
        ylim, color, unit = ((-30, 0), 'green', 'dbV') if ch["type"] == 'analog' else ((-45, -25), 'blue', 'dbFS')
        
        if show_normal:
            for n_idx in normal_indices:
                ax.plot(freqs, pd.to_numeric(current_test_data.loc[n_idx, cols], errors='coerce'), color=color, alpha=0.7, linewidth=1.5)
        
        for h_idx in highlight_indices:
            ax.plot(freqs, pd.to_numeric(current_test_data.loc[h_idx, cols], errors='coerce'), color='red', alpha=1.0, linewidth=2.5)
            
        ax.plot(freqs, pd.to_numeric(limit_low[cols], errors='coerce'), 'k--', linewidth=1.2)
        ax.plot(freqs, pd.to_numeric(limit_high[cols], errors='coerce'), 'k--', linewidth=1.2)
        ax.set_xscale('log')
        ax.set_ylim(ylim)
        ax.set_xticks([100, 200, 1000, 4000, 10000])
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax.set_title(ch["name"], fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Response ({unit})')
        ax.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    return fig

# 4. 분석 프로세스
if uploaded_file is not None:
    raw_data = uploaded_file.read()
    detection = chardet.detect(raw_data)
    decoded_content = raw_data.decode(detection['encoding'] if detection['encoding'] else 'utf-8', errors='replace')
    df = pd.read_csv(io.StringIO(decoded_content), low_memory=False)
    
    config = PRODUCT_CONFIGS[model_type]
    sn_col = 'Unnamed: 3'
    limit_low, limit_high = df.iloc[0], df.iloc[1]
    test_data = df.iloc[2:].copy()
    test_data = test_data.dropna(subset=[sn_col]).reset_index(drop=True)
    
    sample_info, issue_indices, normal_indices = {}, [], []
    for idx, row in test_data.iterrows():
        ch_status_list, is_issue = [], False
        sn = str(row[sn_col]).strip()
        row_table_data = []
        for ch in config["channels"]:
            status = get_channel_status(row, df.columns[ch["range"]], ch["type"], limit_low, limit_high)
            ch_status_list.append(f"{ch['name']}: **{status}**")
            if status != "OK": is_issue = True
            summary = get_row_summary_data(row, ch, df.columns)
            summary["Status"] = status
            row_table_data.append(summary)
        summary_text = f"📄 **SN: {sn}** ｜ " + " , ".join(ch_status_list)
        sample_info[idx] = {"summary": summary_text, "table": pd.DataFrame(row_table_data), "sn": sn, "row_raw": row}
        if is_issue: issue_indices.append(idx)
        else: normal_indices.append(idx)

    # 상단 요약
    c1, c2, c3 = st.columns([1,1,1])
    c1.metric("총 시료 수", len(test_data))
    c2.metric("결함 시료 수", len(issue_indices))

    # 사이드바 선택
    selected_indices = []
    if issue_indices:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📍 결함 시료 선택")
        for i in issue_indices:
            sn = str(test_data.loc[i, sn_col]).strip()
            if st.sidebar.checkbox(f"SN: {sn}", key=f"check_{i}"):
                selected_indices.append(i)

    # 엑셀 리포트 내보내기 버튼 (기존 FR 설정 유지됨)
    if selected_indices:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book
            worksheet = workbook.add_worksheet('Report')
            curr_row = 0
            for s_idx in selected_indices:
                info = sample_info[s_idx]
                worksheet.write(curr_row, 0, info["summary"].replace('**', ''))
                curr_row += 1
                table_df = info["table"][["Channel", "200Hz", "1000Hz", "4000Hz", "THD (1kHz, %)", "Status"]]
                table_df.to_excel(writer, sheet_name='Report', startrow=curr_row, index=False)
                curr_row += len(table_df) + 2
                fig_ex = create_fr_plot(config, df, test_data, limit_low, limit_high, False, [], [s_idx])
                img_stream = io.BytesIO()
                fig_ex.savefig(img_stream, format='png', dpi=90)
                plt.close(fig_ex)
                worksheet.insert_image(curr_row, 0, f'p_{s_idx}.png', {'image_data': img_stream, 'x_scale': 0.8, 'y_scale': 0.8})
                curr_row += (18 * len(config["channels"])) + 5 
        c3.download_button(label="📥 선택 시료 결과 엑셀 내보내기", data=output.getvalue(), 
                           file_name=f"Report_{model_type}.xlsx", mime="application/vnd.ms-excel")

    # [수정] 상세 분석 테이블 표시 여부
    if selected_indices and show_detail_table:
        st.info("🔍 **선택 시료 상세 분석 테이블**")
        for idx in selected_indices:
            st.markdown(sample_info[idx]["summary"])
            st.table(sample_info[idx]["table"][["Channel", "200Hz", "1000Hz", "4000Hz", "THD (1kHz, %)", "Status"]].set_index("Channel"))
        
    # [수정] 분포 그래프 표시 여부 및 내용 (Sensitivity만 표시, LCL/UCL 추가)
    if selected_indices and show_dist_plot:
        st.info("📉 **1kHz Sensitivity 분포 분석 (Bell Curve)**")
        num_ch = len(config["channels"])
        dist_fig, dist_axes = plt.subplots(num_ch, 1, figsize=(8, 4 * num_ch))
        if num_ch == 1: dist_axes = [dist_axes]
        
        for i, ch in enumerate(config["channels"]):
            cols = df.columns[ch["range"]]
            freqs = get_freq_values(cols)
            idx_1k = np.argmin(np.abs(np.array(freqs) - 1000))
            all_1k = pd.to_numeric(test_data[cols[idx_1k]], errors='coerce')
            sel_1k = [float(sample_info[s_idx]["row_raw"][cols[idx_1k]]) for s_idx in selected_indices]
            
            plot_bell_curve(dist_axes[i], all_1k, sel_1k, f"{ch['name']} - 1kHz Sensitivity Distribution")

        plt.tight_layout()
        st.pyplot(dist_fig)

    # FR 그래프 출력 (기존 색상 설정 복구됨)
    st.subheader(f"📊 {model_type} 주파수 응답(FR) 분석")
    fig_ui = create_fr_plot(config, df, test_data, limit_low, limit_high, show_normal, normal_indices, selected_indices)
    st.pyplot(fig_ui)

else:
    st.info("사이드바에서 제품 모델을 확인하고 CSV 파일을 업로드해 주세요.")