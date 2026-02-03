import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import io
import chardet
import os

# 페이지 설정
st.set_page_config(page_title="MIC LOG Analyzer", page_icon="🎙️", layout="wide")

# --- [상단 헤더 및 로고 배치] ---
header_col1, header_col2 = st.columns([5, 1])

with header_col1:
    # 제목 뒤에 회색 작은 글씨로 관계자 명기
    st.markdown(
        """
        <h1 style='display: inline;'>🎙️ MIC LOG 분석 자동화 툴 
            <span style='font-size: 16px; color: gray; font-weight: normal; margin-left: 10px;'>
                ( 제작 : JW Lee, 자문 : JJ Kim )
            </span>
        </h1>
        """, 
        unsafe_allow_html=True
    )

with header_col2:
    if os.path.exists("logo.jpg"):
        st.image("logo.jpg", width=450)

st.markdown("---")

# 1. 제품군 설정 정보
PRODUCT_CONFIGS = {
    "3903": {"channels": [{"name": "Ecall Mic (Analog)", "type": "analog", "range": range(6, 47), "thd_idx": 69}, {"name": "Digital Ch1", "type": "digital", "range": range(107, 157), "thd_idx": 217}, {"name": "Digital Ch2", "type": "digital", "range": range(159, 209), "thd_idx": 220}]},
    "3203": {"channels": [{"name": "Digital Ch1", "type": "digital", "range": range(6, 56), "thd_idx": 116}, {"name": "Digital Ch2", "type": "digital", "range": range(58, 108), "thd_idx": 119}]},
    "RH": {"channels": [{"name": "Digital Ch1", "type": "digital", "range": range(51, 101), "thd_idx": 15}, {"name": "Digital Ch2", "type": "digital", "range": range(103, 153), "thd_idx": 18}, {"name": "Digital Ch3", "type": "digital", "range": range(155, 205), "thd_idx": 21}]},
    "RH_본사": {"channels": [{"name": "Digital Ch1", "type": "digital", "range": range(51, 92), "thd_idx": 15}, {"name": "Digital Ch2", "type": "digital", "range": range(94, 135), "thd_idx": 18}, {"name": "Digital Ch3", "type": "digital", "range": range(137, 178), "thd_idx": 21}]},
    "LITE": {"channels": [{"name": "Analog Mic", "type": "analog", "range": range(6, 47), "thd_idx": 95}]}
}

# 2. 사이드바 설정
st.sidebar.header("🛠️ 모델 및 데이터 설정")
model_type = st.sidebar.selectbox("제품 모델을 선택하세요.", options=list(PRODUCT_CONFIGS.keys()))
uploaded_file = st.sidebar.file_uploader(f"[{model_type}] CSV 파일을 업로드하세요.", type=['csv'])

st.sidebar.markdown("---")
st.sidebar.header("🔍 시각화 옵션")
show_fr_plot = st.sidebar.checkbox("주파수 응답(FR) 그래프 표시", value=True)
show_detail_table = st.sidebar.checkbox("선택 시료 상세 테이블 표시", value=True)
show_dist_plot = st.sidebar.checkbox("정규분포 그래프 표시", value=False)

# [수정] 사이드바에 별도의 '정상 시료' 섹션 생성
st.sidebar.markdown("---")
st.sidebar.header("✅ 정상 시료 설정")
show_normal = st.sidebar.checkbox("정상 시료 FR 표시", value=True)

# 3. 유틸리티 및 그래프 함수 (기존 로직 유지)
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

# 4. 분석 프로세스
if uploaded_file:
    raw = uploaded_file.read()
    det = chardet.detect(raw)
    df = pd.read_csv(io.StringIO(raw.decode(det['encoding'] if det['encoding'] else 'utf-8', errors='replace')), low_memory=False)
    config = PRODUCT_CONFIGS[model_type]
    sn_col, limit_low, limit_high = 'Unnamed: 3', df.iloc[0], df.iloc[1]
    test_data = df.iloc[2:].dropna(subset=[sn_col]).reset_index(drop=True)
    
    sample_info, issue_indices, normal_indices = {}, [], []
    for idx, row in test_data.iterrows():
        is_issue, row_table = False, []
        for ch in config["channels"]:
            status = get_channel_status(row, df.columns[ch["range"]], ch["type"], limit_low, limit_high)
            if status != "OK": is_issue = True
            summary = get_row_summary_data(row, ch, df.columns); summary["Status"] = status
            row_table.append(summary)
        sn = str(row[sn_col]).strip()
        sample_info[idx] = {"table": pd.DataFrame(row_table), "sn": sn, "row_raw": row}
        if is_issue: issue_indices.append(idx)
        else: normal_indices.append(idx)

    # 상단 요약 및 엑셀 내보내기
    c1, c2, c3 = st.columns([1,1,1])
    c1.metric("총 시료 수", len(test_data)); c2.metric("결함 시료 수", len(issue_indices))
    
    # 사이드바 결함 시료 선택 섹션
    st.sidebar.markdown("---")
    st.sidebar.header("📍 결함 시료 선택")
    selected_indices = [i for i in issue_indices if st.sidebar.checkbox(f"SN: {sample_info[i]['sn']}", key=f"check_{i}")]

    if selected_indices:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            ws = writer.book.add_worksheet('Report')
            curr_row = 0
            for s in selected_indices:
                info = sample_info[s]
                ws.write(curr_row, 0, f"SN: {info['sn']}"); curr_row += 1
                table_df = info["table"][["Channel", "200Hz", "1000Hz", "4000Hz", "THD (1kHz, %)", "Status"]]
                table_df.to_excel(writer, sheet_name='Report', startrow=curr_row, index=False); curr_row += len(table_df) + 2
                fig_ex = create_fr_plot(config, df, test_data, limit_low, limit_high, False, [], [s])
                img = io.BytesIO(); fig_ex.savefig(img, format='png', dpi=90); plt.close(fig_ex)
                ws.insert_image(curr_row, 0, f'p_{s}.png', {'image_data': img, 'x_scale': 0.8, 'y_scale': 0.8}); curr_row += (18 * len(config["channels"])) + 5
        c3.download_button("📥 선택 시료 결과 엑셀 내보내기", output.getvalue(), f"Report_{model_type}.xlsx")

    # 메인 화면 출력
    if selected_indices and show_detail_table:
        st.info("🔍 **선택 시료 상세 분석 테이블**")
        for idx in selected_indices:
            st.write(f"📄 **SN: {sample_info[idx]['sn']}**")
            st.table(sample_info[idx]["table"][["Channel", "200Hz", "1000Hz", "4000Hz", "THD (1kHz, %)", "Status"]].set_index("Channel"))
        
    if show_dist_plot:
        st.info("📉 **1kHz Sensitivity 정규분포 분석**")
        fig_d, axes_d = plt.subplots(len(config["channels"]), 1, figsize=(8, 4 * len(config["channels"])))
        if len(config["channels"]) == 1: axes_d = [axes_d]
        for i, ch in enumerate(config["channels"]):
            cols = df.columns[ch["range"]]
            idx_1k = np.argmin(np.abs(np.array(get_freq_values(cols)) - 1000))
            plot_bell_curve(axes_d[i], test_data[cols[idx_1k]], normal_indices, selected_indices, f"{ch['name']} - Distribution", ch["type"])
        st.pyplot(fig_d)

    if show_fr_plot:
        st.subheader(f"📊 {model_type} 주파수 응답(FR) 분석")
        st.pyplot(create_fr_plot(config, df, test_data, limit_low, limit_high, show_normal, normal_indices, selected_indices))
else:
    st.info("사이드바에서 제품 모델을 확인하고 CSV 파일을 업로드해 주세요.")