import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import io
import chardet
import os

# 1. 페이지 설정 및 시각화 상수
st.set_page_config(page_title="MIC Analysis Tool", page_icon="🎙️", layout="wide")
FIG_WIDTH, PLOT_HEIGHT = 14, 6
FONT_SIZE_TITLE, FONT_SIZE_AXIS = 16, 12

# --- [미세조정 설정값] ---
# 이미지의 시작 위치를 셀 모서리에서 얼마나 띄울지 결정합니다 (단위: 픽셀)
X_OFFSET = 10  # 오른쪽으로 이동
Y_OFFSET = 10  # 아래로 이동
# -----------------------

# --- [상단 헤더] ---
col_head1, col_head2 = st.columns([4, 1], vertical_alignment="center")
with col_head1:
    st.markdown("<h1>🎙️ MIC Analysis Tool <span style='font-size: 16px; color: gray; font-weight: normal; margin-left: 10px;'>( 제작 : JW Lee, 자문 : JJ Kim )</span></h1>", unsafe_allow_html=True)
with col_head2:
    if os.path.exists("logo.png"): st.image("logo.png", width=300)
st.markdown("---")

# 2. 제품군 설정
PRODUCT_CONFIGS = {
    "RH": {"pn": ["96575N1100", "96575GJ100"], "channels": [{"name": "Digital Ch1", "type": "digital", "range": range(51, 101), "thd_idx": 15}, {"name": "Digital Ch2", "type": "digital", "range": range(103, 153), "thd_idx": 18}, {"name": "Digital Ch3", "type": "digital", "range": range(155, 205), "thd_idx": 21}]},
    "3903(LH Ecall)": {"pn": ["96575N1050", "96575GJ000"], "channels": [{"name": "Ecall Mic (Analog)", "type": "analog", "range": range(6, 47), "thd_idx": 69}, {"name": "Digital Ch1", "type": "digital", "range": range(107, 157), "thd_idx": 217}, {"name": "Digital Ch2", "type": "digital", "range": range(159, 209), "thd_idx": 220}]},
    "3203(LH non Ecall)": {"pn": ["96575N1000", "96575GJ010"], "channels": [{"name": "Digital Ch1", "type": "digital", "range": range(6, 56), "thd_idx": 116}, {"name": "Digital Ch2", "type": "digital", "range": range(58, 108), "thd_idx": 119}]},
    "LITE(LH)": {"pn": ["96575NR000", "96575GJ200"], "channels": [{"name": "Analog Mic", "type": "analog", "range": range(6, 47), "thd_idx": 95}]},
    "LITE(RH)": {"pn": ["96575NR100", "96575GJ300"], "channels": [{"name": "Analog Mic", "type": "analog", "range": range(6, 47), "thd_idx": 95}]}
}

# 3. [유틸리티 함수]
def clean_sn(val):
    if pd.isna(val): return ""
    return str(val).replace('"', '').replace("'", "").replace('\t', '').strip()

def get_freq_values(cols):
    return [float(str(c).split('.')[0]) for c in cols]

def detect_info(df):
    model, prod_date, matched_pn = None, "Unknown", "Unknown"
    try:
        for i in range(2, min(15, len(df))):
            sn_raw = clean_sn(df.iloc[i, 3])
            if '/' in sn_raw:
                parts = sn_raw.split('/'); pn_part = parts[0]
                for m, info in PRODUCT_CONFIGS.items():
                    if pn_part in info["pn"]: model, matched_pn = m, pn_part; break
                if len(parts[1]) >= 8: prod_date = f"20{parts[1][2:4]}/{parts[1][4:6]}/{parts[1][6:8]}"
                if model: break
    except: pass
    return model, prod_date, matched_pn

def classify_sample(row, cols, freqs, limit_low, limit_high):
    val = pd.to_numeric(row[cols], errors='coerce')
    l_low = pd.to_numeric(limit_low[cols], errors='coerce')
    l_high = pd.to_numeric(limit_high[cols], errors='coerce')
    is_fail = val.isna() | (val < l_low) | (val > l_high)
    if not is_fail.any(): return "Normal"
    check_pts = [200, 1000, 4000]
    pt_idx = [np.argmin(np.abs(np.array(freqs) - p)) for p in check_pts]
    other_idx = [i for i in range(len(cols)) if i not in pt_idx]
    if not is_fail.iloc[other_idx].any(): return "Margin Out"
    return "Defect"

def get_row_summary_data(row, ch_info, all_cols):
    cols = all_cols[ch_info["range"]]; freqs = get_freq_values(cols)
    data = {"Channel": ch_info["name"]}
    for t in [200, 1000, 4000]:
        try:
            idx = np.argmin(np.abs(np.array(freqs) - t))
            val = pd.to_numeric(row[cols[idx]], errors='coerce')
            data[f"{t}Hz"] = f"{val:.3f}" if not pd.isna(val) else "-"
        except: data[f"{t}Hz"] = "-"
    thd_idx = ch_info.get("thd_idx")
    if thd_idx is not None:
        thd_val = pd.to_numeric(row[all_cols[thd_idx]], errors='coerce')
        data["THD (1kHz, %)"] = f"{thd_val:.3f}" if not pd.isna(thd_val) else "-"
    else: data["THD (1kHz, %)"] = "N/A"
    return data

def create_fr_plot(config, df, current_test_data, limit_low, limit_high, show_normal, plotting_normal_indices, highlight_indices, for_excel=False):
    num_draw = 3 if for_excel else len(config["channels"])
    fig, axes = plt.subplots(num_draw, 1, figsize=(FIG_WIDTH, PLOT_HEIGHT * num_draw))
    if num_draw == 1: axes = [axes]
    if for_excel: fig.patch.set_linewidth(0)

    for i in range(num_draw):
        ax = axes[i]
        if i < len(config["channels"]):
            ch = config["channels"][i]; cols = df.columns[ch["range"]]; freqs = get_freq_values(cols)
            ylim, color, unit = ((-30, 0), 'green', 'dbV') if ch["type"] == 'analog' else ((-45, -25), 'blue', 'dbFS')
            ax.set_xscale('log'); ax.set_xticks([50, 100, 200, 1000, 4000, 10000, 14000])
            ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter()); ax.minorticks_off()
            if show_normal:
                for n in plotting_normal_indices: ax.plot(freqs, pd.to_numeric(current_test_data.loc[n, cols], errors='coerce'), color=color, alpha=0.7, lw=1.2)
            for h in highlight_indices: ax.plot(freqs, pd.to_numeric(current_test_data.loc[h, cols], errors='coerce'), color='red', lw=2.5)
            ax.plot(freqs, pd.to_numeric(limit_low[cols], errors='coerce'), 'k--', lw=1.2); ax.plot(freqs, pd.to_numeric(limit_high[cols], errors='coerce'), 'k--', lw=1.2)
            ax.set_title(ch["name"], fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
            ax.set_ylabel(f'Response ({unit})', fontsize=FONT_SIZE_AXIS); ax.set_ylim(ylim); ax.grid(True, alpha=0.4)
        else: ax.axis('off')
    plt.tight_layout(); return fig

def plot_bell_curve_set(config, df, test_data, stats_indices, sel_idx, for_excel=False):
    num_draw = 3 if for_excel else len(config["channels"])
    fig, axes = plt.subplots(num_draw, 1, figsize=(FIG_WIDTH, PLOT_HEIGHT * num_draw))
    if num_draw == 1: axes = [axes]
    if for_excel: fig.patch.set_linewidth(0)
    for i in range(num_draw):
        ax = axes[i]
        if i < len(config["channels"]):
            ch = config["channels"][i]; col_idx = np.argmin(np.abs(np.array(get_freq_values(df.columns[ch["range"]])) - 1000))
            v_clean = pd.to_numeric(test_data[df.columns[ch["range"]][col_idx]].iloc[stats_indices], errors='coerce').dropna()
            lcl, ucl = (-11, -9) if ch["type"] == 'analog' else (-38, -36)
            if len(v_clean) >= 2:
                mu, std = v_clean.mean(), v_clean.std(); cpk = min((ucl-mu)/(3*std), (mu-lcl)/(3*std)) if std > 0 else 0
                x_r = np.linspace(lcl - 2, ucl + 2, 200); p = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_r - mu) / std)**2)
                ax.plot(x_r, p, 'k', lw=2.5, alpha=0.7); ax.fill_between(x_r, p, color='gray', alpha=0.1)
                ax.axvline(lcl, color='blue', ls='--', lw=1.5); ax.axvline(ucl, color='red', ls='--', lw=1.5)
                # [수정] 텍스트 박스: Cpk만 표시
                ax.text(0.95, 0.75, f"Cpk: {cpk:.2f}", transform=ax.transAxes, ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=FONT_SIZE_AXIS, fontweight='bold')
                ax.set_title(f"{ch['name']} - Distribution", fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15); ax.set_xlim(lcl-2, ucl+2); ax.grid(True, alpha=0.2)
        else: ax.axis('off')
    plt.tight_layout(); return fig

# 4. [메인 로직]
uploaded_file = st.sidebar.file_uploader("CSV 로그 파일을 업로드하세요.", type=['csv'])

if uploaded_file:
    raw_bytes = uploaded_file.read()
    det_enc = chardet.detect(raw_bytes)['encoding']
    df = pd.read_csv(io.StringIO(raw_bytes.decode(det_enc if det_enc else 'utf-8-sig', errors='replace')), low_memory=False)
    df.columns = [str(c) for c in df.columns]

    if df is not None:
        detected_model, prod_date, detected_pn = detect_info(df)
        model_list = list(PRODUCT_CONFIGS.keys())
        model_type = st.sidebar.selectbox("제품 모델 선택", options=model_list, index=model_list.index(detected_model) if detected_model in model_list else 0)
        config = PRODUCT_CONFIGS[model_type]; st.sidebar.markdown("---"); show_normal = st.sidebar.checkbox("정상 시료 FR 표시", value=True)

        sn_col, limit_low, limit_high = df.columns[3], df.iloc[0], df.iloc[1]
        raw_test_data = df.iloc[2:].dropna(subset=[sn_col])
        test_data = raw_test_data[raw_test_data[sn_col].astype(str).str.contains('/', na=False)].reset_index(drop=True)

        if len(test_data) > 0:
            sample_info, issue_indices, stats_indices, plotting_normal_indices = {}, [], [], []
            ch_stats_data = {ch["name"]: {"pass": 0, "fail": 0, "vals_1k": []} for ch in config["channels"]}

            for idx, row in test_data.iterrows():
                is_any_defect, is_any_fail, is_pure_normal = False, False, True
                row_f_table = []
                for ch in config["channels"]:
                    cols = df.columns[ch["range"]]; freqs = get_freq_values(cols)
                    status = classify_sample(row, cols, freqs, limit_low, limit_high)
                    if status == "Defect": is_any_defect = True
                    if status != "Normal": is_any_fail, is_pure_normal = True, False
                    val_1k = pd.to_numeric(row[cols[np.argmin(np.abs(np.array(freqs) - 1000))]], errors='coerce')
                    ch_stats_data[ch["name"]]["vals_1k"].append(val_1k)
                    if status == "Normal": ch_stats_data[ch["name"]]["pass"] += 1
                    else: ch_stats_data[ch["name"]]["fail"] += 1
                    res = get_row_summary_data(row, ch, df.columns); res["Status"] = status; row_f_table.append(res)
                sample_info[idx] = {"table": pd.DataFrame(row_f_table), "sn": clean_sn(row[sn_col])}
                if is_any_fail: issue_indices.append(idx)
                if not is_any_defect: stats_indices.append(idx)
                if is_pure_normal: plotting_normal_indices.append(idx)

            # --- [수정: sel_idx 정의를 탭 생성 위로 이동] ---
            st.sidebar.markdown("---"); st.sidebar.header("❌️ 결함 시료 선택")
            sel_idx = [i for i in issue_indices if st.sidebar.checkbox(f"SN: {sample_info[i]['sn']}", key=f"ch_{i}")]

            # --- [UI 출력] ---
            st.subheader("📝 Production Dashboard")
            d1, d2, d3 = st.columns([1.2, 1.3, 2.5])
            total_qty, total_fail = len(test_data), len(issue_indices); total_pass = total_qty - total_fail; yield_val = total_pass / total_qty * 100
            with d1: st.markdown(f"**Model P/N:** `{detected_pn}`\n\n**Prod. Date:** `{prod_date}`\n\n**Quantity:** `{len(test_data)} EA`")
            with d2:
                st.markdown(f"""<div style="display: flex; gap: 8px;"><div style="background-color: #f8f9fa; padding: 10px 15px; border-radius: 10px; border-left: 5px solid #28a745; flex: 1;"><p style="margin:0; font-size:11px; color:#6c757d; font-weight:bold;">PASS</p><p style="margin:0; font-size:24px; font-weight:800; color:#28a745;">{total_pass}</p></div><div style="background-color: #f8f9fa; padding: 10px 15px; border-radius: 10px; border-left: 5px solid #dc3545; flex: 1;"><p style="margin:0; font-size:11px; color:#6c757d; font-weight:bold;">FAIL</p><p style="margin:0; font-size:24px; font-weight:800; color:#dc3545;">{total_fail}</p></div></div>""", unsafe_allow_html=True)
                st.markdown(f"<p style='margin-top:10px; font-weight:bold; font-size:20px;'>Overall Yield: {yield_val:.1f}%</p>", unsafe_allow_html=True)
            with d3:
                summary_rows = []
                for ch_name, stat in ch_stats_data.items():
                    v = np.array(stat["vals_1k"])[stats_indices]; v = v[~np.isnan(v)]
                    v_min, v_max, v_avg, v_std = (v.min(), v.max(), v.mean(), v.std()) if len(v) > 0 else (0,0,0,0)
                    summary_rows.append({"Channel": ch_name, "Pass": stat["pass"], "Fail": stat["fail"], "Yield": f"{(stat['pass']/total_qty)*100:.1f}%", "Min": f"{v_min:.2f}", "Max": f"{v_max:.2f}", "Avg": f"{v_avg:.2f}", "Stdev": f"{v_std:.2f}"})
                ch_sum_df = pd.DataFrame(summary_rows); st.dataframe(ch_sum_df, hide_index=True, width=1200)
            
            st.markdown("---"); tab_fr, tab_dist, tab_detail = st.tabs(["📈 주파수 응답 (FR)", "📉 정규분포 (Cpk)", "🔍 결함 시료 상세"])

            with tab_fr: st.pyplot(create_fr_plot(config, df, test_data, limit_low, limit_high, show_normal, plotting_normal_indices, sel_idx))
            with tab_dist: st.pyplot(plot_bell_curve_set(config, df, test_data, stats_indices, sel_idx))
            with tab_detail:
                if sel_idx:
                    for i in sel_idx: st.write(f"📄 **SN: {sample_info[i]['sn']}**"); st.table(sample_info[i]["table"].set_index("Channel"))
                else: st.warning("결함 시료를 선택하세요.")

            # --- [엑셀 리포트 생성 섹션: 기둥 잔선 제거] ---
            def generate_excel():
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    workbook = writer.book
                    
                    # 1. 서식 라이브러리 (시스템 속성 충돌 방지)
                    base_blue = {'bold': True, 'bg_color': '#DEEAF6', 'align': 'center', 'valign': 'vcenter', 'border': 1}
                    base_green = {'bold': True, 'bg_color': '#E2EFDA', 'align': 'center', 'valign': 'vcenter', 'border': 1}
                    base_yld = {'bold': True, 'font_size': 18, 'font_color': '#2E7D32', 'align': 'center', 'valign': 'vcenter', 'border': 1}
                    base_thin = {'align': 'center', 'valign': 'vcenter', 'border': 1}

                    def get_fmt(base_props, top=1, bottom=1, left=1, right=1):
                        props = base_props.copy()
                        props.update({'top': top, 'bottom': bottom, 'left': left, 'right': right})
                        return workbook.add_format(props)

                    def write_dashboard(ws, last_row_idx=37):
                        # 컬럼 너비 설정
                        ws.set_column('A:A', 3)
                        ws.set_column('B:B', 15); ws.set_column('C:C', 22); ws.set_column('D:F', 10); ws.set_column('G:N', 11)
                        
                        # [A] 상단 헤더 (B2:N2)
                        ws.merge_range('B2:F2', '📝 PRODUCTION SUMMARY', get_fmt(base_blue, top=2, left=2))
                        ws.merge_range('G2:N2', '📈 CHANNEL STATISTICS', get_fmt(base_green, top=2, right=2))

                        # [B] Summary Table (B3:C6)
                        labels = ["Model Type", "Model P/N", "Prod. Date", "Quantity"]
                        vals = [model_type, detected_pn, prod_date, str(total_qty) + " EA"]
                        for i in range(4):
                            r = 2 + i
                            ws.write(r, 1, labels[i], get_fmt(base_blue, left=2, bottom=2 if r==5 else 1))
                            ws.write(r, 2, vals[i], get_fmt(base_thin, bottom=2 if r==5 else 1))
                        
                        # [C] PASS/FAIL/YIELD (D3:F6)
                        ws.write(2, 3, 'PASS', workbook.add_format(base_blue))
                        ws.merge_range('E3:F3', total_pass, workbook.add_format(base_thin))
                        ws.write(3, 3, 'FAIL', workbook.add_format(base_blue))
                        ws.merge_range('E4:F4', total_fail, workbook.add_format(base_thin))
                        ws.merge_range('D5:F6', "Yield: " + str(round(yield_val, 1)) + "%", get_fmt(base_yld, bottom=2))

                        # [D] CHANNEL STATISTICS 데이터 (G3:N6)
                        heads = ch_sum_df.columns.tolist()
                        for i, h in enumerate(heads):
                            col = 6 + i
                            ws.write(2, col, h, get_fmt(base_green, right=2 if col==13 else 1))
                        
                        # 통계 행 채우기 (데이터 부족 시에도 N열 우측선 유지)
                        for r_idx in range(4):
                            r = 3 + r_idx
                            is_last = (r == 5)
                            if r_idx < len(ch_sum_df):
                                r_v = ch_sum_df.iloc[r_idx]
                                ws.write(r, 6, r_v[heads[0]], get_fmt(base_thin, bottom=2 if is_last else 1))
                                for c_idx in range(1, 8):
                                    col = 6 + c_idx
                                    ws.write(r, col, r_v[heads[c_idx]], get_fmt(base_thin, right=2 if col==13 else 1, bottom=2 if is_last else 1))
                            else:
                                # 빈 데이터 행: 내부 격자 없이 우측 굵은 테두리만 생성
                                for c_idx in range(8):
                                    col = 6 + c_idx
                                    ws.write_blank(r, col, "", workbook.add_format({'right': 2 if col==13 else 0, 'bottom': 2 if is_last else 0}))

                        # [E] 차트 영역 외곽 기둥 (B7:N36) - 안쪽 잔선 차단
                        for r_frame in range(6, last_row_idx - 1):
                            # B열: 왼쪽만 굵게, 오른쪽은 0
                            ws.write_blank(r_frame, 1, "", workbook.add_format({'left': 2, 'right': 0, 'top': 0, 'bottom': 0}))
                            # N열: 오른쪽만 굵게, 왼쪽은 0 (이것이 G7~M열 사이 잔선을 없앱니다)
                            ws.write_blank(r_frame, 13, "", workbook.add_format({'right': 2, 'left': 0, 'top': 0, 'bottom': 0}))
                        
                        # [F] 최하단 가로선 (B37:N37)
                        ws.write_blank(last_row_idx - 1, 1, "", workbook.add_format({'left': 2, 'bottom': 2, 'right': 0}))
                        for c_bot in range(2, 13):
                            ws.write_blank(last_row_idx - 1, c_bot, "", workbook.add_format({'bottom': 2, 'top': 0, 'left': 0, 'right': 0}))
                        ws.write_blank(last_row_idx - 1, 13, "", workbook.add_format({'right': 2, 'bottom': 2, 'left': 0}))

                    # --- 시트 생성 및 이미지 삽입 ---
                    ws1 = workbook.add_worksheet('📈 분석 리포트'); write_dashboard(ws1, 37)
                    
                    fig_fr = create_fr_plot(config, df, test_data, limit_low, limit_high, show_normal, plotting_normal_indices, sel_idx, for_excel=True)
                    buf_f = io.BytesIO(); fig_fr.savefig(buf_f, format='png', dpi=100); plt.close(fig_fr)
                    ws1.insert_image('B7', 'fr.png', {'image_data': buf_f, 'x_scale': 0.41, 'y_scale': 0.35, 'x_offset': 10, 'y_offset': 10})
                    
                    fig_dist = plot_bell_curve_set(config, df, test_data, stats_indices, sel_idx, for_excel=True)
                    buf_d = io.BytesIO(); fig_dist.savefig(buf_d, format='png', dpi=100); plt.close(fig_dist)
                    ws1.insert_image('H7', 'dist.png', {'image_data': buf_d, 'x_scale': 0.41, 'y_scale': 0.35, 'x_offset': 10, 'y_offset': 10})

                    # 결함 상세 시트
                    ws2 = workbook.add_worksheet('🔍 결함상세'); write_dashboard(ws2, 37)
                    ws2.merge_range('B8:N8', '🔍 DETAILED FAILURE LOG', get_fmt(base_blue, top=1, left=1, right=1, bottom=1))
                    # ... (결함 데이터 기입 로직)

                return output.getvalue()

            st.sidebar.markdown("---")
            if st.sidebar.button("📥 엑셀 리포트 생성"):
                st.sidebar.download_button(label="💾 파일 다운로드", data=generate_excel(), file_name=f"Report_{detected_pn}.xlsx", mime="application/vnd.ms-excel")
            else: st.info("사이드바에서 CSV 로그 파일을 업로드하세요.")