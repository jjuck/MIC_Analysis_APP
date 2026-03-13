import streamlit as st

st.set_page_config(layout="wide", page_title="Table OneLine Test")

st.title("🛠️ HTML 한 줄(Minified) 렌더링 테스트")
st.write("아래에 **표만 깔끔하게 하나** 딱 나와야 성공입니다. (코드 찌꺼기 X)")

# --- 1. 더미 데이터 ---
d_types = ["Margin Out", "Curved Out", "No Signal", "Nan"]
d_counts = {"Margin Out": 10, "Curved Out": 5, "No Signal": 2, "Nan": 1}
total_f_ch = sum(d_counts.values())

# --- 2. HTML 조립 (리스트에 담아서 한 방에 합치기) ---
# 줄바꿈(\n)이나 들여쓰기 공백을 아예 없애버리는 전략입니다.

html_parts = []

# (1) 테이블 시작 & 헤더
html_parts.append("<table style='width:100%; border-collapse:collapse; border:1px solid #bdc3c7; font-size:13px; text-align:center;'>")
html_parts.append("<thead style='background-color:#F2F2F2; font-weight:bold;'>")
html_parts.append("<tr>")
html_parts.append("<th style='border:1px solid #bdc3c7; padding:8px;'>불량 유형 (Defect Type)</th>")
html_parts.append("<th style='border:1px solid #bdc3c7; padding:8px;'>수량 (Quantity)</th>")
html_parts.append("<th style='border:1px solid #bdc3c7; padding:8px;'>비율 (Rate %)</th>")
html_parts.append("</tr>")
html_parts.append("</thead>")
html_parts.append("<tbody>")

# (2) 데이터 행 반복
for t in d_types:
    qty = d_counts[t]
    rate = (qty / total_f_ch * 100) if total_f_ch > 0 else 0
    
    html_parts.append("<tr>")
    html_parts.append(f"<td style='border:1px solid #bdc3c7; padding:5px; font-weight:bold; background-color:#F9F9F9;'>{t}</td>")
    html_parts.append(f"<td style='border:1px solid #bdc3c7; padding:5px;'>{qty} EA</td>")
    html_parts.append(f"<td style='border:1px solid #bdc3c7; padding:5px;'>{rate:.1f}%</td>")
    html_parts.append("</tr>")

# (3) 합계 행 & 테이블 닫기
html_parts.append("<tr style='background-color:#E2EFDA; font-weight:bold;'>")
html_parts.append("<td style='border:1px solid #bdc3c7; padding:5px;'>Total Failure</td>")
html_parts.append(f"<td style='border:1px solid #bdc3c7; padding:5px;'>{total_f_ch} EA</td>")
html_parts.append("<td style='border:1px solid #bdc3c7; padding:5px;'>100.0%</td>")
html_parts.append("</tr>")
html_parts.append("</tbody>")
html_parts.append("</table>")

# [핵심] 리스트를 빈 문자열로 결합 -> 공백 없는 한 줄 HTML 완성
final_html = "".join(html_parts)

# --- 3. 결과 출력 ---
st.markdown(final_html, unsafe_allow_html=True)