
---

# 🎙️ MIC Analysis Tool v1.0

## 📖 소개 (Introduction)

**MIC Analysis Tool**은 마이크(MIC) 공정의 양산 로그 데이터(CSV)를 업로드하여 주파수 응답(FR)과 THD(Total Harmonic Distortion)를 자동으로 분석하고, 불량 유형을 판정하며, 직관적인 웹 대시보드와 정밀한 엑셀 리포트를 제공하는 Streamlit 기반 데이터 분석 애플리케이션입니다.

## ✨ 주요 기능 (Key Features)

* **자동 모델 식별**: 업로드된 CSV 데이터의 Serial Number 규칙을 분석하여 제품 모델, Part Number, 생산 일자를 자동으로 감지합니다.
* **직관적인 대시보드**: 전체 검사 수량, Pass/Fail, 수율(Yield), 채널별 세부 통계(Min, Max, Avg, Stdev)를 한눈에 파악할 수 있습니다.
* **정밀한 불량 유형 분류**:
* `Nan`: 측정 데이터 누락
* `No Signal`: 신호 없음 (Digital -45dB / Analog -30dB 미만)
* `Margin Out`: 주요 검사 주파수(200Hz, 1kHz, 4kHz) 외 구간에서의 규격 이탈
* `Curved Out`: 주요 검사 주파수 구간 내 규격 이탈


* **데이터 시각화 탭 (Visualizations)**:
* 📈 **주파수 응답 (FR)**: 정상 및 결함 시료의 응답 곡선과 공정 관리 한계선(Limit)을 겹쳐서 비교
* 📉 **정규분포 (Cpk)**: 1kHz 기준 측정값의 정규분포도 및 공정 능력 지수(Cpk) 확인
* 🔍 **결함 시료 상세**: 불량 시료별 주파수/THD 상세 스펙 및 판정 결과 표시
* 📊 **불량 유형 요약**: 전체 검사 수량 대비 각 불량 유형의 발생 비율(%) 파악


* **원클릭 엑셀 리포트 (Excel Export)**:
웹 대시보드에서 분석된 모든 통계, 차트 이미지, 불량 유형 Summary, 상세 결함 로그가 서식에 맞춰 완벽하게 정렬된 엑셀 파일(`.xlsx`)을 다운로드합니다.

## 🛠️ 설치 및 실행 방법 (Installation & Usage)

### 1. 요구 사항 (Prerequisites)

Python 3.8 이상 환경이 필요하며, 아래의 필수 라이브러리를 설치해야 합니다.

```bash
pip install streamlit pandas matplotlib numpy xlsxwriter chardet

```

### 2. 애플리케이션 실행 (Run)

터미널(또는 명령 프롬프트)에서 소스 코드(`App_Main.py`)가 위치한 디렉토리로 이동한 후, 아래 명령어를 실행합니다.

```bash
streamlit run App_Main.py

```

### 3. 사용 방법 (How to Use)

1. 앱이 실행되면 브라우저 좌측 사이드바에서 **CSV 로그 파일**을 업로드합니다.
2. 로딩이 완료되면 우측 대시보드에서 전체 수율과 채널별 통계를 확인합니다.
3. 사이드바의 **[결함 시료 선택]** 메뉴에서 `전체 선택`을 누르거나 개별 시료를 클릭하여 탭별 상세 분석 결과를 확인합니다.
4. 분석이 끝나면 사이드바 하단의 **[📥 Download Report]** 버튼을 클릭하여 완성된 엑셀 리포트를 저장합니다.

## 📁 프로젝트 구성 (Project Structure)

* `App_Main.py`: 메인 스트림릿 애플리케이션 소스 코드
* `logo.png` *(선택)*: 화면 우측 상단에 표시될 로고 이미지
* `excel_icon.png` *(선택)*: 사이드바 엑셀 다운로드 섹션에 표시될 아이콘 이미지

## 👨‍💻 개발 및 담당자 (Credits)

* **Provided by**: JW Lee, JJ Kim

---
