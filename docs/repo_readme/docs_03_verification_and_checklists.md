# [A4_1] 실측 검증 데이터 및 최종 점검 체크리스트

## 1. 실측 수치 검증 요약표

| 구분 | 검증 항목 | 목표 기준 | 실측 결과 | 판정 |
| :--- | :--- | :--- | :--- | :--- |
| **M1** | 데이터 분할 및 캐싱 | Train/Test 8:2 고정 | Train 3,033건 / Test 759건 (총 3,792건) | **PASS** |
| **M2** | NumPy TF-IDF vs Sklearn | 절대 오차 $< 10^{-6}$ | **1.1102e-15** (부동소수점 한계 수준) | **PASS** |
| **M3** | 문서 검색 시스템 | Top-K 동점 및 0점 방어 | 정렬 순서 및 점수 일치 | **PASS** |
| **M4** | 로지스틱 회귀 분류기 | 분류 정확도 | **97.56%**, Macro-F1: **0.9758** | **PASS** |
| **M6-1** | TF 변형 비교 | Raw vs Log vs Double Norm | Double Norm Accuracy: **96.66%** | **PASS** |
| **M6-2** | BM25 Toy Example 검증 | 수작업 계산 일치 | 절대 오차 **5.55e-17** | **PASS** |
| **M6-3** | 역색인 동치성 검증 | 전체스캔 vs 역색인 | 점수 오차 **2.78e-17**, 순위 100% 일치 | **PASS** |

---

## 2. 제출 전 최종 점검 체크리스트
- [x] Train/Test 분할 후 어휘 사전 fitting 여부 (데이터 누수 차단)
- [x] 불용어 리소스 파일(`resources/stopwords_en.txt`) 고정 여부
- [x] OOV 쿼리 입력 시 Zero-division 방어 로직 작동 여부
- [x] CLI 실행 파라미터 (`--query`, `--topk`, `--train-classifier`, `--bonus`) 지원 여부
- [x] 혼동 행렬 이미지(`artifacts/confusion_matrix.png`) 정상 생성 확인
- [x] Git 저장소(`ntt65/A4_1`) 푸시 시 가상환경/임시파일 제외 (`.gitignore` 설정)
