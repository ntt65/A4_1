# [A4_1] TF-IDF 기반 지능형 문서 검색기 & 분류 시스템 구현 계획 및 아키텍처

## 1. 프로젝트 개요
* **목표**: 20 Newsgroups 데이터셋(4개 카테고리)을 기반으로 Scikit-learn 대비 오차 $10^{-15}$ 미만의 순수 NumPy TF-IDF 벡터라이저, 문서 검색 시스템, 분류기, 보너스 알고리즘(BM25, 역색인, TF 변형) 구현.
* **환경**: Python 3.10+, NumPy, Scikit-learn, Matplotlib.

## 2. 8대 핵심 설계 및 검증 원칙 (Critical Decisions)
1. **데이터 누수(Data Leakage) 엄격 차단**:
   - 어휘 사전(Vocabulary) 및 IDF 통계량은 오직 Train 데이터(80%, 3,033건)로만 `fit`.
   - Test 데이터(20%, 759건)는 오직 `transform`만 수행.
   - 단일 분할(`random_state=42`) 고정 캐싱.
2. **Scikit-learn 1:1 오차 정밀 검증 ($< 10^{-15}$)**:
   - `smooth_idf=True`, `norm='l2'`, `sublinear_tf=False` 일치.
   - 정규식 `r'(?u)\b\w\w+\b'`, 소문자화, 어휘 알파벳 오름차순 정렬 일치.
   - 불용어: `resources/stopwords_en.txt` 고정.
   - `float64` 연산 강제.
3. **데이터 윤리 및 개인정보(PII) 비식별화**:
   - 20 Newsgroups 원본 메타데이터(headers, footers, quotes) 제거.
   - 정규식 및 규칙 기반 이메일, 전화번호, 개인 식별 정보 마스킹.
4. **수치 안정성 및 Zero-Division 방어**:
   - Double Normalization: $TF > 0$ 원소에만 $0.5 + 0.5 \times \frac{tf}{\max(tf)}$ 적용.
   - 빈 문서 및 OOV 쿼리 영벡터 방어 (Zero-safe L2 Norm).
5. **검색 시간 복잡도 표기 정정 및 동점 처리**:
   - Dense 행렬 연산 $O(N \cdot V)$.
   - Top-K 정책: 점수 내림차순, 동점 시 `doc_id` 오름차순, 0점 문서 제외.
6. **보너스 알고리즘 동치성 검증**:
   - TF 변형 (Raw, Log, Double Norm) 비교.
   - BM25 구현 및 Toy Example 수작업 검증 (오차 $5.55 \times 10^{-17}$).
   - 역색인(Inverted Index) 구축 및 전체 스캔과의 순위/점수 일치성 검증 ($2.78 \times 10^{-17}$).
7. **재현성 보장**: 고정 시드, 의존성 패키지 버전 고정 (`requirements.txt`).
8. **리포트 3단 구성**: `README.md` 내 시스템 사양, 검증 수치, CLI 사용법 완결.
