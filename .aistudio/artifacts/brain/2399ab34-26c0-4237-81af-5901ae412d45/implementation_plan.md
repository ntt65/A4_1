# [A4_1] TF-IDF 기반 지능형 문서 검색기 & 분류 시스템 구현 계획 (개정안)

Scikit-learn 내장 알고리즘에 의존하지 않고, 순수 **NumPy 기반 벡터 연산**으로 TF-IDF 수식, 코사인 유사도 검색, 머신러닝 분류 및 고급 검색 알고리즘(BM25, 역색인)을 구현하며, 엄밀한 수치적 일치 검증($10^{-6}$ 미만)과 데이터 누수 방지 원칙을 철저히 준수하는 엔지니어링 구현 계획서입니다.

---

### User Review & Critical Decisions

> [!IMPORTANT]
> 전문가 코드 리뷰 및 요구사항 검토를 거쳐 다음 7가지 핵심 엔지니어링 기준이 보완·확정되었습니다.

1. **데이터 누수(Data Leakage) 엄격 차단**:
   - 어휘 사전(Vocabulary)과 IDF 통계량은 **오직 Train 데이터(80%)로만 `fit`** 합니다. Test 데이터(20%)는 오직 `transform`만 수행합니다.
   - 8:2 단일 분할(`random_state=42`)을 초기에 1회 고정 수행하고, 모든 비교 실험(TF 변형, 분류 모델, 검증)에서 동일 분할을 재사용합니다.
2. **Scikit-learn 1:1 오차 $10^{-6}$ 미만 보장 프로토콜**:
   - `smooth_idf=True`, `norm='l2'`, `sublinear_tf=False` 일치화.
   - 토큰화 정규식(`r'(?u)\b\w\w+\b'`, 1글자 단어 제외) 및 소문자화, 단어 사전 알파벳 오름차순 정렬을 Sklearn과 완전 일치화.
   - `float64` 정밀도를 강제하여 TF(Count), DF, IDF, L2 정규화 행렬을 단계별로 assertion 검증.
3. **데이터 윤리 및 저작권 준수 (과제 원문 5쪽)**:
   - 20 Newsgroups 데이터셋의 개인정보 미포함(PII Check) 및 학술/공개 라이선스 상태를 체크리스트와 수락 기준에 공식 반영.
4. **수치 안정성 및 Zero-Division 방어**:
   - **Double Normalization**: $\text{tf} > 0$인 원소에만 $0.5 + 0.5 \times \frac{\text{tf}}{\max(\text{tf})}$를 적용하고, $\text{tf} = 0$은 0으로 보존.
   - **영벡터 방어**: OOV(사전 미등재) 단어만 있는 쿼리나 빈 문서의 L2 Norm이 0일 경우, 나누기 에러를 회피하고 영벡터를 반환.
5. **검색 시간 복잡도 표기 정정 및 벤치마크 기준**:
   - Dense 행렬 연산 $D \cdot q$의 이론적 복잡도는 **$O(N \cdot |V|)$** 이며, Top-K 정렬은 `np.argpartition`을 활용해 $O(N + K \log K)$로 최적화.
   - 역색인 벤치마크는 **인덱스 구축 시간**과 **질의 실행 시간**을 명확히 분리하고, 전체 스캔과 동일 점수·순위가 나오는지 동치성(Equivalence)을 사전 검증.
6. **데이터 로더 및 재현성 강화**:
   - `fetch_20newsgroups`의 메타데이터 누수(헤더, 푸터, 인용문)를 차단하기 위해 `remove=('headers', 'footers', 'quotes')` 옵션 적용. 로컬 캐시 경로 명시 및 문서 수/클래스 분포 로깅.
7. **BM25 수식 및 검증 기준 구체화**:
   - 음수 IDF 방지형 Robertson-Spärck Jones IDF 및 $k_1=1.5, b=0.75$, 문서 길이 페널티 수식 채택. Toy 예제(문서 3건) 수작업 계산 검증 추가.
8. **파이프라인 작업 순서 재정렬**:
   - **M1 (데이터 & 전처리) → M2 (NumPy TF-IDF & 오차 검증) → M3 (검색 시스템) & M4 (분류 모델) → M6 (보너스: TF변형, BM25, 역색인) → M5 (최종 3단 리포트 & 패키징)**.

---

### 1. Overview & Core Concept

- **목적**: 정보 검색(IR) 및 자연어 처리(NLP) 파이프라인 전 과정을 서드파티 라이브러리의 블랙박스 없이 순수 NumPy 기반 수식으로 구현합니다.
- **핵심 목표**: Scikit-learn의 `TfidfVectorizer` 결과와의 최대 오차를 $10^{-6}$ 이내로 제어하고, Bag-of-Words(BoW)의 구조적 한계(어순 무시, 문맥 소실, 동음이의어/다의어)를 정량적 지표와 5건 이상의 실제 오분류 사례 분석을 통해 규명합니다.
- **엔지니어링 철학**: 철저한 데이터 누수(Data Leakage) 방지, 수치적 안정성(Zero-Division Guard), 정확한 계산 복잡도 분석.

---

### 2. User Experience & CLI Interface Design

- **인덱스 빌드 및 저장 / 로드 아키텍처**:
  - `python main.py --build-index`: Train 셋을 기반으로 어휘 사전, IDF 벡터, 문서 정규화 행렬을 생성하여 `artifacts/index.pkl`로 직렬화 저장.
  - `SearchEngine.load()`: 기구축된 인덱스를 로드하여 재학습 없이 초고속 질의 응답 제공.
- **CLI 실행 인터페이스 (과제 요구 안 A)**:
  ```bash
  # 1. 문서 검색 실행
  python main.py --query "satellite orbit launch system" --topk 5

  # 2. Scikit-learn과의 TF-IDF 수치 오차 단계별 검증 (Max error < 1e-6)
  python main.py --verify

  # 3. 문서 분류 모델 평가 및 혼동 행렬 시각화
  python main.py --train-classifier

  # 4. 보너스 과제 실행 (TF 변형 비교, BM25 비교, 역색인 벤치마크)
  python main.py --bonus
  ```
- **함수형 인터페이스 (과제 요구 안 B)**:
  ```python
  from src.search_engine import SearchEngine

  engine = SearchEngine.load("artifacts/index.pkl")
  results = engine.search(query="computer graphics 3d rendering", top_k=5)
  # 반환 시그니처: List[Tuple[float, int, str]]
  # [(score, doc_id, text_snippet), ...]
  for score, doc_id, snippet in results:
      print(f"[{score:.4f}] (Doc #{doc_id}) {snippet}")
  ```

---

### 3. Key Product Decisions & Mathematical Specifications

#### A. 데이터 누수 방지 원칙
$$\text{Train Corpus } D_{\text{train}} \xrightarrow{\text{fit}} \text{Vocab } V, \text{IDF Vector}$$
$$\text{Train Matrix } X_{\text{train}} = \text{transform}(D_{\text{train}}), \quad \text{Test Matrix } X_{\text{test}} = \text{transform}(D_{\text{test}})$$
Test 데이터의 문서 빈도(DF)나 신규 단어는 Vocabulary 및 IDF 계산에 일절 개입하지 않으며, Test 문서 내 OOV 단어는 무시됩니다.

#### B. Scikit-learn 동치 수식 ($10^{-6}$ 오차 보장)
- **토큰화**: Sklearn 기본 정규식 `r'(?u)\b\w\w+\b'` 준수 (한 글자 단어 제외, 소문자화).
- **어휘 정렬**: 사전의 모든 키를 알파벳 오름차순(`sorted(vocab.keys())`)으로 0부터 정수 인덱싱.
- **Smooth IDF**:
  $$\text{idf}(t) = \ln\left(\frac{1 + N_{\text{train}}}{1 + \text{df}_{\text{train}}(t)}\right) + 1$$
- **L2 Normalization & Zero-Division 방어**:
  $$v_{\text{norm}} = \begin{cases} \frac{v}{\|v\|_2}, & \|v\|_2 > 0 \\ \mathbf{0}, & \|v\|_2 = 0 \end{cases}$$

#### C. TF 변형 수식 및 Zero-Safe 규약 (Bonus)
1. **Raw Count**: $\text{TF}(t, d) = f_{t, d}$
2. **Log Normalization**: $\text{TF}(t, d) = \ln(1 + f_{t, d})$
3. **Double Normalization ($K=0.5$)**:
   $$\text{TF}(t, d) = \begin{cases} 0.5 + 0.5 \times \frac{f_{t, d}}{\max_{t' \in d} f_{t', d}}, & f_{t, d} > 0 \\ 0, & f_{t, d} = 0 \end{cases}$$

#### D. BM25 수식 (Robertson-Spärck Jones)
$$\text{IDF}_{\text{BM25}}(t) = \ln\left(\frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5} + 1\right)$$
$$\text{Score}(D, Q) = \sum_{t \in Q} \text{IDF}_{\text{BM25}}(t) \cdot \frac{f_{t, D} \cdot (k_1 + 1)}{f_{t, D} + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
($k_1 = 1.5$, $b = 0.75$, $|D|$는 문서 단어 수, $\text{avgdl}$은 전체 Train 문서의 평균 길이)

---

### 4. Technical Architecture & System Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│               Data Acquisition & Leakage Prevention (M1)              │
│        20 Newsgroups (4 Categories, ~2,500 docs, remove headers/footers)│
│        ─── Fixed 8:2 Split ───► Train Set (80%) | Test Set (20%)      │
│        Check: License (BSD) & PII Sanitization Confirmed               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│           Preprocessing Pipeline (Regex r'(?u)\b\w\w+\b', Stopwords)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│        Custom NumPy TF-IDF (M2)       │  Verification Protocol (M2)    │
│ • fit(Train only) -> Vocab & IDF (f64) │ • Token, Vocab, Shape Match    │
│ • transform(Train), transform(Test)   │ • Stepwise DF/IDF/Matrix Comp  │
│ • Zero-Division Guard (L2 Norm safe)  │ • Max Error < 1e-6 Assertion   │
└─────────────────┬─────────────────────┴────────────────────────────────┘
                  │
        ┌─────────┴───────────────┬────────────────────────┐
        ▼                         ▼                        ▼
┌────────────────┐      ┌──────────────────┐     ┌───────────────────┐
│ Search Engine  │      │  Classification  │     │   Bonus Modules   │
│     (M3)       │      │      (M4)        │     │       (M6)        │
│• Cosine Matrix │      │• Logistic Reg    │     │• TF Variations    │
│  D @ q: O(N|V|)│      │  (Train fit only)│     │  (Zero-safe)      │
│• Top-K Partial │      │• Accuracy & F1   │     │• BM25 Engine      │
│  Sort O(N+KlogK│      │• Confusion Matrix│     │• Inverted Index   │
│• Save/Load Idx │      │• 5 Error Analyses│     │  Equiv & Benchmark│
└────────────────┘      └──────────────────┘     └───────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    Final Deliverables & Packaging (M5)                 │
│ • README.md (3-Tier Report: Summary, Verification/Exp, BoW Limits)     │
│ • requirements.txt (Pinned dependencies for reproducibility)           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 GitHub 이슈(Issue)별 상세 마일스톤 및 체크리스트

GitHub 저장소(`https://github.com/ntt65/A4_1/`)에 복사하여 즉시 등록 가능한 이슈별 명세입니다.

### [Issue #1] [Data] 20 Newsgroups 수집, 전처리 및 데이터 누수 방지 분할
- **Labels**: `enhancement`, `phase-1`, `data-pipeline`
- **Milestone**: `M1 - Dataset & Preprocessing`
- **체크리스트**:
  - [ ] `fetch_20newsgroups` 4개 카테고리(`sci.space`, `comp.graphics`, `rec.sport.baseball`, `talk.politics.mideast`) 다운로드 및 로컬 캐싱 설정
  - [ ] 메타데이터 누수 및 레이블 힌트 제거를 위한 `remove=('headers', 'footers', 'quotes')` 옵션 적용
  - [ ] 실제 로드된 문서 수(총 ~2,500건) 및 카테고리별 분포 로깅
  - [ ] 데이터 윤리 및 보안 확인: 개인정보(PII) 미포함 확인 및 BSD 라이선스 고지
  - [ ] **데이터 누수 방지**: Train/Test 8:2 단일 분할(`random_state=42`) 고정 및 결측치 제거
  - [ ] Sklearn 호환 텍스트 정제 및 토큰화 정규식(`r'(?u)\b\w\w+\b'`) 구현 (1글자 단어 제외, 소문자화)
  - [ ] 영어 표준 불용어(Stopwords) 정의 및 제거 모듈 구축
- **수락 기준 (Acceptance Criteria)**:
  - [ ] Train 80% / Test 20% 분할이 고정 저장되어 이후 모든 모듈에서 동일 데이터셋을 참조할 것
  - [ ] PII 및 저작권 체크리스트가 로그 또는 문서에 명시될 것

### [Issue #2] [Core-NLP] NumPy 기반 TF-IDF 직접 구현 및 Sklearn $10^{-6}$ 검증
- **Labels**: `core-algorithm`, `phase-2`, `math`
- **Milestone**: `M2 - TF-IDF Custom Implementation`
- **체크리스트**:
  - [ ] `CustomTfidfVectorizer` 클래스 구조 설계 (`fit`, `transform`, `fit_transform`)
  - [ ] **데이터 누수 방지 준수**: `fit`은 오직 Train 데이터만으로 수행, 어휘 사전 및 IDF 산출
  - [ ] 어휘 사전 키를 알파벳 오름차순으로 정렬하여 열 인덱스 1:1 매핑
  - [ ] TF 계산: 문서별 단어 빈도 행렬(`float64`) 생성
  - [ ] Smooth IDF 계산: $\text{idf}(t) = \ln((1 + N)/(1 + \text{df}(t))) + 1$
  - [ ] **수치 안정성**: L2 Norm 계산 시 0으로 나누기 방어 코드 (`norm == 0`인 경우 영벡터 반환)
  - [ ] `verify_tfidf.py` 단계별 정밀 검증 스크립트 작성:
    - [ ] Vocabulary 일치 및 Shape 일치 확인
    - [ ] DF 및 IDF 벡터 오차 검증
    - [ ] 최종 L2 정규화 TF-IDF 행렬 최대 오차 $\max |X_{\text{custom}} - X_{\text{sklearn}}| < 1.0 \times 10^{-6}$ 검증
    - [ ] 평균 오차(Mean Absolute Error) 로깅
- **수락 기준 (Acceptance Criteria)**:
  - [ ] Sklearn과의 최대 오차가 $1.0 \times 10^{-6}$ 미만으로 판정 통과(PASS) 로그 출력
  - [ ] Test 문장에 대해 신규 어휘가 배제된 상태로 안정적 transform 수행 확인

### [Issue #3] [Search] 코사인 유사도 검색 엔진, 인덱스 직렬화 및 CLI 구축
- **Labels**: `feature`, `phase-3`, `cli`
- **Milestone**: `M3 - Search Engine System`
- **체크리스트**:
  - [ ] 인덱스 영속화 파이프라인: Train 데이터 기반 인덱스 빌드 및 `artifacts/index.pkl` 저장/로드(`SearchEngine.load()`)
  - [ ] 쿼리 벡터 변환: Train 단어 사전 및 IDF를 이용한 L2 정규화 쿼리 벡터 생성 (Zero-vector 방어)
  - [ ] 코사인 유사도 벡터 연산: 정규화된 문서 행렬 $D$와 쿼리 벡터 $q$ 간 행렬 곱 ($O(N \cdot |V|)$ dense 연산)
  - [ ] Top-K 검색 가속: `np.argpartition`을 이용한 $O(N + K \log K)$ 상위 5개 문서 인덱싱
  - [ ] 검색 결과 포맷터: 유사도 스코어(소수점 4자리), 문서 ID, 150자 스니펫 출력
  - [ ] 과제 필수 인터페이스 구현:
    - [ ] **(안 A) CLI**: `python main.py --query "..." --topk 5`
    - [ ] **(안 B) 함수 시그니처**: `search(query: str, top_k: int = 5) -> List[Tuple[float, int, str]]`
- **수락 기준 (Acceptance Criteria)**:
  - [ ] 사전에 없는 단어만 입력되었을 때 에러 없이 유사도 0과 빈 스니펫/안내 반환
  - [ ] CLI 및 함수 시그니처가 과제 요구사항 포맷과 100% 일치할 것

### [Issue #4] [ML/Evaluation] Logistic Regression 분류 모델 학습 및 BoW 한계 분석
- **Labels**: `machine-learning`, `phase-4`, `evaluation`
- **Milestone**: `M4 - Document Classification`
- **체크리스트**:
  - [ ] 직접 구현한 TF-IDF로 변환된 $X_{\text{train}}$으로 `LogisticRegression(random_state=42)` 학습
  - [ ] Test 데이터 $X_{\text{test}}$에 대해 Accuracy 및 Macro-averaged F1-Score 측정
  - [ ] Scikit-learn `confusion_matrix` 및 `classification_report` 생성
  - [ ] `matplotlib` / `seaborn` 기반 혼동 행렬 시각화 이미지(`assets/confusion_matrix.png`) 저장
  - [ ] **오분류 사례 5건 정밀 분석**:
    - [ ] 실제 레이블 vs 예측 레이블 불일치 문서 5건 추출
    - [ ] BoW 본질적 한계(어순 무시, 문맥 소실, 도메인 중의어, 부정어 역전 등)와 연결하여 원인 서술
- **수락 기준 (Acceptance Criteria)**:
  - [ ] Train/Test 분할 누수 없이 독립된 평가 지표 산출
  - [ ] 혼동 행렬 이미지 생성 및 5건 이상의 구체적인 오분류 원인 분석 리포트 완성

### [Issue #5] [Bonus] TF 변형 비교, BM25 엔진 및 역색인(Inverted Index)
- **Labels**: `bonus`, `advanced-ir`
- **Milestone**: `M6 - Bonus Advanced Tasks`
- **체크리스트**:
  - [ ] **1. 다양한 TF 변형 구현 & 정량 평가**:
    - [ ] Raw Count, Log Normalization $\ln(1+\text{tf})$, Double Normalization $0.5 + 0.5 \frac{\text{tf}}{\max(\text{tf})}$ 구현
    - [ ] **Zero-Safe 규약 준수**: Double Normalization 적용 시 $\text{tf}=0$인 항목은 0으로 보존
    - [ ] 각 TF 방식별 분류 F1-Score 및 검색 랭킹 변화 정량 비교표 작성
  - [ ] **2. BM25 알고리즘 직접 구현**:
    - [ ] 음수 방지형 Robertson-Spärck Jones IDF 및 $k_1=1.5, b=0.75$, $\text{avgdl}$ 수식 적용
    - [ ] 3개 문서로 구성된 Toy example 수작업 계산 일치성 단위 테스트 통과
    - [ ] 동일 쿼리에 대해 TF-IDF vs BM25 상위 Top-5 검색 결과 비교 분석
  - [ ] **3. 역색인(Inverted Index) 구현 & 동치성/성능 벤치마크**:
    - [ ] 단어 $\rightarrow$ 포스팅 리스트(`doc_id`, `tf`) 구조 구현
    - [ ] **동치성 검증**: 역색인 기반 검색 결과가 전체 스캔(Full-scan) 검색과 동일한 점수·순위를 반환하는지 검증
    - [ ] **벤치마크 분리**: 인덱스 구축 시간과 단일 쿼리 실행 시간(ms)을 분리 측정하여 전체 스캔 대비 속도 비교
- **수락 기준 (Acceptance Criteria)**:
  - [ ] Toy example 단위 테스트 통과 및 역색인-전체스캔 간 검색 결과 100% 일치
  - [ ] TF 변형 비교 및 BM25 비교 분석 결과 리포트 수록

### [Issue #6] [Documentation] 필수 3단 구성 README.md 및 제출 패키징
- **Labels**: `documentation`, `phase-5`
- **Milestone**: `M5 - Final Report & Packaging`
- **체크리스트**:
  - [ ] `README.md` 과제 규격 필수 3단 구성 작성:
    - [ ] **1단: 구현 요약**: 전처리 파이프라인, TF-IDF 수학 공식, 검색/분류/보너스 아키텍처
    - [ ] **2단: 검증/실험 결과**: TF-IDF 검증 오차 로그, 분류 성능표, 혼동 행렬 시각화, BM25/역색인 실험표
    - [ ] **3단: 한계와 개선 방향**: BoW 한계 분석, 5개 오분류 원인, 사전학습 언어모델(Transformer) 후속 학습 방향
  - [ ] 실행 방법 안내 (CLI 커맨드 예시 및 `search()` 함수 호출 예시)
  - [ ] Python 3.10+ 구동용 의존성 명시 (`requirements.txt`)
- **수락 기준 (Acceptance Criteria)**:
  - [ ] 과제 PDF 가이드라인의 모든 필수 문항 및 결과 양식이 누락 없이 완비될 것
  - [ ] 새 가상환경에서 `pip install -r requirements.txt` 후 재현 실행이 완벽할 것
