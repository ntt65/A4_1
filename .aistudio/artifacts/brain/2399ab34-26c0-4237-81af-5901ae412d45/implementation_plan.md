# [A4_1] TF-IDF 기반 지능형 문서 검색기 & 분류 시스템 구현 계획

Scikit-learn의 내장 알고리즘에 의존하지 않고, 순수 **NumPy 기반 벡터 연산**으로 TF-IDF 수식, 코사인 유사도 검색, 머신러닝 분류 및 고급 검색 알고리즘(BM25, 역색인)을 구현하고 $10^{-6}$ 이내 오차로 정밀 검증하는 NLP 시스템 구축 계획서입니다.

---

### User Review & Critical Decisions

> [!IMPORTANT]
> 사용자 질의응답을 통해 아래와 같이 핵심 아키텍처 및 구현 범위가 확정되었습니다.

- **데이터셋 및 언어**: `20 Newsgroups` 영문 데이터셋 확정 (`fetch_20newsgroups` 4개 카테고리, 약 2,500건 문서 확보, 환경 의존성 제로 및 높은 재현성 확보)
- **분류 기준 모델**: `Logistic Regression` 확정 (확률 추정치 해석, L2 정규화 특성과의 정합성 및 가중치 해석 용이)
- **과제 범위**: 필수 요구사항(전처리, TF-IDF 직접 구현, 오차 검증, 코사인 검색, 분류 및 오분류 분석) + **보너스 과제 3종 전체(TF 변형 수식 비교, BM25 엔진 구현, 역색인 구조 벤치마크) 전면 구현**

---

### 1. Overview & Core Concept

- **What It Does**: 텍스트 데이터의 원시 문자열에서부터 토큰화, 어휘 사전 구축, TF-IDF 가중치 계산, 코사인 유사도 검색, 머신러닝 분류(Logistic Regression), 그리고 BM25와 역색인(Inverted Index) 검색에 이르는 자연어 처리 및 정보 검색(IR)의 전 과정을 라이브러리 블랙박스 없이 직접 구현합니다.
- **Target Audience / Purpose**: 정보 검색(IR) 및 자연어 처리(NLP) 원리를 학습하고, Bag-of-Words(BoW)의 동작 원리와 한계(어순/문맥 소실, 동음이의어 등)를 정량적 실험 및 오분류 사례를 통해 입증하는 교육·엔지니어링 프로젝트입니다.
- **Key Value**: Scikit-learn의 `TfidfVectorizer` 결과와 최대 오차 $10^{-6}$ 이내로 일치하는 수학적으로 엄밀한 NumPy 구현체 제공.

---

### 2. User Experience & CLI Interface Design

- **CLI 실행 인터페이스 (안 A)**:
  ```bash
  # 1. 문서 검색 실행
  python main.py --query "space satellite orbit launch" --topk 5

  # 2. Scikit-learn과의 TF-IDF 수치 오차 검증
  python main.py --verify

  # 3. 문서 분류 모델 평가 및 혼동 행렬 생성
  python main.py --train-classifier

  # 4. 보너스 과제 실행 (BM25 vs TF-IDF 비교 및 역색인 벤치마크)
  python main.py --bonus
  ```
- **함수형 인터페이스 (안 B)**:
  ```python
  from src.search_engine import SearchEngine

  engine = SearchEngine.load()
  results = engine.search(query="computer graphics 3d rendering", top_k=5)
  # 반환 형식: List[Tuple[score: float, doc_id: int, text_snippet: str]]
  for score, doc_id, snippet in results:
      print(f"[{score:.4f}] (Doc #{doc_id}) {snippet}")
  ```

---

### 3. Key Product Decisions & Trade-Offs

- **데이터셋: 20 Newsgroups 4개 카테고리 선정**
  - *선택 접근*: `sci.space`, `comp.graphics`, `rec.sport.baseball`, `talk.politics.mideast` 4개 분야 약 2,500건 추출.
  - *선정 이유*: 외부 파일 다운로드 실패나 자바(JVM) 가상머신/KoNLPy 설치 충돌 위험이 전혀 없으며, Scikit-learn 내장 데이터셋으로 평가 및 재현이 100% 보장됨.
- **수학적 일치성: Scikit-learn `smooth_idf=True` 및 `norm='l2'` 수식 준수**
  - *선택 접근*:
    $$\text{idf}(t) = \ln\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1$$
    $$\text{tfidf}(d, t) = \text{tf}(d, t) \times \text{idf}(t), \quad v_{\text{norm}} = \frac{v}{\|v\|_2}$$
  - *선정 이유*: Scikit-learn 공식 수식과 단어 정렬 순서를 완벽히 일치시켜 $10^{-9}$ 이하의 극소 오차를 달성.
- **코사인 유사도 연산 최적화**:
  - *선택 접근*: 행렬 정규화가 이미 완료된 문서 행렬 $D$와 정규화된 쿼리 $q$ 간의 행렬 곱 $D \cdot q^T$ 적용.
  - *선정 이유*: 루프문 없는 순수 벡터 내적으로 검색 레이턴시를 $1\text{ms}$ 미만으로 단축.

---

### 4. Technical Architecture & System Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Text Corpus (20 Newsgroups)                     │
│                        (4 Categories, ~2,500 Docs)                     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               Preprocessor (Cleaning, Tokenize, Stopwords)             │
│            • Regex Lowercase, Strip Header/Punctuation                 │
│            • Alpha Token Filtering & Stopwords Removal                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
┌───────────────────────────────────┐ ┌──────────────────────────────────┐
│   NumPy TF-IDF Vectorizer (Core)  │ │   Verification vs Scikit-learn   │
│   • Vocabulary Alpha-sorted Index │ │   • Max Error < 1e-6 Assertion   │
│   • Smooth IDF & L2 Normalization │ │   • Mean Error Logging           │
└─────────────────┬─────────────────┘ └──────────────────────────────────┘
                  │
        ┌─────────┴───────────────┬────────────────────────┐
        ▼                         ▼                        ▼
┌────────────────┐      ┌──────────────────┐     ┌───────────────────┐
│ Search Engine  │      │  Classification  │     │   Bonus Modules   │
│ • Cosine Sim   │      │  • Logistic Reg  │     │   • BM25 Engine   │
│ • Top-K Snippet│      │  • Acc / F1 / CM │     │   • Inverted Index│
│ • CLI Handler  │      │  • Error Analysis│     │   • TF Variations │
└────────────────┘      └──────────────────┘     └───────────────────┘
```

---

## 📋 GitHub 이슈(Issue)별 상세 마일스톤 및 체크리스트

GitHub 저장소(`https://github.com/ntt65/A4_1/`)에 등록하여 진행 상황을 체크할 수 있는 이슈별 명세입니다.

### [Issue #1] [Data] 20 Newsgroups 데이터셋 로더 및 전처리 파이프라인
- **Labels**: `enhancement`, `phase-1`, `data-pipeline`
- **체크리스트**:
  - [ ] Scikit-learn `fetch_20newsgroups` 4개 카테고리(`sci.space`, `comp.graphics`, `rec.sport.baseball`, `talk.politics.mideast`) 로더 작성
  - [ ] Train/Test 8:2 분할 (`random_state=42`) 고정 및 결측치 검증
  - [ ] 텍스트 정제(소문자화, 이메일/특수문자/숫자 정규식 제거)
  - [ ] 영문 공백 및 알파벳 단위 토큰화 파이프라인 함수 구현
  - [ ] 영어 표준 불용어(Stopwords) 제거 및 어휘 정제

### [Issue #2] [Core] NumPy 기반 TF-IDF 직접 구현 및 Sklearn 오차 검증 (< 1e-6)
- **Labels**: `core-algorithm`, `phase-2`, `math`
- **체크리스트**:
  - [ ] `CustomTfidfVectorizer` 클래스 설계 (`fit`, `transform`, `fit_transform`)
  - [ ] 알파벳 순 어휘 사전(Vocabulary) 인덱싱
  - [ ] TF 계산 (Raw Frequency matrix)
  - [ ] Smooth IDF 계산: $\text{idf}(t) = \ln((1 + N)/(1 + \text{df}(t))) + 1$
  - [ ] Euclidean L2 Norm 단위 벡터화 적용
  - [ ] `verify_tfidf.py` 검증 스크립트 작성 및 최대/평균 오차 $10^{-6}$ 이내 합격 로그 출력

### [Issue #3] [Search] 코사인 유사도 벡터 검색 엔진 & CLI 인터페이스
- **Labels**: `feature`, `phase-3`, `cli`
- **체크리스트**:
  - [ ] 쿼리 텍스트를 학습된 사전 기반 L2 정규화 TF-IDF 벡터로 변환하는 함수 구현
  - [ ] NumPy 행렬-벡터 곱($D \cdot q$) 기반 코사인 유사도 연산 함수 구현
  - [ ] 유사도 상위 Top-K 문서 정렬 및 점수, 문서 ID, 150자 스니펫 포맷터 구현
  - [ ] `python main.py --query "..." --topk 5` CLI 인터페이스 완성
  - [ ] 파이썬 모듈 인터페이스 `search(query, top_k=5)` 시그니처 구현

### [Issue #4] [ML] Logistic Regression 문서 분류 모델 학습, 평가 및 오분류 분석
- **Labels**: `machine-learning`, `phase-4`, `evaluation`
- **체크리스트**:
  - [ ] 직접 구현한 TF-IDF 특성 행렬 기반 `LogisticRegression` 학습 파이프라인 구축
  - [ ] Test Set 20%에 대한 Accuracy, Macro-F1 지표 산출
  - [ ] 혼동 행렬(Confusion Matrix) 계산 및 `assets/confusion_matrix.png` 시각화 저장
  - [ ] 오분류된 문서 5건 추출 및 BoW 본질적 한계(어순 무시, 문맥 소실, 다의어 등) 원인 분석 리포트 작성

### [Issue #5] [Docs] 필수 3단 구성 README.md 및 재현 환경(requirements.txt)
- **Labels**: `documentation`, `phase-5`
- **체크리스트**:
  - [ ] 1단: 구현 요약 (전처리 파이프라인, 수식, 시스템 구조)
  - [ ] 2단: 검증 및 실험 결과 (TF-IDF 검증 오차 로그, 모델 분류 성능표, 혼동 행렬)
  - [ ] 3단: 한계와 개선 방향 (BoW 한계, 오분류 5건 분석, Word2Vec/BERT 등 후속 발전 방향)
  - [ ] CLI 실행 커맨드 및 `search()` 함수 사용법 명시
  - [ ] `requirements.txt` 의존성 패키징

### [Issue #6] [Bonus] TF 변형 비교, BM25 엔진 및 역색인(Inverted Index)
- **Labels**: `bonus`, `advanced-ir`
- **체크리스트**:
  - [ ] TF 변형 3종(Raw, Log Norm $\ln(1+\text{tf})$, Double Norm $0.5+0.5\frac{\text{tf}}{\max}$) 구현 및 분류 성능 비교
  - [ ] BM25 수식 ($k_1=1.5, b=0.75$, 문서 길이 페널티) NumPy 직접 구현
  - [ ] 동일 쿼리에 대한 TF-IDF vs BM25 검색 랭킹 비교 분석
  - [ ] 단어 $\rightarrow$ 문서 ID 리스트 형태의 역색인(Inverted Index) 구조 구현
  - [ ] 역색인 기반 검색 vs 전체 스캔 검색 간 실행 속도 및 시간 복잡도($O(|Q| \cdot L)$ vs $O(N \cdot |V|)$) 벤치마크
