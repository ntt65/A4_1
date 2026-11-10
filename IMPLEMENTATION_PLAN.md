# [Project] TF-IDF 기반 지능형 문서 검색기 & 분류 시스템
> **과제명**: 원하는 내용이 들어있는 문서를 똑똑하게 찾아주는 검색기 만들기  
> **용도**: GitHub 저장소 개설 시 마일스톤 및 이슈(GitHub Issues)로 바로 등록하여 사용할 수 있는 종합 실시계획서 & 체크리스트

---

## 📌 전체 마일스톤 로드맵 (Milestones Overview)

| 마일스톤 ID | 단계명 | 핵심 산출물 | 예상 공수 | 상태 |
| :--- | :--- | :--- | :--- | :--- |
| **M1** | 데이터셋 준비 및 전처리 파이프라인 | `dataset.py`, `preprocess.py` | 8h | 대기 |
| **M2** | NumPy 기반 TF-IDF 직접 구현 & 검증 | `tfidf_custom.py`, `verify_tfidf.py` | 18h | 대기 |
| **M3** | 코사인 유사도 기반 문서 검색 엔진 & CLI | `search_engine.py`, `main.py` | 12h | 대기 |
| **M4** | 문서 분류 모델 학습 & 오분류 분석 | `classifier.py`, `confusion_matrix.png` | 14h | 대기 |
| **M5** | 필수 3단 구성 리포트 및 패키징 | `README.md`, `requirements.txt` | 8h | 대기 |
| **M6 (Bonus)** | 고도화 (TF 변형, BM25, 역색인 구현) | `bonus/bm25.py`, `bonus/inverted_index.py` | 10h | 대기 |

---

## 📋 GitHub 이슈(Issue) 단위 상세 실시계획서 & 체크리스트

각 이슈는 GitHub의 **New Issue** 본문으로 바로 복사하여 사용할 수 있도록 Markdown 체크박스(`- [ ]`), 라벨, 수락 기준(Acceptance Criteria)으로 구성되었습니다.

---

### [Issue #1] [Data/Preprocessing] 데이터셋 확보 및 텍스트 전처리 파이프라인 구축
- **Labels**: `enhancement`, `phase-1`, `data-pipeline`
- **Milestone**: `M1 - Dataset & Preprocessing`

#### 🎯 목표
최소 500개 이상의 문서와 2개 이상의 카테고리를 포함하는 텍스트 데이터를 확보하고, 검색 및 모델 학습에 적합하도록 정제·토큰화·불용어 제거 파이프라인을 구축합니다.

#### 📝 작업 내용
- [ ] 데이터셋 선정 및 로더 모듈 작성 (`20 Newsgroups` 또는 `NSMC`)
  - [ ] 20 Newsgroups 기준: 최소 4개 카테고리(`sci.space`, `comp.graphics`, `rec.sport.baseball`, `talk.politics.mideast` 등 1,500~3,000건) 확보
  - [ ] Train/Test 8:2 분할 및 `random_state=42` 고정
- [ ] 텍스트 정제(Cleaning) 함수 구현
  - [ ] 소문자 변환(Lowercasing)
  - [ ] 특수문자, 숫자, 이메일 헤더, 노이즈 정규표현식(`re`) 제거
- [ ] 토큰화(Tokenization) 및 어휘 사전(Vocabulary) 구축
  - [ ] 공백 및 영문 정규식 토큰화 (또는 NLTK WordPunctTokenizer)
  - [ ] 불용어(Stopwords) 목록 정의 및 필터링 (NLTK English stopwords 또는 기본 리스트)
- [ ] 전처리 기준(특수문자 허용 범위, 불용어 제거 기준) 정리 문서화

#### 🔍 수락 기준 (Acceptance Criteria)
- [ ] 500개 이상의 문서가 정상 로드되고 결측치가 제거될 것
- [ ] 전처리 파이프라인 함수 `preprocess_pipeline(text: str) -> list[str]` 단위 테스트 통과
- [ ] Train/Test 데이터셋 크기와 카테고리 분포가 명확히 분리될 것

---

### [Issue #2] [Core-NLP] NumPy 기반 TF-IDF 직접 구현 및 Scikit-learn 정밀 검증
- **Labels**: `core-algorithm`, `phase-2`, `math`
- **Milestone**: `M2 - TF-IDF Custom Implementation`

#### 🎯 목표
외부 머신러닝 라이브러리의 내장 TF-IDF를 쓰지 않고, 순수 **NumPy**만을 사용하여 TF 계산, Smooth IDF 계산, L2 정규화 행렬 연산을 구현하며, Scikit-learn의 `TfidfVectorizer` 결과와 오차 $10^{-6}$ 이내를 달성합니다.

#### 📝 작업 내용
- [ ] `CustomTfidfVectorizer` 클래스 구조 설계
  - [ ] `fit(documents)`: Vocabulary 사전 생성 및 오름차순 인덱싱, DF(Document Frequency) 계산
  - [ ] `transform(documents)`: TF 계산 및 TF-IDF 행렬 변환
  - [ ] `fit_transform(documents)`
- [ ] 핵심 수식 NumPy 벡터화 구현
  - [ ] **TF 계산**: 문서별 단어 출현 횟수(Raw count)
  - [ ] **Smooth IDF 계산**: $\text{idf}(t) = \ln\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1$
  - [ ] **L2 정규화**: 문서 벡터 단위 $v_{\text{norm}} = \frac{v}{\sqrt{\sum v_i^2}}$
- [ ] Scikit-learn `TfidfVectorizer`와의 1:1 비교 검증 스크립트 작성 (`verify_tfidf.py`)
  - [ ] `smooth_idf=True`, `norm='l2'`, `sublinear_tf=False` 파라미터 일치
  - [ ] `np.max(np.abs(custom_matrix - sklearn_matrix))` 계산
  - [ ] `np.mean(np.abs(custom_matrix - sklearn_matrix))` 계산
  - [ ] 오차 $10^{-6}$ 이하 판정 검증 로그 출력

#### 🔍 수락 기준 (Acceptance Criteria)
- [ ] Scikit-learn 결과와의 최대 오차(Max error)가 $1.0 \times 10^{-6}$ 미만일 것
- [ ] `CustomTfidfVectorizer`가 쿼리 단일 문장에 대해서도 동일 차원의 벡터를 안정적으로 생성할 것

---

### [Issue #3] [Search] 코사인 유사도 기반 문서 검색 엔진 구축 & CLI 인터페이스
- **Labels**: `feature`, `phase-3`, `cli`
- **Milestone**: `M3 - Search Engine System`

#### 🎯 목표
입력된 자연어 쿼리를 TF-IDF 벡터로 변환하고, NumPy 벡터 연산으로 코사인 유사도를 계산하여 가장 유사한 상위 5개 문서를 추출하는 시스템 및 CLI를 구현합니다.

#### 📝 작업 내용
- [ ] 코사인 유사도 벡터 연산 구현
  - [ ] $\text{Cosine Sim}(q, d) = \frac{q \cdot d}{\|q\|_2 \|d\|_2} = q_{\text{norm}} \cdot d_{\text{norm}}$
  - [ ] 이미 L2 정규화된 문서 행렬과 L2 정규화된 쿼리 벡터 간의 단순 행렬 곱($D \cdot q^T$)으로 $O(N)$ 최적화
- [ ] 검색 결과 추출기 구현
  - [ ] `np.argsort()` 내림차순 정렬 및 Top-K 인덱스 슬라이싱
  - [ ] 문서 ID, 유사도 스코어(0~1), 본문 스니펫(Snippet, 150자 내외) 포맷팅
- [ ] 필수 실행 인터페이스 구현
  - [ ] **(안 A) CLI 명령행 인터페이스**: `python main.py --query "deep learning image" --topk 5`
  - [ ] **(안 B) 함수 시그니처**: `search(query: str, top_k: int = 5) -> List[Tuple[float, int, str]]`

#### 🔍 수락 기준 (Acceptance Criteria)
- [ ] CLI 커맨드로 임의의 질의어 입력 시 상위 5개 문서와 점수, 스니펫이 정상 출력될 것
- [ ] 검색 함수 단위 테스트 및 예외 처리(사전에 없는 단어만 입력되었을 때 0점 반환 등) 통과

---

### [Issue #4] [ML/Evaluation] TF-IDF 기반 문서 분류 모델 학습, 평가 및 오분류/BoW 분석
- **Labels**: `machine-learning`, `phase-4`, `evaluation`
- **Milestone**: `M4 - Document Classification`

#### 🎯 목표
직접 생성한 TF-IDF 벡터 행렬을 입력 특성(Feature)으로 하여 분류 모델을 학습하고 성능(Accuracy, Macro F1)을 평가하며, Confusion Matrix 시각화 및 오분류 5건을 BoW의 본질적 한계와 연계하여 분석합니다.

#### 📝 작업 내용
- [ ] 분류 모델 파이프라인 구현
  - [ ] Train/Test 8:2 분할 (`random_state` 고정)
  - [ ] 모델: Logistic Regression (또는 Linear SVM) 학습
- [ ] 성능 지표 측정 및 출력
  - [ ] Accuracy 및 Macro-averaged F1-Score 계산
  - [ ] Scikit-learn `classification_report` 출력
- [ ] 혼동 행렬(Confusion Matrix) 시각화
  - [ ] `matplotlib` / `seaborn` 히트맵 생성 및 이미지 파일(`assets/confusion_matrix.png`) 저장
- [ ] 오분류(Misclassified) 케이스 상세 분석 (최소 5건)
  - [ ] 실제 레이블 vs 예측 레이블 불일치 샘플 추출
  - [ ] BoW 방식의 핵심 한계점(어순 무시, 문맥 소실, 동음이의어/다의어, 부정 표현 반전)과 매핑하여 원인 분석

#### 🔍 수락 기준 (Acceptance Criteria)
- [ ] Accuracy와 Macro F1-Score가 수치로 산출되어 로그에 기록될 것
- [ ] 혼동 행렬 시각화 이미지 파일이 저장될 것
- [ ] 5건 이상의 구체적인 오분류 사례 분석 문안 작성

---

### [Issue #5] [Documentation] 필수 3단 구성 README.md 및 제출 패키징
- **Labels**: `documentation`, `phase-5`
- **Milestone**: `M5 - Final Report & Packaging`

#### 🎯 목표
과제 지침에서 요구하는 필수 3단 구성(구현 요약, 검증/실험 결과, 한계와 개선 방향)에 맞추어 `README.md`를 완성하고, 재현 가능한 가상환경 설정(`requirements.txt`)을 패키징합니다.

#### 📝 작업 내용
- [ ] `README.md` 필수 3단 구조 작성
  - [ ] **1단: 구현 요약**: 전처리 파이프라인, TF-IDF 수식과 구현 과정, 검색/분류 시스템 아키텍처
  - [ ] **2단: 검증/실험 결과**: TF-IDF 검증 로그(오차율, 파라미터), 분류 모델 성능 지표, 혼동 행렬 이미지
  - [ ] **3단: 한계와 개선 방향**: BoW 한계 분석, 오분류 원인(5건), N-gram/Word2Vec/Transformer 등 후속 학습 방향
- [ ] 실행 방법 가이드 명시
  - [ ] CLI 실행 커맨드 예시 및 출력 결과 블록
  - [ ] 파이썬 코드 내 `search()` 함수 호출 예시
- [ ] `requirements.txt` 작성 (Python 3.10+ 환경용)

#### 🔍 수락 기준 (Acceptance Criteria)
- [ ] README.md에 누락된 필수 항목이 없을 것
- [ ] `pip install -r requirements.txt` 후 안내된 CLI 명령어로 즉시 재현 가능할 것

---

### [Issue #6] [Bonus] (선택 과제) TF 변형 비교, BM25 구현 및 역색인(Inverted Index)
- **Labels**: `bonus`, `advanced-ir`
- **Milestone**: `M6 - Bonus Advanced Tasks`

#### 🎯 목표
기본 과제를 확장하여 다양한 TF 변형 수식 비교, 검색 업계 표준 BM25 수식 구현, 그리고 역색인(Inverted Index) 구조를 통한 검색 시간 복잡도 개선을 검증합니다.

#### 📝 작업 내용
- [ ] **1. 다양한 TF 변형 구현 및 비교**
  - [ ] Raw Count, Log Normalization $\ln(1 + \text{tf})$, Double Normalization $0.5 + 0.5 \frac{\text{tf}}{\max(\text{tf})}$ 구현
  - [ ] TF 변형 방식에 따른 검색 랭킹 및 분류 F1 영향도 정량 비교
- [ ] **2. BM25 알고리즘 직접 구현**
  - [ ] $k_1 = 1.5, b = 0.75$, 문서 평균 길이($\text{avgdl}$) 수식 반영
  - [ ] 동일 쿼리에 대해 TF-IDF vs BM25 상위 검색 결과 비교 분석
- [ ] **3. 역색인(Inverted Index) 구현 & 성능 벤치마크**
  - [ ] `word -> [(doc_id, tf), ...]` 사전 기반 역색인 구조 구축
  - [ ] 전수 조사(Full-scan) vs 역색인 기반 검색의 시간 복잡도($O(N \cdot |V|)$ vs $O(|Q| \cdot L)$) 및 실행 시간(ms) 측정

#### 🔍 수락 기준 (Acceptance Criteria)
- [ ] `bonus/` 디렉토리에 보너스 모듈 분리 작성
- [ ] README에 TF-IDF와 BM25 결과 비교 표 포함

---

## 📅 권장 실행 일정 (예시 주차별 계획)

```
[Day 1~2] M1: 데이터셋 수집 및 정제 파이프라인 (Issue #1)
[Day 3~4] M2: NumPy TF-IDF 구현 및 Sklearn 오차 검증 (Issue #2)
[Day 5]   M3: 코사인 유사도 검색 및 CLI 구현 (Issue #3)
[Day 6]   M4: 분류 모델 학습 및 오분류 케이스 분석 (Issue #4)
[Day 7]   M5: 3단 구성 README.md 및 requirements.txt 완성 (Issue #5)
[Day 8]   M6: (선택) BM25 / 역색인 보너스 과제 구현 (Issue #6)
```
