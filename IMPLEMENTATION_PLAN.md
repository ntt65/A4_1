# [A4_1] TF-IDF 기반 지능형 문서 검색기 & 분류 시스템 구현 계획 (개정안)

현재 통합 검토 경로는 `chore/a4-1-git-workflow`에서 `main`으로 향하는 PR #12입니다. 아래 PR #9·#10 기록은 단계별 작업 이력이며, 두 PR의 모든 변경을 확인한 뒤 #12로 대체 종료하고 중복 브랜치를 정리했습니다. [통합 PR #12](https://github.com/ntt65/A4_1/pull/12), [Git 관리 정리 계획](docs/git_workflow_repair_plan.md)

**NumPy 기반 벡터 연산**으로 TF-IDF, 코사인 유사도 검색, TF 변형 및 BM25를 직접 구현하고, 역색인으로 검색을 가속합니다. 데이터 로딩, 분류 모델(Logistic Regression), 평가 및 참조 검증에는 Scikit-learn을 사용합니다. TF-IDF 최대 절대 오차 $10^{-6}$ 미만 검증과 학습·시험 데이터 분리를 구현 기준으로 삼습니다.

## 2026-11-12 실행 계획 및 진행 현황

작업 브랜치는 `feat/a4-1-nlp-verification`입니다. 실행 소스 복원과 실측 검증, 평가 산출물 통합을 이슈 #7에서 추적합니다. 기존 마일스톤 이슈 #1~#6도 관련 범위로 연결합니다. [통합 작업 이슈 #7](https://github.com/ntt65/A4_1/issues/7)

| 순서 | 실행 항목 | 현재 상태 | 확인 근거 |
| --- | --- | --- | --- |
| 1 | M1 데이터·전처리·고정 분할 | 구현·실측 완료 | 원본 3,894건, 빈 본문 107건 제외, Train 3,029 / Test 758 |
| 2 | M2 NumPy TF-IDF 대조 | 완료 | 어휘·TF·DF·IDF·Train/Test/질의 검증, Train 최대 오차 2.3315e-15 |
| 3 | M3 검색·CLI·함수·인덱스 | 완료 | Top-5 출력, 영벡터·동점 처리, 저장 후 함수 API 재로딩 |
| 4 | M4 분류·오분류 분석 | 완료 | Accuracy 92.4802%, Macro-F1 0.925329, 혼동 행렬과 실제 오분류 5건 |
| 5 | M6 TF 변형·BM25·역색인 | 완료 | BM25 Toy 오차 2.2204e-16, 역색인 7개 질의 Top-5 100% 일치 |
| 6 | M5 평가 문서·실행 패키지 | 작성·검사 완료 | README 3단 구성, Q&A, 발표 개요, 소스 ZIP |
| 7 | 데이터 개인정보·이용 조건 검수 | 부분 완료 | 배포 조건·불용어 근거·자동 검사·공개 생성 경로 확인, 전체 인명 검수는 이슈 #8 |
| 8 | 작업 브랜치 커밋·푸시·PR | 완료 | 구현 커밋 e3ff7e8, 원격 푸시, main 대상 PR #9 |

실제 명령과 환경은 [검증 기록](artifacts/verification_results.json)에 있고, 상세 완료 체크는 [실행 체크표](docs/repo_readme/docs_03_verification_and_checklists.md)에서 관리합니다. Git 관리 검사를 포함한 전체 회귀 검사 59개와 요청된 네 CLI 명령을 통과했습니다.

현재 환경은 Python 3.14.7과 고정된 설치 패키지 조합입니다. 제공 소스의 구버전 의존성 범위를 현재 환경에 맞추고, 누락 import·초기화 파일명·실패 종료 코드·보너스 검증을 보완했습니다. 공통 analyzer의 실제 진입점은 `TextPreprocessor.tokenize`입니다.

후속 브랜치 `feat/a4-1-data-review`는 구현 수정 전에 생성하고 첫 커밋 8979244에 실행 계획을 기록했습니다. 정확한 Figshare 다운로드 항목의 CC BY 4.0 표기와 고정 불용어 채택 근거를 확인했습니다. 자동 검사와 공개 산출물 생성 경로를 보완했으며, 전체 코퍼스의 인명·서명 검수는 이슈 #8에서 추적합니다. [후속 실행 계획](docs/A4_1_data_review_plan.md), [출처 보고서](docs/A4_1_data_sources.md), [데이터 검수 이슈 #8](https://github.com/ntt65/A4_1/issues/8)

후속 구현 커밋 78c399c를 원격에 푸시하고 기존 구현 브랜치를 기준으로 PR을 생성했습니다. [후속 PR #10](https://github.com/ntt65/A4_1/pull/10)

구현과 산출물의 검토는 PR에서 진행합니다. [구현 커밋 e3ff7e8](https://github.com/ntt65/A4_1/commit/e3ff7e8caf528324e099eb580665c9a5756cb53a), [PR #9](https://github.com/ntt65/A4_1/pull/9)

---

### User Review & Critical Decisions

> [!IMPORTANT]
> 다음 8가지 기준과 하단 상세 체크리스트는 최초 설계 명세입니다. 실제 완료 상태는 위 실행 현황과 연결된 실행 체크표에 기록합니다. 개인정보·데이터 이용 조건의 남은 검수는 이슈 #8에서 추적합니다.

1. **데이터 누수(Data Leakage) 엄격 차단**:
   - 어휘 사전(Vocabulary)과 IDF 통계량은 **오직 Train 데이터(80%)로만 `fit`** 합니다. Test 데이터(20%)는 오직 `transform`만 수행합니다.
   - 8:2 단일 분할(`random_state=42`)을 초기에 1회 고정 수행하고, 모든 비교 실험(TF 변형, 분류 모델, 검증)에서 동일 분할을 재사용합니다.
2. **Scikit-learn 1:1 오차 $10^{-6}$ 미만 검증 프로토콜**:
   - `smooth_idf=True`, `norm='l2'`, `sublinear_tf=False` 일치화.
   - 토큰화 정규식(`r'(?u)\b\w\w+\b'`, 1글자 단어 제외) 및 소문자화, 단어 사전 알파벳 오름차순 정렬을 Sklearn과 완전 일치화.
   - 소문자화·토큰화·불용어 제거를 수행하는 공통 `preprocess_pipeline`을 양쪽 구현에서 사용합니다. 불용어 목록은 `resources/stopwords_en.txt`에 고정하고 출처·버전을 기록합니다.
   - `float64` 정밀도를 강제하여 TF(Count), DF, IDF, L2 정규화 행렬을 단계별로 assertion 검증.
3. **데이터 윤리 및 저작권 준수 (과제 원문 5쪽)**:
   - 데이터셋 원본 배포처의 이용·재배포 조건, 근거 URL, 확인일을 기록합니다. Scikit-learn 코드의 라이선스를 데이터셋에 그대로 적용하거나, 근거 없이 BSD로 단정하지 않습니다.
   - 헤더·푸터·인용문 제거와 별도로 이메일·전화번호 등 자동 검사 및 이름·본문·서명 수동 검수를 수행합니다. 식별 가능한 개인정보가 있는 문서는 비식별화하거나 제외하고 재검사합니다. 확인되지 않은 사항은 미확인으로 기록하며, 과제의 데이터 사용 조건을 충족한 자료만 사용합니다.
4. **수치 안정성 및 Zero-Division 방어**:
   - **Double Normalization**: $\text{tf} > 0$인 원소에만 $0.5 + 0.5 \times \frac{\text{tf}}{\max(\text{tf})}$를 적용하고, $\text{tf} = 0$은 0으로 보존.
   - **영벡터 방어**: OOV(사전 미등재) 단어만 있는 쿼리나 빈 문서의 L2 Norm이 0일 경우, 나누기 에러를 회피하고 영벡터를 반환.
5. **검색 시간 복잡도 표기 정정 및 벤치마크 기준**:
   - Dense 행렬 연산 $D \cdot q$는 **$O(N \cdot |V|)$** 입니다. 양수 점수 문서만 반환하며, Top-K 경계 동점까지 점수 내림차순·문서 ID 오름차순을 적용합니다. 부분 선택과 선택된 결과 정렬 비용은 $O(N + K \log K)$입니다.
   - 역색인 벤치마크는 **인덱스 구축 시간**과 **질의 실행 시간**을 분리합니다. 전체 스캔과 역색인에 동일한 0점 제외·동점 정책을 적용하고, 점수 허용오차와 문서 ID 순서를 각각 검증합니다.
6. **데이터 로더 및 재현성 강화**:
   - `fetch_20newsgroups`에 `remove=('headers', 'footers', 'quotes')`를 적용하여 메타데이터에 의한 분류 편향을 줄입니다. 이 옵션을 개인정보 제거의 완료 기준으로 사용하지 않습니다. 로컬 캐시 경로와 데이터 선택 설정을 명시하고 문서 수·클래스 분포를 기록합니다.
7. **BM25 수식 및 검증 기준 구체화**:
   - 음수 IDF 방지형 Robertson-Spärck Jones IDF 및 $k_1=1.5, b=0.75$, 문서 길이 페널티 수식 채택. Toy 예제(문서 3건) 수작업 계산 검증 추가.
8. **파이프라인 작업 순서 재정렬**:
   - **M1 (데이터 & 전처리) → M2 (NumPy TF-IDF & 오차 검증) → M3 (검색 시스템) & M4 (분류 모델) → M6 (보너스: TF변형, BM25, 역색인) → M5 (최종 3단 리포트 & 패키징)**.

---

### 1. Overview & Core Concept

- **목적**: TF-IDF·코사인 유사도·TF 변형·BM25의 수식을 NumPy로 직접 구현하고 역색인 구조를 구축합니다. Scikit-learn의 데이터 로더, `LogisticRegression`, 평가 지표 및 검증용 벡터라이저를 사용하여 전체 파이프라인을 구성합니다.
- **핵심 목표**: Scikit-learn의 `TfidfVectorizer` 결과와 최대 절대 오차 $10^{-6}$ 미만인지 검증하고, Bag-of-Words(BoW)의 구조적 한계(어순 무시, 문맥 소실, 동음이의어/다의어)를 정량적 지표와 5건 이상의 실제 오분류 사례를 근거로 분석합니다.
- **엔지니어링 철학**: 철저한 데이터 누수(Data Leakage) 방지, 수치적 안정성(Zero-Division Guard), 정확한 계산 복잡도 분석.

---

### 2. User Experience & CLI Interface Design

- **인덱스 빌드 및 저장 / 로드 아키텍처**:
  - `python main.py --build-index`: Train 셋을 기반으로 어휘 사전, IDF 벡터, 문서 정규화 행렬을 생성하여 `artifacts/index.pkl`로 직렬화 저장.
  - `SearchEngine.load()`: 기구축된 인덱스를 로드하여 재학습 없이 초고속 질의 응답 제공.
- **CLI 실행 인터페이스 (과제의 선택 가능한 안 A, 본 프로젝트에서 구현)**:
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
- **함수형 인터페이스 (과제의 선택 가능한 안 B, 본 프로젝트에서 함께 구현)**:
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

#### B. Scikit-learn 동치 수식 및 검증 조건 ($10^{-6}$ 미만)
- **토큰화**: Sklearn 기본 정규식 `r'(?u)\b\w\w+\b'` 준수 (한 글자 단어 제외, 소문자화).
- **불용어 및 공통 입력**: `resources/stopwords_en.txt`에 소문자·중복 없는 불용어 목록을 고정하고 출처·버전을 README에 기록합니다. `preprocess_pipeline(text: str) -> list[str]` 내부에서 소문자화, 위 정규식 토큰화, 고정 목록 기반 불용어 제거를 수행합니다. 동의어 치환은 적용하지 않으며, 그 선택 이유를 README에 설명합니다.
- **검증용 참조 설정**: Custom 구현과 Scikit-learn에 같은 원문과 동일한 `preprocess_pipeline`을 적용합니다. 참조 구현은 아래 callable analyzer를 사용하므로, 불용어 제거를 포함한 전처리는 analyzer 안에서 완료합니다. Scikit-learn의 기본 불용어 설정에 처리를 맡기지 않습니다.
  `TfidfVectorizer(analyzer=preprocess_pipeline, stop_words=None, dtype=np.float64, smooth_idf=True, norm='l2', sublinear_tf=False)`
- **어휘 정렬**: 사전의 모든 키를 알파벳 오름차순(`sorted(vocab.keys())`)으로 0부터 정수 인덱싱.
- **검증 입력과 순서**: Train으로 각각 학습한 두 구현에 대해 토큰열, 어휘 집합·열 순서, Shape, TF, DF, IDF를 확인한 뒤 Train·Test·쿼리의 최종 벡터를 비교합니다. 빈 문서, 불용어만 있는 문장, OOV만 있는 쿼리도 포함합니다.
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
($k_1 = 1.5$, $b = 0.75$; $N$과 $\text{df}$는 Train 기준이며, $f_{t,D}$는 정규화하지 않은 단어 출현 횟수입니다. $|D|$는 공통 전처리와 불용어 제거 후 토큰 수, $\text{avgdl}$은 같은 기준의 Train 문서 평균 길이입니다. $Q$는 사전에 있는 고유 쿼리 용어의 집합으로, 반복 쿼리어는 한 번만 합산합니다.)

빈 문서의 점수는 0으로 정의합니다. Train 데이터가 없거나 전처리 후 전체 어휘가 비어 있으면 인덱스 구축을 오류로 종료하고 입력을 점검하도록 안내합니다.

#### E. Top-K 반환 및 전체 스캔·역색인 동치성
- **반환 정책**: 점수가 0보다 큰 문서만 반환합니다. 양수 점수 문서가 $K$건보다 적으면 실제 건수만 반환하며, 빈 쿼리·OOV 쿼리·검색 결과 없음은 함수에서 `[]`로 반환하고 CLI에서 안내합니다.
- **입력 및 정렬**: `top_k`는 양의 정수여야 하며, 그 외에는 `ValueError`로 처리합니다. 양수 점수 문서 수를 $M$, 실제 반환 수를 $K'=\min(K,M)$로 정합니다. 결과 순서는 **점수 내림차순 → 동일 점수에서 `doc_id` 오름차순**입니다.
- **경계 동점 처리**: $M=0$이면 즉시 `[]`를 반환합니다. 그 외에는 `np.argpartition`으로 $K'$번째 점수 $\tau$를 구하고, $\text{score}>\tau$인 문서를 먼저 선택합니다. $\text{score}=\tau$인 문서에서는 부족한 개수만큼 작은 `doc_id`를 부분 선택한 뒤, 선택된 $K'$건만 위 정렬 규칙으로 정렬합니다. $K'=M$이면 전체 양수 후보를 정렬하며, 동점 후보 전체를 별도 정렬하여 비용을 늘리지 않습니다.
- **점수와 순위 검증**: 전체 스캔과 역색인은 같은 코퍼스·쿼리·점수식·`float64`·쿼리 용어 순서 및 합산 방식을 사용합니다. 점수는 같은 `doc_id`끼리 `np.allclose(rtol=0, atol=1e-12)`로 비교하고, 반환 문서 ID 순서는 별도로 완전히 일치해야 합니다. 점수 허용오차를 동점 판정 기준으로 사용하지 않습니다.
- **필수 비교 예제**: 일반 쿼리, 빈/OOV 쿼리, 결과가 $K$건 미만인 경우, $K$가 전체 문서 수보다 큰 경우, $K$ 경계의 동점 및 잘못된 `top_k` 입력을 검증합니다.

---

### 4. Technical Architecture & System Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│               Data Acquisition & Leakage Prevention (M1)              │
│        20 Newsgroups (4 Categories, ~2,500 docs, remove headers/footers)│
│        ─── Fixed 8:2 Split ───► Train Set (80%) | Test Set (20%)      │
│        Gates: Dataset Usage Terms Review & PII Inspection             │
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
│• Top-K + Ties  │      │• Accuracy & F1   │     │• BM25 Engine      │
│  O(N+K log K)  │      │• Confusion Matrix│     │• Inverted Index   │
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

GitHub 저장소(`https://github.com/ntt65/A4_1/`)에 이미 등록한 이슈를 기준으로 번호를 유지합니다. **#5는 Documentation, #6은 Bonus**이며, 실행 순서는 이슈 번호와 별개로 **#1 → #2 → #3·#4 → #6 → #5**입니다. 아래는 기존 이슈에 반영할 개정 명세입니다.

### [Issue #1] [Data] 20 Newsgroups 수집, 전처리 및 데이터 누수 방지 분할
- **Labels**: `enhancement`, `phase-1`, `data-pipeline`
- **Milestone**: `M1 - Dataset & Preprocessing`
- **체크리스트**:
  - [ ] `fetch_20newsgroups` 4개 카테고리(`sci.space`, `comp.graphics`, `rec.sport.baseball`, `talk.politics.mideast`) 다운로드 및 로컬 캐싱 설정
  - [ ] 메타데이터에 의한 분류 편향을 줄이기 위한 `remove=('headers', 'footers', 'quotes')` 옵션 적용
  - [ ] 실제 로드된 문서 수(총 ~2,500건) 및 카테고리별 분포 로깅
  - [ ] 데이터셋 원본 배포처의 이용·재배포 조건, 근거 URL, 확인일을 기록하고 Scikit-learn 코드 라이선스와 구분
  - [ ] 개인정보 검사: 이메일·전화번호 등 자동 검사와 이름·본문·서명 수동 검수를 수행하고, 대상 문서를 비식별화하거나 제외한 뒤 재검사
  - [ ] **데이터 누수 방지**: Train/Test 8:2 단일 분할(`random_state=42`) 고정 및 결측치 제거
  - [ ] Sklearn 호환 텍스트 정제 및 토큰화 정규식(`r'(?u)\b\w\w+\b'`) 구현 (1글자 단어 제외, 소문자화)
  - [ ] `resources/stopwords_en.txt`에 출처·버전이 명확한 불용어 목록을 고정하고 공통 `preprocess_pipeline`에서 제거
- **수락 기준 (Acceptance Criteria)**:
  - [ ] Train 80% / Test 20% 분할이 고정 저장되어 이후 모든 모듈에서 동일 데이터셋을 참조할 것
  - [ ] 원본 이용 조건의 근거와 개인정보 검사·처리 결과가 문서에 기록되고, 사용 자료가 과제의 데이터 사용 조건을 충족할 것 (미확인 상태를 완료로 표시하지 않음)
  - [ ] Custom 및 sklearn 참조 구현에서 동일한 전처리·불용어 목록을 사용하여 같은 토큰열을 생성할 것

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
    - [ ] 공통 `preprocess_pipeline`을 callable analyzer로 전달하고 고정 불용어 목록·토큰열 일치 확인
    - [ ] Vocabulary 일치 및 Shape 일치 확인
    - [ ] DF 및 IDF 벡터 오차 검증
    - [ ] 최종 L2 정규화 TF-IDF 행렬 최대 오차 $\max |X_{\text{custom}} - X_{\text{sklearn}}| < 1.0 \times 10^{-6}$ 검증
    - [ ] 평균 오차(Mean Absolute Error) 로깅
- **수락 기준 (Acceptance Criteria)**:
  - [ ] Sklearn과의 최대 오차가 $1.0 \times 10^{-6}$ 미만으로 판정 통과(PASS) 로그 출력
  - [ ] Test 문장에 대해 신규 어휘가 배제된 상태로 안정적 transform 수행 확인
  - [ ] Train·Test·쿼리 및 빈 문서·불용어/OOV 전용 입력에 대한 오차·영벡터 검증 통과

### [Issue #3] [Search] 코사인 유사도 검색 엔진, 인덱스 직렬화 및 CLI 구축
- **Labels**: `feature`, `phase-3`, `cli`
- **Milestone**: `M3 - Search Engine System`
- **체크리스트**:
  - [ ] 인덱스 영속화 파이프라인: Train 데이터 기반 인덱스 빌드 및 `artifacts/index.pkl` 저장/로드(`SearchEngine.load()`)
  - [ ] 쿼리 벡터 변환: Train 단어 사전 및 IDF를 이용한 L2 정규화 쿼리 벡터 생성 (Zero-vector 방어)
  - [ ] 코사인 유사도 벡터 연산: 정규화된 문서 행렬 $D$와 쿼리 벡터 $q$ 간 행렬 곱 ($O(N \cdot |V|)$ dense 연산)
  - [ ] Top-K 검색 가속: E절 규칙에 따라 `np.argpartition`과 경계 동점 처리를 적용하여 $O(N + K \log K)$로 최대 `top_k`건 반환
  - [ ] 양수 점수만 반환하고, 점수 내림차순·동점 시 `doc_id` 오름차순 및 `top_k` 입력 검증을 공통 결과 선택 함수에 구현
  - [ ] 검색 결과 포맷터: 유사도 스코어(소수점 4자리), 문서 ID, 150자 스니펫 출력
  - [ ] 과제 필수 인터페이스 구현:
    - [ ] **(안 A) CLI**: `python main.py --query "..." --topk 5`
    - [ ] **(안 B) 함수 시그니처**: `search(query: str, top_k: int = 5) -> List[Tuple[float, int, str]]`
- **수락 기준 (Acceptance Criteria)**:
  - [ ] 빈/OOV 쿼리는 함수에서 `[]`, CLI에서 검색 결과 없음 안내를 반환
  - [ ] $K$ 경계 동점, 결과 부족, `top_k`가 문서 수보다 큰 경우 및 잘못된 입력의 결과가 E절 규칙과 일치
  - [ ] CLI 및 함수 반환 형식이 명시된 인터페이스와 일치할 것

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

### [Issue #6] [Bonus] TF 변형 비교, BM25 엔진 및 역색인(Inverted Index)
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
    - [ ] **동치성 검증**: E절의 공통 결과 선택 함수를 사용하고, 전체 스캔과 역색인의 문서별 점수를 `np.allclose(rtol=0, atol=1e-12)`로 비교하며 반환 문서 ID 순서는 완전히 일치하는지 검증
    - [ ] **벤치마크 분리**: 인덱스 구축 시간과 단일 쿼리 실행 시간(ms)을 분리 측정하여 전체 스캔 대비 속도 비교
- **수락 기준 (Acceptance Criteria)**:
  - [ ] BM25 Toy example 단위 테스트 통과 및 E절의 일반·동점·0점·결과 부족 예제에서 전체 스캔/역색인 점수 허용오차와 문서 ID 순서 검증 통과
  - [ ] TF 변형 비교 및 BM25 비교 분석 결과 리포트 수록

### [Issue #5] [Documentation] 필수 3단 구성 README.md 및 제출 패키징
- **Labels**: `documentation`, `phase-5`
- **Milestone**: `M5 - Final Report & Packaging`
- **체크리스트**:
  - [ ] `README.md` 과제 규격 필수 3단 구성 작성:
    - [ ] **1단: 구현 요약**: 전처리 파이프라인, TF-IDF 수학 공식, 검색/분류/보너스 아키텍처
    - [ ] **2단: 검증/실험 결과**: TF-IDF 검증 오차 로그, 분류 성능표, 혼동 행렬 시각화, BM25/역색인 실험표
    - [ ] **3단: 한계와 개선 방향**: BoW 한계 분석, 5개 오분류 원인, 사전학습 언어모델(Transformer) 후속 학습 방향
  - [ ] 실행 방법 안내 (CLI 커맨드 예시 및 `search()` 함수 호출 예시)
  - [ ] NumPy 직접 구현 범위와 Scikit-learn 사용 범위, 고정 불용어 출처·버전, 동의어 미적용 근거, 데이터 이용 조건·개인정보 검사 결과 및 검색 동점·0점 정책 명시
  - [ ] Python 3.10+ 구동용 의존성 명시 (`requirements.txt`)
- **수락 기준 (Acceptance Criteria)**:
  - [ ] 과제 PDF 가이드라인의 모든 필수 문항 및 결과 양식이 누락 없이 완비될 것
  - [ ] 새 가상환경에서 `pip install -r requirements.txt` 후 재현 실행이 완벽할 것
