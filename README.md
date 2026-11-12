# A4_1 문서 검색 및 분류 시스템

NumPy로 TF-IDF와 코사인 검색을 구현하고, 20 Newsgroups 네 카테고리의 문서를 Logistic Regression으로 분류합니다. 전처리부터 검증·검색·분류·BoW 한계 분석까지 CLI, Python 함수, 단일 Jupyter 노트북으로 재현할 수 있습니다.

2026-11-12 실측은 **TF-IDF 최대 오차 2.3315e-15 · Accuracy 92.4802% · Macro-F1 0.925329**입니다. 회귀 검사 59개와 노트북의 새 커널 전체 실행을 통과했습니다. 아래 수치는 실행 근거이며 평가자의 공식 PASS/FAIL 판정과 구분합니다.

| 목적 | 문서·산출물 |
| --- | --- |
| 과제 요구사항과 평가 원문 | [미션 PDF](docs/A4_1_mission.pdf), [평가 PDF](docs/A4_1_eval.pdf) |
| 개념 학습과 구술 대비 | [미션 Q&A: Q1~Q5](docs/A4_1_mission_QnA.md) |
| 평가 질문별 답변과 확인 근거 | [평가 Q&A: 16개 질문·보너스](docs/A4_1_eval_QnA.md) |
| 한 파일에서 처음부터 실행 | [A4_1_mission.ipynb](A4_1_mission.ipynb) |
| 발표와 완료 상태 확인 | [발표 개요](docs/A4_1_presentation_outline.md), [실행 체크표](docs/repo_readme/docs_03_verification_and_checklists.md) |
| 다운로드용 소스와 실측 근거 | [소스 ZIP](A4_1_complete_project_source.zip), [명령별 실행 기록](artifacts/verification_results.json), [노트북 실행 기록](artifacts/notebook_verification.json) |

## 1. 구현 요약

### Windows PowerShell 빠른 실행

프로젝트 루트에서 실행합니다. 기존 `.venv`가 있으면 그대로 사용하며, 아래처럼 가상환경의 Python 경로를 직접 지정하면 활성화 명령 없이 실행할 수 있습니다.

```powershell
cd D:\git\A4_1
if (-not (Test-Path -LiteralPath .venv/Scripts/python.exe)) { uv venv --python 3.14 }
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

.\.venv\Scripts\python.exe verify_tfidf.py
.\.venv\Scripts\python.exe main.py --query "space satellite launch orbit" --topk 5
.\.venv\Scripts\python.exe main.py --train-classifier
.\.venv\Scripts\python.exe main.py --bonus
```

최초 데이터 다운로드에는 네트워크가 필요하며 이후에는 `artifacts/data_cache`를 재사용합니다. Train dense TF-IDF 행렬만 약 770 MiB이므로 실험 시 여유 메모리를 확보합니다. 검증된 환경·설정과 각 실행 방식은 아래에 설명합니다.

### 실행 방식 C: 단일 Jupyter 노트북

[A4_1_mission.ipynb](A4_1_mission.ipynb)에 전처리·NumPy TF-IDF·검색·분류·실제 오분류 5건·Q1~Q5·별도 보너스를 모두 담았습니다. 구현과 불용어 목록이 파일 안에 있으며 저장소의 `src`·기존 지표·이미지 없이 실행합니다. 새 폴더에 이 파일 하나만 복사한 뒤 의존성과 최초 데이터 다운로드를 준비해 사용할 수 있습니다.

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-notebook.txt
.\.venv\Scripts\python.exe -m jupyterlab A4_1_mission.ipynb
```

VS Code에서는 노트북을 열고 `.venv`의 Python 커널을 선택한 뒤 **Restart Kernel → Run All**을 실행합니다. 혼동 행렬은 저장 출력에 내장했습니다. 노트북 파일 하나만 있는 빈 임시 폴더와 새 커널에서 28개 셀(코드 14개)이 통과했고 최종 전체 실행은 47.323초였습니다. 검증 실행에서는 기존 sklearn 데이터 캐시만 `A4_1_DATA_HOME`으로 지정했습니다. 의존성·캐시·메모리 조건은 첫 셀에 있습니다. [개요·실행 계획](docs/A4_1_notebook_plan.md), [실제 실행 결과](artifacts/notebook_verification.json)

CLI는 `artifacts/index.pkl`에 인덱스를 저장합니다. 노트북은 독립 구현에서 생성한 `artifacts/notebook/search_index/`의 JSON·NPY를 사용하고 재로딩 시 읽기 전용 mmap을 적용합니다. 두 경로의 파일 형식은 서로 다르며 각 실행 경로에서 생성·재사용합니다.

저장된 출력과 실행 기록의 일치를 검사하거나 새 커널 전체 실행을 재현합니다.

```powershell
.\.venv\Scripts\python.exe scripts/verify_notebook.py
.\.venv\Scripts\python.exe scripts/verify_notebook.py --execute
```

### 데이터와 전처리

`comp.graphics`, `rec.sport.baseball`, `sci.space`, `talk.politics.mideast`의 원본 3,894건을 사용했습니다. `headers`, `footers`, `quotes`를 제거하고 이메일과 일반적인 전화번호 형식을 마스킹했습니다. 공백만 남은 107건을 제외해 3,787건을 확보했습니다. `train_test_split(test_size=0.2, random_state=42, stratify=labels)`로 Train 3,029건, Test 758건을 고정했습니다. 이번 분할에서 Train/Test 사이 완전히 동일한 본문은 0건입니다.

| 카테고리 | Train | Test |
| --- | --- | --- |
| comp.graphics | 764 | 191 |
| rec.sport.baseball | 766 | 192 |
| sci.space | 764 | 191 |
| talk.politics.mideast | 735 | 184 |

소문자화 후 `r'(?u)\b\w\w+\b'`로 두 글자 이상의 Unicode 단어 토큰을 추출합니다. 숫자와 밑줄도 단어 문자에 포함됩니다. 한 글자 `C`와 `C++` 같은 용어의 손실은 현재 규칙의 한계입니다. 동의어 치환은 적용하지 않습니다.

고정 불용어 123개는 [stopwords_en.txt](resources/stopwords_en.txt)에 있습니다. 사용자 제공 소스의 목록을 프로젝트 기준 `a4-1-en-v1`로 채택하고 기능어 제거의 선정 근거와 UTF-8·LF 정규화 해시를 기록했습니다. 외부 목록의 원출처·버전은 제공되지 않았습니다. `no`, `nor`, `not`도 포함돼 부정 표현이 손실될 수 있습니다. 파일 경로는 모듈 위치를 기준으로 합니다. [목록 선정 근거와 해시](docs/A4_1_data_sources.md)

어휘 33,297개와 IDF는 Train에서만 학습합니다. Test와 질의에는 학습된 어휘·IDF를 그대로 사용하고 OOV는 제외합니다. 데이터 분포와 코퍼스 SHA-256은 [데이터 메타데이터](artifacts/dataset_metadata.json)에 기록했습니다.

벡터화 결과의 행은 문서, 열은 학습 어휘의 단어입니다. Train shape는 `(3029, 33297)`, Test shape는 `(758, 33297)`, 한 질의의 shape는 `(1, 33297)`입니다. Test에만 있는 단어로 어휘를 확장하거나 IDF를 다시 학습하지 않습니다.

### TF-IDF와 코사인 검색

```mermaid
flowchart LR
    A["원본문서"] --> B["전처리 · 고정 8:2 분할"]
    B -->|Train| C["어휘 · IDF 학습"]
    C --> D["고정된 어휘 · IDF로 변환"]
    B -->|Test: 변환만| D
    Q["질의: 변환만"] --> D
    D --> E["코사인 검색"]
    D --> F["분류: Train 학습 · Test 평가"]
    D --> G["sklearn 참조 검증"]
```

문서 수를 $N$, 단어 출현 횟수를 $c_{d,t}$라 두면 다음과 같습니다.

$$tf(d,t)=c_{d,t},\quad df(t)=\sum_d\mathbf{1}[c_{d,t}>0]$$
$$idf(t)=\ln\left(\frac{1+N}{1+df(t)}\right)+1,\quad x_{d,t}=tf(d,t)idf(t)$$
$$\hat{x}_d=x_d/\|x_d\|_2\quad(\|x_d\|_2>0)$$

영벡터는 0으로 유지합니다. 어휘 열을 알파벳 순으로 정렬하고 `float64`로 계산합니다. 핵심 계산은 [tfidf.py](src/tfidf.py)에 있습니다. sklearn은 데이터 로딩·검증·분류에 사용하며, 직접 구현한 TF-IDF와 코사인 점수 계산은 NumPy를 사용합니다.

$$\cos(d,q)=\frac{d\cdot q}{\|d\|_2\|q\|_2}=\hat{d}\cdot\hat{q}$$

[search_engine.py](src/search_engine.py)는 정규화된 문서 행렬과 질의 벡터를 곱합니다. 점수가 양수인 문서만 반환하고 점수 내림차순, 동점 시 문서 ID 오름차순을 적용합니다. `np.partition`으로 Top-K 경계를 선택한 뒤 경계 동점을 포함해 정렬합니다. Dense 점수 계산은 $O(N|V|)$이고, 경계 동점이 많으면 후보 정렬 비용이 커집니다. 빈 입력과 OOV 질의는 빈 리스트를 반환합니다.

### 모듈과 실행 환경

| 모듈 | 역할 |
| --- | --- |
| src/dataset.py | 다운로드·캐시·마스킹·고정 분할 |
| src/preprocess.py | 불용어 로딩·소문자화·토큰화 |
| src/tfidf.py | Train 어휘·DF·IDF와 NumPy 변환 |
| src/search_engine.py | 코사인 점수·Top-K·인덱스 저장·로드 |
| src/classifier.py | Logistic Regression·지표·혼동 행렬·오분류 근거 |
| verify_tfidf.py | 어휘·TF·DF·IDF·Train/Test/질의 대조 |
| main.py | CLI와 `search()` 진입점 |
| A4_1_mission.ipynb | 불용어·핵심 구현·실험·Q&A·보너스를 내장한 독립 실행 노트북 |

전처리와 벡터화를 분리해 검색·분류에서 같은 표현을 재사용합니다. 검색 전략과 분류 모델을 교체할 때 전처리 규칙을 유지하고 단계별 오류를 확인할 수 있습니다. 분류에는 `LogisticRegression(max_iter=1000, random_state=42)`를 사용합니다. NumPy 결과를 CSR로 변환해 분류기에 전달하고 BLAS 스레드는 2개로 제한했습니다. 이번 학습은 27회 반복으로 종료됐습니다.

검증 환경은 Windows, Python 3.14.7, uv 0.13.7, NumPy 2.5.4, scikit-learn 1.9.1입니다. 의존성은 [requirements.txt](requirements.txt)에 고정했습니다. 첨부 소스의 NumPy `<2`, sklearn `<1.4` 범위는 현재 Python과 맞지 않아 설치된 조합을 사용했습니다. 미션은 Python 3.10 이상을 요구하며 현재 패키지 조합은 Python 3.14.7에서 검증했습니다.

### 실행 방식 A: CLI

상단의 PowerShell 명령은 가상환경 Python을 직접 지정합니다. `python`이라는 짧은 명령을 사용하려면 먼저 해당 가상환경을 활성화합니다. CMD의 활성화 명령은 `.venv\Scripts\activate.bat`입니다.

```powershell
.\.venv\Scripts\Activate.ps1
```

활성화한 터미널에서는 다음 명령을 그대로 사용할 수 있습니다.

```bash
python verify_tfidf.py
python main.py --query "space satellite launch orbit" --topk 5
python main.py --train-classifier
python main.py --bonus
python -m unittest discover -s tests -v
```

인덱스를 저장하면 다음 검색부터 학습 없이 로드합니다. 데이터를 변경하면 인덱스를 다시 생성해야 합니다.

```bash
python main.py --build-index
python main.py --query "computer graphics rendering image" --topk 5
```

### 실행 방식 B: Python 함수

프로젝트 루트에서 가상환경 Python 또는 같은 환경의 커널로 호출합니다. `search(query: str, top_k: int = 5) -> List[Tuple[float, int, str]]`는 점수·Train 문서 ID·최대 150자 스니펫을 반환합니다.

```python
from main import search

# search(query: str, top_k: int = 5) -> List[Tuple[float, int, str]]
for score, doc_id, text_snippet in search("space satellite launch orbit", top_k=5):
    print(f"{score:.4f} {doc_id} {text_snippet}")
```

`top_k`는 양의 정수여야 합니다. 빈 문자열·불용어만 있는 질의·OOV 질의는 `[]`를 반환합니다. 최초 함수 호출은 저장 인덱스가 있으면 로드하고, 없으면 Train에서 시스템을 구성합니다. 전처리·데이터·어휘 정책을 바꾸면 `--build-index`로 인덱스를 다시 생성합니다.

구현·Git 관리 [PR #12](https://github.com/ntt65/A4_1/pull/12)와 노트북 [PR #14](https://github.com/ntt65/A4_1/pull/14)는 모두 `main`에 병합됐습니다. 병합 기준 커밋은 `744cfc7`입니다. 커밋·이슈 관리 절차와 과거 변경 근거는 [Git 작업 절차](docs/GIT_WORKFLOW.md), [공개 이력 추적표](docs/git_history_map.md)에 있습니다.

## 2. 검증 및 실험 결과

### TF-IDF 수치 검증

sklearn 참조 설정은 공통 `TextPreprocessor.tokenize` analyzer, `use_idf=True`, `smooth_idf=True`, `sublinear_tf=False`, `norm='l2'`, `dtype=np.float64`입니다. callable analyzer가 소문자화·토큰화·불용어 제거를 모두 수행하므로 별도의 기본 tokenizer를 혼용하지 않습니다. 어휘와 열 순서, TF 및 DF를 동일성 비교하고 IDF와 최종 벡터를 대조했습니다.

| 입력 | Shape | 최대 절대 오차 | 평균 절대 오차 |
| --- | --- | --- | --- |
| Train | 3,029 × 33,297 | 2.3315e-15 | 3.0568e-20 |
| Test | 758 × 33,297 | 8.3267e-16 | 2.2134e-20 |
| 질의 5개 | 5 × 33,297 | 0 | 0 |

IDF 오차는 0입니다. 미션의 `1e-6` 허용 기준을 통과했습니다. 질의에는 빈 문자열, 불용어만 있는 문자열, OOV를 포함했습니다. 실패 시 검증 스크립트가 예외로 종료되도록 수정했습니다. [검증 결과 JSON](artifacts/tfidf_verification.json), [실행 출력](artifacts/run_logs/verify_tfidf.txt)

```text
train: shape=[3029, 33297] max_error=2.3315e-15 mean_error=3.0568e-20
test: shape=[758, 33297] max_error=8.3267e-16 mean_error=2.2134e-20
queries: shape=[5, 33297] max_error=0.0000e+00 mean_error=0.0000e+00
Vocabulary/TF/DF/IDF and Train/Test/query checks PASS; IDF error=0.0000e+00
```

### 검색 실행 결과

아래는 실제 출력에서 점수·문서 ID·스니펫의 일부를 발췌한 결과입니다. 문서 ID는 고정 Train 목록의 0 기반 인덱스입니다. [전체 검색 출력](artifacts/run_logs/search.txt)

```text
=== Search Results for 'space satellite launch orbit' (Top 5) ===
[1] Score: 0.4737 | Doc #1934: COMMERCIAL SPACE NEWS/SPACE TECHNOLOGY INVESTOR NUMBER 22
[2] Score: 0.3076 | Doc #2609: The Space Calendar is updated monthly
[3] Score: 0.2954 | Doc #755: [Regarding orbital billboards...]
[4] Score: 0.2932 | Doc #2145: Navstar GPS Constellation Status
[5] Score: 0.2701 | Doc #1725: Putting a satellite as high as possible
```

### 분류 지표와 혼동 행렬

Test 758건 중 701건을 맞혀 Accuracy **92.4802%**, Macro-F1 **0.925329**를 얻었습니다. 오분류는 57건입니다. [전체 분류 지표와 사례](artifacts/classifier_metrics.json), [실행 출력](artifacts/run_logs/classifier.txt)

Accuracy는 전체 정답 비율이고, Macro-F1은 각 클래스의 F1을 같은 비중으로 평균한 값입니다. 혼동 행렬은 **행이 실제 레이블, 열이 예측 레이블**이며 대각선 합이 701입니다.

| 클래스 | Precision | Recall | F1 | Test 건수 |
| --- | --- | --- | --- | --- |
| comp.graphics | 0.9471 | 0.9372 | 0.9421 | 191 |
| rec.sport.baseball | 0.9124 | 0.9219 | 0.9171 | 192 |
| sci.space | 0.8725 | 0.9319 | 0.9013 | 191 |
| talk.politics.mideast | 0.9766 | 0.9076 | 0.9408 | 184 |

![행은 실제 레이블, 열은 예측 레이블인 혼동 행렬](artifacts/confusion_matrix.png)

```text
Classification Accuracy: 92.4802% | Macro-F1: 0.925329
[[179   4   8   0]
 [  4 177  10   1]
 [  5   5 178   3]
 [  1   8   8 167]]
```

요청된 네 CLI 명령은 모두 종료 코드 0으로 완료했습니다. [명령별 검증 기록](artifacts/verification_results.json)에 실행 시각과 로그 경로를 기록했습니다. `tests/`의 회귀 검사는 빈 입력·OOV·Train 전용 학습·동점 경계·인덱스 저장/로드·BM25 수식·역색인 재구축 등을 확인합니다.

선행 문서의 정확도 97.56%, Macro-F1 0.9758 및 오차 1.1102e-15는 과거 제공 자료의 기록입니다. 이번 소스와 고정 데이터 처리에서 나온 결과는 위 표에 반영했습니다. 원본 코드에서 빈 문서 제거, 실행 오류 수정, 실제 보너스 검증 및 오분류 근거 수집을 보완했습니다.

### 검증 기록 확인과 회귀 검사

CLI 지표·시간은 `artifacts/*_metrics.json`과 명령별 로그, 독립 노트북 결과는 `artifacts/notebook_verification.json`에 각각 기록했습니다. 두 실행 경로의 재현 조건과 실행 시각을 구분해 확인합니다. 노트북 기본 검사는 저장된 셀 출력·실행 순서·내장 그림과 파일 SHA-256을 실행 보고서에 대조하며, `--execute`를 추가하면 새 커널에서 모든 셀을 다시 실행합니다.

전체 회귀 검사 59개를 재현하는 PowerShell 명령은 다음과 같습니다. BLAS 스레드 수를 2로 맞춰 검증했습니다.

```powershell
$env:OPENBLAS_NUM_THREADS = "2"
$env:OMP_NUM_THREADS = "2"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/verify_notebook.py
```

## 3. BoW 한계와 오분류 원인

### 실제 Test 오분류 5건

아래 ID는 Test 목록의 0 기반 인덱스입니다. 설명은 실제 본문과 선형 모델의 클래스 점수 차이를 근거로 한 원인 가설입니다. 기여도는 $x_t(w_{pred,t}-w_{true,t})$로 계산하며 양수이면 예측 클래스를 지지합니다. 전체 점수 차이는 기여도 합에 절편 차이를 더한 값입니다. 인명은 본문 발췌에서 생략했습니다.

| Test ID | 실제 → 예측 | 본문 근거 | 기여도 및 분석 |
| --- | --- | --- | --- |
| 3 | mideast → space | `palestinian police force`, `concrete proposals`, `idea` | `palestinian` -0.3580, `police` -0.1445가 실제 클래스를 지지했지만 `idea` +0.2409 등 일반 어휘와 절편 +0.2461이 합쳐져 space가 0.0123 앞섰습니다. 정치적 구문의 관계를 독립 단어 가중치만으로 충분히 표현하기 어렵다는 가설입니다. |
| 5 | graphics → space | `What is 9 x 6?`, `base 13 system` | `system` +0.1815, `base` +0.0935가 space를 지지했습니다. 한 글자 `9`, `x`, `6`은 정규식에서 사라집니다. 게시글 대화 맥락과 그래픽 주제 단서가 부족한 짧은 문서의 오류입니다. |
| 13 | graphics → baseball | `RS6000 compiler`, `COBOL`, `pascal` | `pascal` -0.0931, `compiler` -0.0895가 실제 클래스를 지지했지만 잔존 서명 토큰 +0.2122와 `think` +0.1741이 다른 클래스를 밀었습니다. 서명 제거의 불완전성과 이름·일반 단어의 우연한 상관관계가 드러납니다. |
| 19 | baseball → space | `SI`, `article`, `subscription`, `date` | `si` +0.1221, `date` +0.0700이 space를 지지했고 최종 점수 차이는 0.0233입니다. 약어의 의미와 기사 대상의 문맥을 unigram만으로 복원하기 어려운 사례입니다. |
| 25 | mideast → space | `What are you stupid?` | 불용어 제거 후 `stupid`만 남습니다. 이 단어의 기여도는 -0.1325지만 절편 차이 +0.2461이 커 최종 space 점수가 0.1136 앞섰습니다. 주제 정보가 거의 없는 짧은 문서에서 클래스 기본 경향에 의존합니다. |

### 표현 방식의 한계와 개선 실험

`dog bites man`과 `man bites dog`는 같은 빈도로 표현돼 주어·목적어의 차이를 잃습니다. `bank`와 `사과` 같은 다의어는 하나의 차원에 합쳐지고 `car`와 `automobile` 같은 동의어는 다른 차원에 나뉩니다. 위 실제 사례에서는 짧은 문서, 약어, 잔존 서명과 전처리 손실을 확인했습니다. 각 사례의 개선 여부는 추가 실험으로 검증해야 합니다.

후속 비교는 부정어 보존, 도메인 토큰 정규식, bigram, 문맥 임베딩 또는 Transformer 모델을 대상으로 합니다. 같은 분할에서 Accuracy·Macro-F1을 비교하고, 관련도 정답이 있는 평가 쿼리로 Precision@5와 nDCG@5를 측정해야 합니다. 현재 검색 결과의 주제 적합성과 순위 변화는 관찰했으며 검색 품질 정답셋은 구축 대기 상태입니다.

백만 문서·어휘 5만 개의 dense float64 행렬은 원소만 약 400GB가 필요합니다. 희소 저장·역색인·후보 가지치기·캐시를 비교하고 인덱스 구축 비용과 질의 지연 p50/p95를 나눠 측정하는 방향입니다.

### 데이터 출처와 공개 산출물

헤더·푸터·인용 제거와 이메일·일반 전화번호 마스킹을 적용했고 사용 문서 3,787건에서 해당 자동 검사 패턴은 0건입니다. 이 결과는 지정한 패턴의 검사 결과이며 전체 식별정보 검수 완료를 뜻하지는 않습니다. 원문 캐시·검토 문맥·생성 인덱스는 Git과 소스 ZIP에서 제외합니다. [자동 검사 보고서](artifacts/privacy_audit.json), [sklearn 데이터셋 설명](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset)

실제 sklearn 다운로드 파일과 일치하는 Figshare file ID `5975967`, 항목 `3829962` 버전 2에서 CC BY 4.0 표기를 확인했습니다. 출처 표기는 Nelson Liu, 2016, `scikit-learn 20newsgroups dataset 20news-bydate.tar.gz`, version 2, Figshare입니다. 네 카테고리 선택·메타데이터 제거·마스킹·빈 문서 제외·재분할·수치 변환의 변경 내용을 출처 보고서에 기록했습니다. [배포 기록 DOI](https://doi.org/10.6084/m9.figshare.3829962.v2), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), [출처와 이용 조건 보고서](docs/A4_1_data_sources.md)

분류·보너스 JSON 생성기는 스키마 버전 2에서 수치·레이블·문서 ID·기여도 값을 저장합니다. 분류 설명은 `feature_id`로 연결하며 본문·스니펫·원본 기여 단어를 저장하지 않습니다. 공개 검색 출력 파일은 점수와 문서 ID로 구성했고 로컬 CLI·함수는 미션 규격대로 스니펫을 반환합니다. README의 오분류 표에는 검토한 비식별 발췌와 기여 단어를 사용했습니다.

공개 산출물 검사와 패키지 생성은 다음 명령으로 재현합니다. `audit_data.py --check-public`은 공개 JSON·검색 로그·노트북 출력·실행 해시·ZIP을 검사하고, `package_project.py`는 지정한 소스·문서·검증 산출물로 ZIP을 생성합니다.

```bash
python audit_data.py --check-public
python package_project.py
```

작업 개요·단계별 계획·실행 체크표와 GitHub 전달 상태를 다음 문서에 연결했습니다. [개요와 아키텍처](docs/repo_readme/docs_01_system_architecture_and_plan.md), [상세 계획](IMPLEMENTATION_PLAN.md), [실행 체크표](docs/repo_readme/docs_03_verification_and_checklists.md), [진행 이력](docs/repo_readme/docs_04_development_log_timeline.md)

## 보너스 과제 구현 및 검증

보너스 전체 결과는 [측정 JSON](artifacts/bonus_metrics.json)과 [실행 출력](artifacts/run_logs/bonus.txt)에 있습니다.

### Raw·Log·Double TF 비교

Raw는 $tf$, Log는 $\ln(1+tf)$, Double은 양수 빈도에 $0.5+0.5tf/\max(tf)$를 적용합니다. 0 빈도와 빈 문서는 0으로 보존합니다. 같은 학습 어휘·IDF·분할·분류기를 사용했습니다.

| TF 방식 | Accuracy | Macro-F1 | Raw 대비 우주 질의 Top-5 집합 일치율 |
| --- | --- | --- | --- |
| Raw | 92.4802% | 0.925329 | 100% |
| Log | 92.4802% | 0.925139 | 60% |
| Double | 91.9525% | 0.919962 | 40% |

질의는 `space satellite launch orbit`입니다. 네 주제 질의 전체의 집합 일치율도 JSON에 기록했습니다. 이 일치율은 순위 변화의 측정값이며, 관련도 정답에 따른 검색 품질 평가는 별도로 필요합니다.

### BM25 수식과 수작업 검증

$$score(d,q)=\sum_{t\in U(q)}c_{q,t}\ln\left(1+\frac{N-df(t)+0.5}{df(t)+0.5}\right)\frac{tf(t,d)(k_1+1)}{tf(t,d)+k_1(1-b+b|d|/avgdl)}$$

$k_1=1.5$, $b=0.75$입니다. CLI 구현에서 $U(q)$는 고유 질의어 집합, $c_{q,t}$는 질의 내 빈도이므로 반복 단어는 반복 횟수만큼 기여합니다. 독립 노트북의 BM25는 고유 질의어를 한 번씩 가산하는 정책을 사용합니다. 아래 Toy와 네 주제 질의에는 반복 단어가 없어 이 차이의 영향을 받지 않습니다.

Toy corpus `[[cat, sat, sat], [cat, dog], [dog]]`와 질의 `[cat, sat]`에서 $N=3$, $avgdl=2$입니다. 수작업 기대값은 `[1.5908508972825368, 0.47000362924573563, 0]`, 최대 오차는 **2.2204e-16**으로 기준 `1e-6`을 통과했습니다.

같은 우주 질의에서 TF-IDF의 문서 ID 순서는 `[1934, 2609, 755, 2145, 1725]`, BM25는 `[755, 1934, 1725, 186, 3027]`입니다. BM25는 TF 포화와 문서 길이 보정을 사용해 순위와 후보 집합이 달라집니다. 점수는 각 방식의 척도로 해석하고, 품질 향상은 관련도 정답을 확보한 뒤 평가해야 합니다.

### 역색인 동치성과 시간

역색인은 단어 ID를 `(doc_id, normalized_tfidf_weight)` posting으로 연결합니다. 네 주제 질의와 빈 입력·OOV·반복 단어 질의까지 총 7개에서 dense 전체 스캔 대비 최대 점수 오차 **0**, Top-5 순위 일치율 **100%**를 확인했습니다.

CLI 측정 기록의 인덱스 구축은 **0.463321초**였습니다. 우주 질의의 점수 계산과 Top-5 선택은 dense **24.0746ms**, 역색인 **0.3283ms**였습니다. 토큰화·벡터화 시간은 제외했고 BLAS 2스레드, 워밍업 후 3회 중앙값입니다. 이 시간은 해당 환경·실행의 측정값이며 전체 질의 분포의 지연을 대표하지 않습니다.

Dense 점수 계산은 $O(N|V|)$이고, 이 구현의 역색인 점수 누적은 전체 점수 배열 초기화를 포함해 $O(N+P_Q)$입니다. $P_Q$는 질의어 posting의 총길이이며 이후 Top-K 선택·동점 정렬 비용이 추가됩니다. 자주 등장하는 단어는 후보가 많아집니다. 노트북은 TF-IDF뿐 아니라 BM25 전체 스캔·역색인도 각각 7개 질의에서 검증하며, 해당 실행 시간은 [노트북 실행 기록](artifacts/notebook_verification.json)의 `results.bonus`에서 확인할 수 있습니다.
