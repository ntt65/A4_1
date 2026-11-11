# A4_1 문서 검색 및 분류 시스템

20 Newsgroups 문서를 대상으로 NumPy TF-IDF와 코사인 검색을 구현하고, Logistic Regression으로 분류 성능을 평가하는 프로젝트입니다. 아래 알고리즘 설명은 현재 구현 계획에 근거합니다.

2026-11-12 로컬 점검에서 `main.py`, `verify_tfidf.py`, Python 알고리즘 모듈, 데이터, 혼동 행렬 파일의 부재를 확인했습니다. 현재 `src/`에는 React 화면 코드가 있습니다. Python 가상환경과 의존성은 준비됐으며, 알고리즘 실행과 성능 재검증은 코드 이관 후 진행해야 합니다.

## 1. 구현 요약

### 데이터와 전처리 설계

대상 카테고리는 `comp.graphics`, `rec.sport.baseball`, `sci.space`, `talk.politics.mideast`입니다. 계획은 `fetch_20newsgroups`로 데이터를 읽고 `remove=('headers', 'footers', 'quotes')`를 적용한 뒤 Train/Test를 8:2, `random_state=42`로 고정하는 방식입니다. 기존 문서는 총 3,792건, Train 3,033건, Test 759건을 보고합니다. 로컬 데이터와 분할 결과는 확인 대기 상태입니다.

소문자화 후 정규식 `r'(?u)\b\w\w+\b'`로 토큰을 추출합니다. 이 패턴은 Unicode 단어 문자 두 개 이상을 허용하며 숫자와 밑줄도 포함합니다. 구두점은 토큰 경계가 되고 한 글자 토큰은 제외됩니다. `C++`, `3D`, 하이픈 단어 등의 보존 여부는 검색 목적에 따라 검토해야 합니다. 동의어 치환은 계획에서 제외했습니다. 의미가 다른 표현을 한 단어로 합치는 오류를 피하고 참조 구현과 같은 입력을 사용하기 위한 선택입니다.

불용어는 `resources/stopwords_en.txt`에 고정하고 출처와 버전을 기록하는 설계입니다. 해당 파일은 현재 이관 대기 상태입니다. 특히 부정어를 제거하면 문장의 극성이 사라질 수 있어 목록 검토가 필요합니다. 어휘와 IDF는 Train에만 `fit`하고 Test와 쿼리에는 동일한 변환을 적용합니다.

미션은 최소 500개 문서, 두 개 이상 카테고리, 결측치 처리, 개인정보 미포함 확인, 라이선스 확인을 요구합니다. 헤더 제거만으로 본문의 식별 가능한 이름과 연락처가 모두 제거됐다고 판단할 수 없습니다. 실제 데이터 검수와 이용 조건 확인 기록이 필요합니다.

### NumPy TF-IDF 수식

문서 수를 $N$, 학습 어휘 크기를 $|V|$, 문서 $d$의 단어 $t$ 출현 횟수를 $c_{d,t}$라 두면 다음과 같습니다.

$$\mathrm{tf}(d,t)=c_{d,t},\quad \mathrm{df}(t)=\sum_{d=1}^{N}\mathbf{1}[c_{d,t}>0]$$

$$\mathrm{idf}(t)=\ln\left(\frac{1+N}{1+\mathrm{df}(t)}\right)+1,\quad x_{d,t}=\mathrm{tf}(d,t)\mathrm{idf}(t)$$

$$\hat{x}_d=\frac{x_d}{\|x_d\|_2}\quad(\|x_d\|_2>0)$$

영벡터는 그대로 유지합니다. 행렬 shape는 `(문서 수, 학습 어휘 크기)`이며, 어휘 열을 알파벳 순으로 정렬하고 `float64`로 계산합니다. 사전 미등록 단어는 변환 시 제외하는 설계입니다.

참조 검증 설정은 아래와 같습니다. `preprocess_pipeline`은 구현 계획상의 함수이며 코드 이관 후 실제 이름과 동작을 대조해야 합니다.

```python
TfidfVectorizer(
    analyzer=preprocess_pipeline,
    stop_words=None,
    dtype=np.float64,
    smooth_idf=True,
    sublinear_tf=False,
    norm="l2",
)
```

callable analyzer가 소문자화, 토큰화와 불용어 제거를 모두 수행해야 합니다. 양쪽 구현에서 토큰열, 어휘와 열 순서, TF, DF, IDF, 최종 벡터를 단계별로 비교해야 합니다.

### 검색과 분류 구조

전처리 → Train 전용 어휘·IDF 학습 → 문서 및 쿼리 벡터화 → 검색 또는 분류 순서입니다. 전처리 규칙을 공유하면 참조 검증과 실서비스 입력의 차이를 줄일 수 있습니다. 벡터화와 검색을 분리하면 TF 변형과 검색 전략을 독립적으로 비교할 수 있습니다.

코사인 유사도는 $\mathrm{cos}(d,q)=\frac{d\cdot q}{\|d\|_2\|q\|_2}$입니다. 두 벡터가 L2 정규화돼 있으면 NumPy 행렬곱 `D @ q`로 계산할 수 있습니다. 점수 내림차순, 동점 시 문서 ID 오름차순, 0점 문서 제외가 계획상의 정책입니다. OOV만 있는 쿼리나 빈 입력은 빈 검색 결과로 처리합니다.

분류는 동일한 Train TF-IDF로 `LogisticRegression(random_state=42)`를 학습하고 Test에서 Accuracy, Macro-F1 및 혼동 행렬을 산출하는 설계입니다. 실제 학습 옵션과 수렴 상태는 원본 코드 확인이 필요합니다.

### 실행 방법

현재 검증된 환경은 Windows, Python 3.14.7, uv 0.13.7입니다. 미션의 Python 3.10 이상 조건과 별도로, 현재 고정 패키지 조합은 Python 3.14.7에서 확인했습니다.

CMD에서 실행합니다.

```bat
cd /d D:\git\A4_1
.venv\Scripts\activate
uv pip install -r requirements.txt
```

PowerShell에서 실행합니다.

```powershell
Set-Location D:\git\A4_1
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

최초 환경 생성은 `uv venv`를 사용합니다. 현재 `.venv`는 생성 및 패키지 설치가 완료됐습니다. 활성화는 해당 터미널의 Python 경로를 선택합니다. 자동화에서는 `uv pip install --python .venv\Scripts\python.exe -r requirements.txt`처럼 대상을 지정할 수도 있습니다.

CLI 안 A는 아래 명령으로 검증하도록 요청됐습니다. 현재는 Python 소스 이관이 선행돼야 합니다.

```bash
python verify_tfidf.py
python main.py --build-index
python main.py --query "space satellite launch orbit" --topk 5
python main.py --train-classifier
python main.py --bonus
```

함수 안 B는 다음 반환 규격을 계획합니다. 아래 예시는 구현 계획상의 API이며 이관 후 실제 엔트리포인트와 대조해야 합니다.

```python
from src.search_engine import SearchEngine

engine = SearchEngine.load()
results = engine.search(query="space satellite launch orbit", top_k=5)
for score, doc_id, text_snippet in results:
    print(f"{score:.4f} {doc_id} {text_snippet}")
# 반환 규격 List[Tuple[float, int, str]]
```

## 2. 검증 및 실험 결과

### 로컬 실행 결과

NumPy 2.5.4, scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2 로딩을 확인했으며 `uv pip check`에서 설치된 19개 패키지의 의존성 호환성 검사를 통과했습니다. 알고리즘 테스트 파일은 현재 없습니다.

요청된 네 명령은 모두 실제 실행했으며 종료 코드는 각각 2입니다. 다음은 실행 출력의 일부입니다.

```text
$ python verify_tfidf.py
exit_code=2
can't open file 'D:\\git\\A4_1\\verify_tfidf.py': [Errno 2] No such file or directory

$ python main.py --query space satellite launch orbit --topk 5
exit_code=2
can't open file 'D:\\git\\A4_1\\main.py': [Errno 2] No such file or directory
```

전체 출력과 분류·보너스 실행 결과는 [로컬 실행 로그](artifacts/verification_local.log)에 있습니다. 로그는 `.gitignore`의 `*.log` 규칙으로 기본 제외되므로 제출에 포함할 때 추적 여부를 확인해야 합니다.

### 기존 보고 수치의 재현 상태

아래 수치는 [선행 체크리스트](docs/repo_readme/docs_03_verification_and_checklists.md)에 기록돼 있습니다. 이번 로컬 실행에서는 성능 수치를 산출하지 못했습니다.

| 항목 | 기존 문서 기록 | 로컬 검증 상태 |
| --- | --- | --- |
| Train / Test | 3,033 / 759 | 데이터 부재로 대조 대기 |
| TF-IDF 최대 절대 오차 | 1.1102e-15 | 검증 스크립트 부재 |
| Accuracy | 97.56% | 분류 코드 부재 |
| Macro-F1 | 0.9758 | 분류 코드 부재 |
| 혼동 행렬 | 생성 완료로 기록 | 이미지 부재 |

미션의 허용 오차는 `1e-6` 이내입니다. 기존 기록 `1.1102e-15`는 이 기준보다 작습니다. 아키텍처 문서에 적힌 `1e-15 미만` 기준과 비교하면 이 기록은 해당 기준을 초과합니다. 발표에서는 미션 기준과 실제 측정값을 각각 명시해야 합니다.

혼동 행렬의 요청 대상은 `artifacts/confusion_matrix.png`입니다. 이미지가 생성되면 README에서 해당 경로를 연결하고 실제 클래스 순서, 축 레이블, 건수를 확인해야 합니다. 현재 파일 부재로 이미지 삽입을 보류했습니다.

검색 품질은 관련도 정답과 평가 쿼리가 확보돼야 평가할 수 있습니다. 분류 Accuracy와 TF-IDF 수치 오차만으로 검색 관련도를 입증할 수 없으므로, 재검증 시 Top-5 출력과 문서별 관련성 판단도 기록해야 합니다.

## 3. BoW 한계와 개선 방향

### 표현 방식의 한계

`dog bites man`과 `man bites dog`는 단어 빈도가 같아 unigram BoW 벡터가 같습니다. 문장 역할과 어순에 따른 의미 차이를 잃습니다. `bank`처럼 금융기관과 강둑을 함께 뜻하는 단어는 문맥을 구분하기 어렵습니다. 동의어는 서로 다른 열로 분리되므로 의미가 가까운 문서의 점수가 낮아질 수 있습니다. 짧거나 OOV가 많은 문서는 벡터가 빈약해지고, 제거된 부정어는 의미 반전을 숨길 수 있습니다.

### 실제 오분류 5건 분석의 확보 상태

미션은 Test에서 실제로 잘못 분류한 문서 5건 이상의 분석을 요구합니다. 현재 예측 결과와 문서 원문이 없어 개별 원인을 확정할 수 없습니다. 재실행 후 아래 항목을 각 사례에 기록해야 합니다.

| 사례 | 필요한 실측 근거 | 검토할 BoW 원인 |
| --- | --- | --- |
| 1 | Test 문서 ID, 실제·예측 레이블, 비식별 본문, 기여 단어 | 어순·부정 표현 손실 |
| 2 | Test 문서 ID, 실제·예측 레이블, 비식별 본문, 기여 단어 | 다의어의 문맥 혼동 |
| 3 | Test 문서 ID, 실제·예측 레이블, 비식별 본문, 기여 단어 | 두 주제의 공유 어휘 |
| 4 | Test 문서 ID, 실제·예측 레이블, 비식별 본문, 기여 단어 | 짧은 문서·OOV·희소성 |
| 5 | Test 문서 ID, 실제·예측 레이블, 비식별 본문, 기여 단어 | 동의어 분리·전처리 정보 손실 |

위 다섯 행은 분석 관점이며 실제 오분류 사례를 확보한 상태는 아닙니다. 문서당 실제·예측 클래스 간 선형 모델 기여도 차이와 원문 문맥을 함께 검토하면 원인 가설을 점검할 수 있습니다.

후속 비교는 bigram으로 어순 일부를 보존하고, 문맥 임베딩 또는 Transformer 모델로 다의어와 동의어를 다루는 방향입니다. 같은 데이터 분할과 관련도 정답을 유지해 Accuracy, Macro-F1, Precision@5, nDCG@5를 비교해야 합니다. 백만 문서 확장 시 dense 행렬의 메모리와 전체 스캔이 병목이 됩니다. 희소 저장, 역색인, Top-K 가지치기 및 캐시를 비교하고 인덱스 구축 시간과 질의 지연 p50/p95를 별도로 측정해야 합니다.

근거 문서는 [미션 PDF](docs/A4_1_mission.pdf) 2~5쪽, [평가 PDF](docs/A4_1_eval.pdf) 10~11쪽, [구현 계획](IMPLEMENTATION_PLAN.md)입니다. [평가 대비 Q&A](docs/A4_1_mission_QnA.md)와 [발표 개요](docs/A4_1_presentation_outline.md)를 함께 제공합니다.

## 보너스 과제 구현 및 검증

보너스 검증은 `python main.py --bonus`의 코드 이관 후 다시 수행해야 합니다.

### TF 변형

Raw Count는 $tf$, Log Normalization은 $\ln(1+tf)$, Double Normalization은 양수 빈도에 대해 $0.5+0.5tf/\max(tf)$를 사용합니다. 빈도가 0인 원소는 0으로 유지하고 빈 문서는 영벡터로 처리합니다. IDF·분할·쿼리를 고정해 각 방식의 Macro-F1, Accuracy, 검색 관련도와 순위 변화를 비교해야 합니다. 선행 문서의 Double Norm Accuracy 기록은 96.66%이며 로컬 재측정 대기 상태입니다.

### BM25

$$\mathrm{score}(d,q)=\sum_{t\in q}\ln\left(1+\frac{N-\mathrm{df}(t)+0.5}{\mathrm{df}(t)+0.5}\right)\frac{tf(t,d)(k_1+1)}{tf(t,d)+k_1(1-b+b|d|/\mathrm{avgdl})}$$

계획값은 $k_1=1.5$, $b=0.75$입니다. Toy 문서 세 건을 수작업 계산한 값과 대조하고, 같은 쿼리에서 TF-IDF와 BM25 Top-5 결과를 비교해야 합니다. 쿼리 중복 토큰 처리도 구현과 수작업 계산에서 일치시켜야 합니다.

### 역색인

단어에서 `(doc_id, tf)` posting 리스트로 연결합니다. Dense 전체 스캔은 질의당 $O(N|V|)$이며 역색인은 쿼리 단어의 posting을 순회하고 후보를 정렬하는 비용이 발생합니다. 흔한 단어로 후보가 늘어나면 가속 폭이 줄어듭니다. 같은 0점 제외 및 동점 정책으로 점수 오차와 순위를 비교하고 구축·질의 시간을 분리해 측정해야 합니다.

| 항목 | 기존 보고 | 로컬 검증 상태 |
| --- | --- | --- |
| BM25 Toy 오차 | 5.55e-17 | 코드와 원시 계산 부재 |
| 역색인 점수 오차 | 2.78e-17 | 코드와 실행 로그 부재 |
| 역색인 순위 일치율 | 100% | 평가 쿼리·순위 결과 부재 |
