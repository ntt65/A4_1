# A4_1 평가표 순서별 질의응답

2026-11-12 기준으로 [평가 원문 PDF](A4_1_eval.pdf) 10~11쪽의 질문 순서를 따라 답변했습니다. 원문에는 **항목 1~5**만 표시돼 있으며, `1-1`~`4-4`는 각 항목 안의 질문을 찾기 위해 붙인 순서 번호입니다. 숫자 배점은 원문에 없고 항목별 PASS/FAIL 및 보너스 부여/미부여 형식입니다. 아래 내용은 현재 프로젝트의 설명과 검증 근거이며 공식 평가 결과를 예측하지 않습니다.

질문은 16개이며, 마지막 항목 5에서 보너스 3가지를 별도로 설명합니다. 학습목표별 정리는 [미션 Q&A](A4_1_mission_QnA.md), 전체 구현·실험·한계 분석은 [README](../README.md), 실행 가능한 통합본은 [단일 노트북](../A4_1_mission.ipynb)에 있습니다.

## 답변과 실측의 기준

| 항목 | 이 저장소의 확인 결과 |
| --- | --- |
| 데이터·분할 | 원본 3,894건 → 빈 본문 107건 제외 → Train 3,029 / Test 758, 4개 범주 |
| TF-IDF | 어휘 33,297개, Train 최대 절대 오차 `2.3315e-15`, Test `8.3267e-16`, IDF·검증 질의 오차 0 |
| 분류 | Accuracy `92.4802%`, Macro-F1 `0.925329`, Test 758건 중 701건 정답 |
| 실제 오류 분석 | Test ID `3`, `5`, `13`, `19`, `25`의 실제·예측 레이블과 기여도 확인 |
| 보너스 | TF 3종 비교, BM25 Toy 오차 `2.2204e-16`, TF-IDF 역색인 7질의 Top-5 순위 일치 |
| 실행 근거 | 회귀 검사 59개 통과, 단일 노트북 28셀·코드 14셀을 새 커널에서 실행 |

명령은 **프로젝트 루트 `D:\git\A4_1`**에서 실행합니다. 아래 PowerShell 명령은 가상환경의 Python 경로를 직접 지정하므로 활성화 여부에 의존하지 않습니다. `python` 코드 블록은 같은 가상환경이나 노트북 커널에서 실행할 수 있는 작은 예제입니다. 전체 데이터를 다시 학습하는 명령은 별도로 표시했습니다.

## 항목 1

구현 결과, 보고서의 근거와 실행 재현성을 확인하는 여섯 질문입니다.

### 1-1. TF-IDF 직접 구현과 Scikit-learn의 오차가 허용 범위 이내이고, 검증 설정이 README에 명시돼 있는가?

**핵심 답변 3줄**

1. NumPy로 Raw TF·Smooth IDF·L2 정규화를 구현하고 sklearn에도 같은 전처리 함수를 전달했습니다.
2. 참조 설정은 `sublinear_tf=False`, `smooth_idf=True`, `norm='l2'`, `dtype=np.float64`입니다.
3. Train 최대 오차 `2.3315e-15`, Test `8.3267e-16`으로 미션 기준 `1e-6` 이내입니다.

**수식·코드 근거**

$$idf(t)=\ln\left(\frac{1+N_{train}}{1+df_{train}(t)}\right)+1,\qquad x_{d,t}=tf(d,t)idf(t)$$

[벡터라이저](../src/tfidf.py)는 정렬된 어휘 열, `float64`, 영벡터를 보존하는 정규화를 사용합니다. [검증 스크립트](../verify_tfidf.py)는 토큰·어휘·TF·DF·IDF를 확인한 뒤 Train/Test/질의를 대조합니다. 벡터 값만 비슷하고 열 순서가 다른 경우를 통과시키지 않습니다.

**실측·확인 명령**

[단계별 검증 JSON](../artifacts/tfidf_verification.json)과 [실제 출력](../artifacts/run_logs/verify_tfidf.txt)을 확인합니다. 다음 명령은 전체 데이터를 재계산합니다.

```powershell
.\.venv\Scripts\python.exe verify_tfidf.py
```

**구술 답변 예시**

“두 구현의 전처리·어휘 순서·TF와 IDF 정의를 먼저 맞췄습니다. 최대 오차는 Train에서 약 `2.33e-15`여서 `1e-6` 기준을 통과했고, 설정과 로그를 README에 연결했습니다.”

### 1-2. 코사인 유사도를 NumPy로 구현했으며 관련 문서가 상위에 반환되는가?

**핵심 답변 3줄**

1. 문서와 질의를 같은 어휘·IDF로 변환한 뒤 정규화된 행렬과 벡터의 내적으로 점수를 계산합니다.
2. 양수 점수만 선택하고 점수 내림차순, 동점에서는 문서 ID 오름차순으로 Top-K를 반환합니다.
3. 우주 질의의 상위 결과는 우주 관련 문서로 확인했으며, 검색 품질 정답셋에 대한 Precision@5 측정은 수행하지 않았습니다.

**수식·코드 근거**

$$s(d,q)=\cos(d,q)=\frac{d\cdot q}{\|d\|_2\|q\|_2}=\hat d\cdot\hat q$$

[검색 구현](../src/search_engine.py)의 핵심은 `scores = self.doc_vectors @ query_vector`입니다. `rank_scores`는 양수 후보에서 Top-K 경계를 부분 선택하고 경계 동점도 포함해 최종 정렬합니다. 빈 입력·불용어만 있는 입력·OOV 질의는 빈 리스트를 반환합니다.

**실측·확인 명령**

`space satellite launch orbit`의 Top-5 Train 문서 ID는 `1934, 2609, 755, 2145, 1725`이며 첫 점수는 `0.4737`입니다. [검색 로그](../artifacts/run_logs/search.txt), [재계산한 노트북 결과](../artifacts/notebook_verification.json)

```powershell
.\.venv\Scripts\python.exe main.py --query "space satellite launch orbit" --topk 5
```

**구술 답변 예시**

“L2 정규화한 TF-IDF에 NumPy 내적을 적용해 코사인 점수를 직접 계산합니다. 우주 질의에서 관련 문서를 관찰했지만, 이것만으로 검색 정확도를 수치로 주장하지는 않으며 별도 관련도 정답이 필요합니다.”

### 1-3. Accuracy·F1-Score와 혼동 행렬이 리포트에 명시돼 있는가?

**핵심 답변 3줄**

1. Train/Test를 층화 8:2, `random_state=42`로 고정하고 Logistic Regression을 학습했습니다.
2. Test 758건 중 701건을 맞혀 Accuracy `92.4802%`, Macro-F1 `0.925329`를 얻었습니다.
3. README에 클래스별 Precision·Recall·F1과 혼동 행렬을 제시하고 노트북에도 그림을 내장했습니다.

**수식·코드 근거**

$$Accuracy=\frac{701}{758},\qquad F1_c=\frac{2P_cR_c}{P_c+R_c},\qquad MacroF1=\frac14\sum_{c=1}^{4}F1_c$$

[분류기](../src/classifier.py)는 `LogisticRegression(max_iter=1000, random_state=42)`와 `average='macro'`를 사용합니다. 혼동 행렬의 행은 실제 클래스, 열은 예측 클래스이며 클래스 순서는 `target_names`와 같습니다.

**실측·확인 명령**

[분류 지표 JSON](../artifacts/classifier_metrics.json), [혼동 행렬](../artifacts/confusion_matrix.png). 다음 명령은 모델을 다시 학습합니다.

```powershell
.\.venv\Scripts\python.exe main.py --train-classifier
```

**구술 답변 예시**

“정확도와 함께 클래스별 F1을 동일한 비중으로 평균한 Macro-F1을 보고했습니다. 혼동 행렬을 보면 어떤 범주끼리 섞이는지 알 수 있고, 전체 57건의 오류 중 다섯 사례를 본문 맥락과 모델 기여도로 분석했습니다.”

### 1-4. 전처리가 정상 동작하며 토큰화·불용어 기준의 설계 근거가 있는가?

**핵심 답변 3줄**

1. 소문자화 후 `r'(?u)\b\w\w+\b'`로 두 글자 이상의 Unicode 단어를 추출합니다.
2. 제공 소스의 123개 불용어를 프로젝트 기준 `a4-1-en-v1`로 고정하고 해시와 선정 근거를 기록했습니다.
3. `C++`·한 글자·부정어 손실을 한계로 설명하고, 같은 analyzer를 NumPy와 sklearn에 적용합니다.

**수식·코드 근거**

```python
from src.preprocess import TextPreprocessor

tokens = TextPreprocessor(stopwords=set()).tokenize("Space, SATELLITE! a 3D_x")
assert tokens == ["space", "satellite", "3d_x"]
```

숫자·밑줄도 `\w`에 포함되므로 특수문자를 전부 없애는 규칙은 아닙니다. 영어용 기준이며 한국어에 적용할 때는 형태소 분석과 공백 토큰화의 어휘·OOV·성능 차이를 비교해야 합니다. [전처리 구현](../src/preprocess.py), [불용어 목록](../resources/stopwords_en.txt), [선정 근거·해시](A4_1_data_sources.md)

**실측·확인 명령**

불용어는 123개, 정규화 UTF-8·LF SHA-256은 `566c5909c1523a651dc2691ff0fa4fddabcc2d047daceccd2f4b962789af313d`입니다. 이 명령은 작은 입력을 사용한 전처리·벡터·검색 회귀 검사를 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_core.py -v
```

**구술 답변 예시**

“검증의 기준을 고정하기 위해 공통 토큰화 함수와 불용어 목록을 사용했습니다. 기능어를 줄이는 이점이 있지만 `not`이나 `C++` 정보가 사라질 수 있어, 규칙 변경의 개선 효과는 같은 분할에서 별도로 비교해야 합니다.”

### 1-5. 실제 오분류 5건 이상의 원인 분석이 포함돼 있는가?

**핵심 답변 3줄**

1. 실제 Test 오분류를 ID 오름차순으로 추출해 `3, 5, 13, 19, 25`의 다섯 사례를 분석했습니다.
2. 실제·예측 클래스, 짧은 본문 근거, 단어별 기여와 절편 차이를 대조했습니다.
3. 짧은 문서·약어·서명 잔존·전처리 손실을 BoW 한계와 연결하되 개선 효과는 가설과 구분합니다.

**수식·코드 근거**

$$\Delta=(b_{\hat y}-b_y)+\sum_t x_t(w_{\hat y,t}-w_{y,t})$$

양수 기여는 예측 클래스를, 음수 기여는 실제 클래스를 지지합니다. [분류 구현](../src/classifier.py)은 기여도와 절편 차이를 계산하고 공개 JSON에는 문서 ID·feature ID·수치를 저장합니다.

**실측·확인 근거**

| Test ID | 실제 → 예측 | 분석 핵심 |
| --- | --- | --- |
| 3 | mideast → space | 정치적 구문 단서보다 범용 어휘와 절편의 합이 커 최종 점수 차이 `+0.0123` |
| 5 | graphics → space | `base`, `system`과 짧은 대화 문맥, 한 글자 토큰 손실이 작용해 `+0.4278` |
| 13 | graphics → baseball | 기술 어휘보다 잔존 서명·일반 표현의 기여가 커 `+0.2199` |
| 19 | baseball → space | 짧은 약어·기사 날짜 표현을 주제 맥락과 분리해 처리해 `+0.0233` |
| 25 | mideast → space | 전처리 후 한 토큰만 남고 절편 차이가 커 `+0.1136` |

[README의 실제 사례 분석](../README.md), [오분류 수치](../artifacts/classifier_metrics.json). 노트북은 동일한 다섯 ID와 점수 분해를 다시 확인합니다.

**구술 답변 예시**

“선택한 다섯 건은 만들어 낸 예제가 아니라 Test에서 실제 틀린 문서입니다. 모델 점수의 단어 기여와 절편을 분해해 주제 단서 부족이나 일반 표현의 영향이 어떻게 오판으로 이어졌는지 설명했습니다.”

### 1-6. README에 실행 방법이 명시돼 있고 실제 실행 가능한가?

**핵심 답변 3줄**

1. CLI `--query`, `--topk`와 함수 `search(query, top_k)`를 모두 제공합니다.
2. 반환 형식은 `(score, doc_id, text_snippet)` 목록이며 인덱스 저장·재로딩도 확인했습니다.
3. 실행 명령·예시 출력·의존성을 README에 제시하고 단일 노트북의 독립 실행도 검증했습니다.

**수식·코드 근거**

```python
from main import search

# search(query: str, top_k: int = 5) -> List[Tuple[float, int, str]]
results = search("space satellite launch orbit", top_k=5)
assert len(results) == 5
assert all(len(row) == 3 for row in results)
for score, doc_id, _text_snippet in results:
    print(f"{score:.4f} | {doc_id}")
```

이 예제는 데이터·인덱스를 로드하므로 처음에는 다운로드나 인덱스 구축이 필요합니다. 함수는 150자 이하 스니펫을 반환하며 위 출력은 점수·ID만 표시합니다. CLI 인덱스는 `artifacts/index.pkl`, 노트북에서 생성하는 인덱스는 `artifacts/notebook/search_index`의 NPY·JSON입니다.

**실측·확인 명령**

요청된 네 CLI 명령의 종료 코드는 모두 0입니다. 노트북은 빈 폴더에 파일 하나만 복사하고 새 커널에서 28셀·코드 14셀을 실행했습니다. [명령별 기록](../artifacts/verification_results.json), [노트북 실행 보고서](../artifacts/notebook_verification.json)

```powershell
.\.venv\Scripts\python.exe scripts/verify_notebook.py
```

**구술 답변 예시**

“README의 CLI와 함수 예제 중 어느 쪽으로도 시연할 수 있습니다. 환경 활성화 없이도 가상환경 Python 경로를 직접 지정하면 되고, 노트북에서는 구현·설명·실험 결과를 한 파일에서 확인할 수 있습니다.”

## 항목 2

구조를 나눈 이유와 다른 데이터에 적용할 때의 설계를 설명합니다.

### 2-1. 전처리·벡터화·검색·분류를 분리했으며 이유를 설명할 수 있는가?

**핵심 답변 3줄**

1. 데이터 로딩, 토큰화, 벡터화, 검색, 분류를 각각 함수·클래스와 모듈로 나눴습니다.
2. 검색·분류가 같은 표현을 재사용하고, 검증에서 문제 발생 단계를 좁힐 수 있습니다.
3. 전처리·분류기·검색 전략을 교체할 때 나머지 구성과의 입력·출력 계약을 유지하기 쉽습니다.

**수식·코드 근거**

```text
dataset.load_data → TextPreprocessor.tokenize → CustomTfidfVectorizer
                                               ├─ SearchEngine.search
                                               └─ train_and_evaluate_classifier
```

[dataset.py](../src/dataset.py), [preprocess.py](../src/preprocess.py), [tfidf.py](../src/tfidf.py), [search_engine.py](../src/search_engine.py), [classifier.py](../src/classifier.py)에 역할을 나눴습니다. 노트북은 같은 책임을 셀 안의 함수·클래스로 표현합니다.

**실측·확인 근거**

어휘·TF·DF·IDF 단계 검증과 검색·분류 결과를 각각 확인할 수 있습니다. 예를 들어 단어 열 순서가 다르면 벡터 검증에서 원인을 찾고, 정렬 정책은 검색 회귀 검사에서 독립적으로 확인합니다.

**구술 답변 예시**

“공통 전처리와 벡터를 재사용하면서 검색과 분류의 책임을 분리했습니다. 파일을 나누는 것 자체보다 입력·출력 계약을 고정하고 각 단계를 독립적으로 검증할 수 있다는 점이 핵심입니다.”

### 2-2. TF → IDF → TF-IDF의 단계별 구현을 설명할 수 있는가?

**핵심 답변 3줄**

1. Train의 토큰 집합을 정렬해 어휘 열을 고정하고 문서별 빈도 행렬 TF를 만듭니다.
2. 각 단어가 나타난 Train 문서 수 DF를 세고 Smooth IDF를 계산합니다.
3. TF의 각 열에 IDF를 곱한 뒤 행별 L2 정규화하며 Test·질의에는 기존 어휘·IDF를 적용합니다.

**수식·코드 근거**

```python
import numpy as np

tf = np.array([[2, 1], [0, 1]], dtype=np.float64)
df = np.count_nonzero(tf, axis=0)
idf = np.log((1 + len(tf)) / (1 + df)) + 1
weighted = tf * idf
norms = np.linalg.norm(weighted, axis=1, keepdims=True)
x = np.divide(weighted, norms, out=np.zeros_like(weighted), where=norms != 0)
assert np.allclose(idf, [1.4054651081081644, 1.0])
assert np.allclose(np.linalg.norm(x, axis=1), 1.0)
```

실제 구현의 `fit`은 어휘·DF·IDF를 학습하고, `count_matrix`는 TF, `transform`은 IDF 곱셈·정규화를 담당합니다. [TF-IDF 구현](../src/tfidf.py)

**실측·확인 근거**

Train 어휘 33,297개에서 TF·DF가 참조와 일치하고 IDF 최대 오차는 0입니다. Train/Test에서 `fit`을 각각 실행하면 서로 다른 공간이 되므로 Test는 `transform`만 수행합니다.

**구술 답변 예시**

“단어 빈도 행렬을 먼저 만들고, 같은 Train 코퍼스의 문서 빈도로 IDF를 계산해 열별로 곱합니다. 마지막 행 정규화를 거친 결과를 검색과 분류가 공유하며, Test에서 어휘나 IDF를 새로 학습하지 않습니다.”

### 2-3. 다른 데이터셋에 쉽게 적용하도록 어떤 점을 고려했는가?

**핵심 답변 3줄**

1. 데이터 로더가 문서 목록·레이블·클래스 이름을 반환하도록 입력 규격을 분리했습니다.
2. 벡터라이저가 전처리 객체를 받으므로 언어별 tokenizer와 불용어 정책을 바꿀 수 있습니다.
3. 데이터가 바뀌면 분할·어휘·IDF·인덱스를 다시 만들고 동일한 검증 절차를 반복해야 합니다.

**수식·코드 근거**

`load_data()`의 반환 규격은 `(train_docs, y_train), (test_docs, y_test), target_names`입니다. `CustomTfidfVectorizer(preprocessor=...)`로 다른 tokenizer를 주입할 수 있습니다. [데이터 로더](../src/dataset.py), [벡터라이저 생성자](../src/tfidf.py)

**실측·확인 근거**

현재는 네 범주와 시드 42를 고정했습니다. 한국어 데이터로 바꾸는 경우 형태소 분석 또는 공백 토큰화, OOV, 범주 분포, 같은 데이터 내 고정 분할을 다시 확인합니다. 현재의 92.4802%를 다른 데이터셋의 성능으로 재사용할 수는 없습니다.

**구술 답변 예시**

“데이터 입력과 토큰화 규칙을 교체하고 이후 벡터화·검색 코드를 재사용할 수 있게 했습니다. 다만 기존 인덱스를 그대로 쓰는 자동 호환을 구현한 것은 아니므로, 데이터가 바뀌면 학습과 인덱스 구축을 다시 수행해야 합니다.”

## 항목 3

TF-IDF와 코사인의 수학적 의미를 설명합니다.

### 3-1. TF·IDF·TF-IDF의 정의와 IDF에서 로그를 쓰는 이유는 무엇인가?

**핵심 답변 3줄**

1. TF는 문서 안에서 단어가 나타난 횟수이며 DF는 그 단어를 포함하는 문서 수입니다.
2. IDF는 여러 문서에 흔한 단어보다 드문 단어의 구별력을 높이는 가중치입니다.
3. 로그는 역문서빈도 비율의 큰 차이를 완화하고 smoothing은 안정적인 계산 기준을 제공합니다.

**수식·코드 근거**

$$tf(d,t)=\sum_{u\in T(d)}\mathbf1[u=t],\qquad df(t)=\sum_d\mathbf1[tf(d,t)>0]$$
$$idf(t)=\ln\left(\frac{1+N}{1+df(t)}\right)+1,\qquad TFIDF(d,t)=tf(d,t)idf(t)$$

여기서 $N$은 Train 문서 수입니다. 모든 Train 문서에 등장하면 IDF는 1이고, 드문 단어의 IDF는 더 큽니다. 스무딩 없는 $\ln(N/df)$도 정의할 수 있지만, 이번 검증에서는 sklearn의 Smooth IDF와 정확히 맞췄습니다.

**실측·확인 근거**

2-2의 작은 예제에서는 DF가 1인 첫 단어의 IDF가 약 `1.405465`, DF가 2인 두 번째 단어는 `1`입니다. 실제 구현과 sklearn의 IDF 오차는 0입니다.

**구술 답변 예시**

“TF는 문서 안의 빈도, IDF는 코퍼스 전체에서 얼마나 흔한지를 보정하는 값입니다. 로그를 통해 희귀성의 차이를 완만하게 반영하고, 두 값을 곱해 문서에서 자주 나타나면서 코퍼스에서는 덜 흔한 단어를 강조합니다.”

### 3-2. 정규화가 왜 필요하며 코사인을 유클리드 거리 대신 사용하는 이유는 무엇인가?

**핵심 답변 3줄**

1. 원시 내적과 유클리드 거리는 문서 길이에 따른 벡터 크기의 영향을 받습니다.
2. L2 정규화한 벡터의 내적은 코사인이므로 단어 가중치의 방향을 비교할 수 있습니다.
3. 단위 벡터에서는 유클리드 거리와 코사인이 같은 순위를 줄 수 있으며 영벡터는 별도 처리합니다.

**수식·코드 근거**

$$\cos(x,q)=\frac{x\cdot q}{\|x\|_2\|q\|_2},\qquad
\|\hat x-\hat q\|_2^2=2-2\cos(x,q)$$

```python
import numpy as np

x, q = np.array([3.0, 4.0]), np.array([0.0, 2.0])
score = (x @ q) / (np.linalg.norm(x) * np.linalg.norm(q))
assert np.isclose(score, 0.8)
assert np.isclose((x / np.linalg.norm(x)) @ (q / np.linalg.norm(q)), score)
```

**실측·확인 근거**

[검색 구현](../src/search_engine.py)은 이미 정규화된 문서·질의에 내적을 사용합니다. 빈 문자열·OOV는 0벡터를 유지해 검색 결과 `[]`로 처리하며, 0으로 나눠 NaN을 만드는 방법을 사용하지 않습니다.

**구술 답변 예시**

“주제가 같은 문서라도 길이가 다르면 벡터의 크기가 달라질 수 있어 방향 중심의 코사인을 사용했습니다. 정규화된 경우 유클리드 거리와의 관계도 성립하므로 코사인이 언제나 다른 기준보다 우월하다는 뜻은 아닙니다.”

### 3-3. TF-IDF 행렬의 shape가 `(문서 수, 어휘 크기)`인 이유는 무엇인가?

**핵심 답변 3줄**

1. 행 하나는 문서 하나이며 열 하나는 Train에서 학습한 어휘의 단어 하나입니다.
2. 각 원소는 해당 문서·단어 조합의 TF-IDF 가중치입니다.
3. Test와 질의도 같은 열을 사용하므로 문서 수만 달라지고 어휘 차원은 고정됩니다.

**수식·코드 근거**

$$X_{train}\in\mathbb R^{3029\times33297},\quad
X_{test}\in\mathbb R^{758\times33297},\quad q\in\mathbb R^{33297}$$

따라서 `X_train @ q`의 결과 shape는 `(3029,)`로, Train 문서별 점수 하나씩입니다. 질의를 행 형태 `(1, 33297)`로 만들었다면 내적 전에 1차원으로 꺼내거나 전치해야 합니다.

**실측·확인 근거**

[TF-IDF 검증 JSON](../artifacts/tfidf_verification.json)에 Train/Test/질의 shape가 저장돼 있습니다. 빈/OOV 입력도 열 수는 유지하며 모든 원소가 0이 됩니다.

**구술 답변 예시**

“행은 비교할 문서이고 열은 공통 사전의 단어이므로 두 축의 크기가 그렇게 정해집니다. Test 문서에 새 단어가 있어도 열을 추가하지 않고 OOV로 처리해야 Train과 같은 공간에서 비교할 수 있습니다.”

## 항목 4

표현 방식의 한계, 확장 병목과 후속 품질 평가를 설명합니다.

### 4-1. 오분류에서 발견한 BoW의 근본적 한계를 구체적 사례로 설명할 수 있는가?

**핵심 답변 3줄**

1. Unigram BoW는 어순과 문장 안의 관계를 버리므로 같은 빈도라면 의미가 달라도 같은 벡터가 됩니다.
2. 실제 오류에서는 짧은 문서·약어·일반 단어·전처리 손실이 주제 단서를 약하게 만들었습니다.
3. 개별 오분류의 원인 해석과 개선 성능의 증명은 구분하고 후속 실험으로 확인해야 합니다.

**수식·코드 근거**

```python
from collections import Counter

assert Counter("dog bites man".split()) == Counter("man bites dog".split())
```

두 문장은 같은 빈도와 IDF를 적용하므로 같은 TF-IDF가 됩니다. 한편 `not`을 불용어로 처리해 입력에서 제거하면 `not good`과 `good`의 차이가 사라질 수 있습니다.

**실측·확인 근거**

Test ID 25는 전처리 후 한 토큰만 남아 절편 영향이 컸고, ID 19는 약어와 이전 대화 맥락이 주제 판단에 충분히 반영되지 않았습니다. 다섯 사례의 실제 수치와 설명은 [1-5](#1-5-실제-오분류-5건-이상의-원인-분석이-포함돼-있는가)와 [README](../README.md)에서 대조할 수 있습니다.

**구술 답변 예시**

“단어가 무엇인지는 남지만 누가 무엇을 했는지는 잃습니다. 실제 오류에서도 주제 어휘가 적은 짧은 글이나 문맥에 의존하는 약어를 일반 단어의 빈도만으로 판단하는 한계를 확인했습니다.”

### 4-2. ‘사과’의 여러 의미가 섞이는 원인과 해결 접근은 무엇인가?

**핵심 답변 3줄**

1. BoW는 같은 표면형을 하나의 어휘 열로 합치므로 과일·기업·용서의 의미를 직접 구별하지 못합니다.
2. 주변 단어·구·개체명을 활용하면 일부 문맥 단서를 추가할 수 있습니다.
3. Bigram·질의 확장·문맥 임베딩 등을 같은 관련도 정답셋에서 비교해 효과를 검증해야 합니다.

**수식·코드 근거**

`사과`라는 한 열에 서로 다른 의미의 출현 횟수가 누적됩니다. `사과 과수원`, `애플 신제품`, `사과를 받아들이다`처럼 주변 표현은 다르지만 Unigram 자체에는 단어 의미를 구분하는 별도 상태가 없습니다. 이 문장들은 한계를 설명하기 위해 작성한 예시입니다.

**실측·확인 근거**

현재 실험은 영어 네 범주를 사용하므로 한국어 `사과` 질의에 대한 실제 성능 수치는 없습니다. 한국어 형태소 분석·`Apple`과 `사과`의 자동 치환·의미 구분은 구현하지 않았으며 위 해결 접근은 후속 제안입니다. TF-IDF·BM25 비교는 수행했지만 의미별 검색 품질의 우열을 검증한 결과로 해석할 수는 없습니다.

**구술 답변 예시**

“같은 글자 형태가 같은 열로 모이는 것이 원인입니다. 주변 구문이나 문맥 표현을 넣는 방향으로 개선하되, 과일·기업·용서 의도별 정답 문서를 만들어 기존 TF-IDF와 비교하겠습니다.”

### 4-3. 문서 100만 건의 실시간 검색에서 병목은 무엇이며 어떻게 대응하겠는가?

**핵심 답변 3줄**

1. 현재 Dense TF-IDF는 메모리 `O(N|V|)`와 질의별 전체 내적 `O(N|V|)`이 병목입니다.
2. 희소 저장·역색인으로 질의 단어의 posting과 후보만 처리하고 캐시·인덱스 갱신을 설계합니다.
3. 구축 시간·메모리·질의 지연 p50/p95·동시성·검색 품질을 함께 측정해야 합니다.

**수식·코드 근거**

$$8N|V|=8\times10^6\times50{,}000=400\times10^9\text{ bytes}\approx372.5\text{ GiB}$$

위 값은 어휘 5만 개, `float64`의 원소 저장만 계산한 추정치입니다. 실제 객체·인덱스 등의 추가 비용은 제외했습니다. 현재 [역색인](../src/bonus/inverted_index.py)은 질의의 비영 단어 posting을 누적합니다. 구현상 전체 점수 배열 초기화에 `O(N)`, posting 누적에 `O(P_Q)`가 들며 Top-K 후보 선택·동점 정렬 비용도 더해집니다.

**실측·확인 근거**

[기존 CLI 보너스 측정](../artifacts/bonus_metrics.json)에서 우주 질의는 Dense `24.0746ms`, 역색인 `0.3283ms`였습니다. 점수 계산·Top-5 선택의 3회 중앙값이며 토큰화·벡터화는 제외했습니다. 백만 문서의 실제 부하 시험이나 p95를 측정한 결과는 아닙니다.

**구술 답변 예시**

“현재 방식은 모든 문서와 단어를 순회하는 비용과 Dense 메모리가 먼저 한계가 됩니다. 희소 역색인으로 후보를 줄인 다음 인덱스 갱신과 동시 질의까지 포함한 부하 시험으로 실시간 요구를 확인하겠습니다.”

### 4-4. 다시 구현한다면 무엇을 개선하고 검색 품질을 어떻게 정량 평가하겠는가?

**핵심 답변 3줄**

1. 부정어 보존·도메인 토큰 규칙·Bigram을 작은 비교 실험으로 검토한 뒤 문맥 모델을 비교하겠습니다.
2. 분류는 같은 고정 분할에서 Accuracy·Macro-F1·클래스별 지표·오분류 유형의 변화를 측정합니다.
3. 검색은 질의별 관련도 정답을 만들고 Precision@5·Recall@K·nDCG@5와 지연 시간을 함께 평가합니다.

**수식·코드 근거**

$$Precision@5=\frac{\text{Top-5 중 관련 문서 수}}{5}$$
$$DCG@K=\sum_{i=1}^{K}\frac{2^{rel_i}-1}{\log_2(i+1)},\qquad nDCG@K=\frac{DCG@K}{IDCG@K}$$

관련도 등급, 질의 수, 정답 작성 기준을 먼저 고정하고 같은 질의로 방식들을 비교해야 합니다. 모델·전처리 선택을 위해 Test 정답을 반복적으로 사용하기보다 Train 내부 검증 분할을 마련하고 최종 Test는 마지막 확인에 사용합니다.

**실측·확인 근거**

현재 TF 변형의 분류 성능과 Top-5 겹침, BM25 결과 비교, 역색인 동치성과 시간은 측정했습니다. 관련도 정답셋에 대한 Precision@5·nDCG@5는 아직 측정하지 않았으며 겹침률을 대신 사용할 수 없습니다.

**구술 답변 예시**

“실제 오류와 연결되는 전처리·Bigram부터 바꾸고 분류 성능을 비교하겠습니다. 검색은 먼저 정답 질의를 구축해 Precision과 순위 지표를 측정해야 하며, 역색인의 순위 일치율 100%는 검색 품질 100%라는 뜻이 아닙니다.”

## 항목 5

원문은 **‘보너스 문제 해결에 따른 크레딧 부여’**이며 별도의 숫자 배점은 없습니다. 구현 내용과 검증 근거를 아래 세 질문으로 정리했습니다. 이 세 소제목은 답변 편의를 위한 구성이며 원문 평가표의 추가 필수 문항은 아닙니다.

### 5-1. Raw·Log·Double TF를 구현하고 검색·분류 영향을 비교했는가?

**핵심 답변 3줄**

1. Raw Count, `log(1+tf)`, Double TF를 같은 분할·어휘·IDF·정규화 조건에서 비교했습니다.
2. Double TF는 양수 빈도에만 `0.5 + 0.5 × tf/max(tf)`를 적용하고 0과 빈 문서는 0으로 보존합니다.
3. 분류 지표와 질의별 Top-5 구성을 비교했으며 순위 겹침을 관련도 기반 품질 지표로 해석하지 않습니다.

**수식·코드 근거**

$$tf_{log}=\ln(1+c),\qquad
tf_{double}=\begin{cases}0.5+0.5c/c_{max},&c>0\\0,&c=0\end{cases}$$

[TF 변형 구현](../src/bonus/tf_variations.py), [실험 실행기](../src/bonus/runner.py)

**실측·확인 근거**

| TF | Accuracy | Macro-F1 | 우주 질의 Raw Top-5와 겹침 |
| --- | --- | --- | --- |
| Raw | 92.4802% | 0.925329 | 100% |
| Log | 92.4802% | 0.925139 | 60% |
| Double | 91.9525% | 0.919962 | 40% |

이 분할에서는 Double이 Raw보다 분류 성능이 낮았고 Log는 Accuracy가 같아도 Macro-F1과 순위 구성이 달랐습니다. [실측 JSON](../artifacts/bonus_metrics.json)

**구술 답변 예시**

“빈도를 압축하는 방식이 언제나 유리하지는 않았습니다. 같은 분할에서 실제 분류 지표와 순위 변화를 비교했고, 검색 정확도의 개선을 주장하려면 별도 관련도 정답이 필요하다고 설명했습니다.”

### 5-2. BM25를 직접 구현하고 같은 질의의 TF-IDF 결과와 비교했는가?

**핵심 답변 3줄**

1. BM25의 TF 포화·문서 길이 보정과 양수 Robertson–Sparck Jones IDF를 NumPy로 구현했습니다.
2. `k1=1.5`, `b=0.75`를 사용하고 세 문서 Toy의 독립 수작업 계산과 대조했습니다.
3. 실제 네 질의에서 TF-IDF와 BM25의 Top-5 문서·점수를 비교했습니다.

**수식·코드 근거**

$$idf(t)=\ln\left(1+\frac{N-df(t)+0.5}{df(t)+0.5}\right)$$
$$s(D,Q)=\sum_{t\in unique(Q)}q_t\,idf(t)
\frac{f_{t,D}(k_1+1)}{f_{t,D}+k_1(1-b+b|D|/avgdl)}$$

위 식은 현재 [CLI 구현 `BM25Okapi.score`](../src/bonus/bm25.py)의 규칙으로, $q_t$는 질의 내 반복 횟수입니다. 단일 노트북의 `BonusBM25`는 고유 질의 단어를 한 번만 더하는 별도 규칙입니다. 비교 실험의 네 질의에는 반복어가 없어 이 차이가 해당 결과에 영향을 주지 않습니다.

**실측·확인 명령**

Toy 기대 점수 `[1.5908508973, 0.4700036292, 0]`와 실제 계산의 최대 오차는 `2.2204e-16`입니다. 이 명령은 작은 합성 입력으로 보너스 회귀 검사를 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_bonus.py -v
```

**구술 답변 예시**

“BM25는 단어가 반복될 때 기여가 포화되고 문서 길이를 보정합니다. 작은 예제를 수식에 직접 대입해 계산을 검증한 뒤 실제 동일 질의의 TF-IDF 순위와 비교했으며, 두 점수는 척도가 달라 절댓값만으로 품질을 비교하지 않습니다.”

### 5-3. 역색인을 구현하고 전체 스캔과 시간 복잡도를 비교했는가?

**핵심 답변 3줄**

1. 단어별 문서 ID·가중치 posting을 만들고 질의에서 필요한 posting만 누적합니다.
2. 일곱 질의에서 TF-IDF 전체 스캔과 점수·Top-5 순서가 일치하는지 확인했습니다.
3. 구축·질의 시간을 나눠 측정하고 구현상 `O(N)` 점수 배열 초기화와 Top-K 비용도 설명합니다.

**수식·코드 근거**

$$score_j=\sum_{t:q_t\ne0}X_{j,t}q_t$$

Dense 전체 내적은 `O(N|V|)`이며 현재 역색인은 점수 배열 초기화와 posting 순회에 `O(N+P_Q)`를 사용합니다. 후보 정렬 비용은 추가됩니다. [TF-IDF 역색인](../src/bonus/inverted_index.py), [비교 실험](../src/bonus/runner.py)

**실측·확인 근거**

CLI 실험은 빈 입력·OOV·반복어를 포함한 7질의에서 점수 최대 오차 0, Top-5 순위 일치율 100%입니다. 노트북은 TF-IDF뿐 아니라 자체 BM25의 전체/역색인도 각각 7질의로 대조합니다. [CLI 결과](../artifacts/bonus_metrics.json), [노트북 결과](../artifacts/notebook_verification.json)

전체 보너스를 다시 계산하려면 다음 명령을 실행합니다. 분류 세 번과 인덱스 구축이 포함되므로 작은 회귀 검사보다 시간·메모리가 필요합니다.

```powershell
.\.venv\Scripts\python.exe main.py --bonus
```

**구술 답변 예시**

“역색인은 관련 단어가 등장한 문서의 가중치만 순회해 같은 점수를 더합니다. 일치 검증과 시간 측정을 각각 수행했으며, 여기서 100%는 두 알고리즘의 순위가 같다는 뜻이지 관련성 정답을 모두 맞혔다는 뜻이 아닙니다.”

## 구술 시연 순서

1. README에서 환경·CLI·함수 API·최신 실측 표를 설명합니다.
2. TF-IDF의 작은 예제로 TF → DF → IDF → 정규화를 설명하고 전체 검증 로그를 제시합니다.
3. 우주 질의 Top-5, 빈/OOV 처리, 저장·재로딩 근거를 보여 줍니다.
4. Accuracy·Macro-F1·혼동 행렬과 실제 오류 다섯 건을 연결합니다.
5. 항목 2~4의 구조·수식·한계·검색 품질 평가 계획을 답합니다.
6. 마지막에 보너스의 TF 변형·BM25·역색인 결과를 별도로 설명합니다.

**근거 문서:** [평가 PDF](A4_1_eval.pdf), [미션 PDF](A4_1_mission.pdf), [README](../README.md), [학습목표 Q&A](A4_1_mission_QnA.md), [단일 노트북](../A4_1_mission.ipynb), [전체 실행 체크표](repo_readme/docs_03_verification_and_checklists.md)
