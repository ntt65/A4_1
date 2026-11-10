"""
GitHub Issues 일괄 등록 스크립트 (개정안 반영)
저장소: https://github.com/ntt65/A4_1
"""

import os
import sys
import json
import urllib.request
import urllib.error

REPO = "ntt65/A4_1"
API_URL = f"https://api.github.com/repos/{REPO}/issues"

token = os.environ.get("GITHUB_TOKEN")
if len(sys.argv) > 1 and not token:
    token = sys.argv[1]

if not token:
    print("=" * 60)
    print("❌ 오류: GitHub Personal Access Token (PAT)이 필요합니다.")
    print("사용법:")
    print("  python3 create_github_issues.py <YOUR_GITHUB_TOKEN>")
    print("=" * 60)
    sys.exit(1)

issues = [
    {
        "title": "[Issue #1] [Data] 20 Newsgroups 수집, 전처리 및 데이터 누수 방지 분할",
        "labels": ["enhancement", "phase-1", "data-pipeline"],
        "body": """### 🎯 목표
20 Newsgroups 데이터셋을 수집·정제하고, 시험 데이터 정보가 훈련 과정에 유입되지 않도록 고정 8:2 분할과 데이터 누수 방지 파이프라인을 구축합니다.

### 📝 작업 체크리스트
- [ ] `fetch_20newsgroups` 4개 카테고리(`sci.space`, `comp.graphics`, `rec.sport.baseball`, `talk.politics.mideast`) 다운로드 및 캐싱
- [ ] 메타데이터 누수 및 레이블 힌트 제거를 위한 `remove=('headers', 'footers', 'quotes')` 옵션 적용
- [ ] 실제 로드된 문서 수(~2,500건) 및 카테고리별 분포 로깅
- [ ] 데이터 윤리 및 보안 확인: 개인정보(PII) 미포함 확인 및 BSD 라이선스 고지
- [ ] **데이터 누수 방지**: Train/Test 8:2 단일 분할(`random_state=42`) 고정 및 결측치 제거
- [ ] Sklearn 호환 텍스트 정제 및 토큰화 정규식(`r'(?u)\\b\\w\\w+\\b'`) 구현 (1글자 단어 제외, 소문자화)
- [ ] 영어 표준 불용어(Stopwords) 정의 및 제거 모듈 구축

### 🔍 수락 기준 (Acceptance Criteria)
- [ ] Train 80% / Test 20% 분할이 고정 저장되어 이후 모든 모듈에서 동일 데이터셋을 참조할 것
- [ ] PII 및 저작권 체크리스트가 로그 또는 문서에 명시될 것"""
    },
    {
        "title": "[Issue #2] [Core-NLP] NumPy 기반 TF-IDF 직접 구현 및 Sklearn $10^{-6}$ 검증",
        "labels": ["core-algorithm", "phase-2", "math"],
        "body": """### 🎯 목표
순수 **NumPy**만을 사용하여 Train 데이터에 한정된 어휘 사전/IDF를 학습하고, Sklearn의 `TfidfVectorizer` 결과와 최대 오차 $10^{-6}$ 미만으로 일치함을 증명합니다.

### 📝 작업 체크리스트
- [ ] `CustomTfidfVectorizer` 클래스 구조 설계 (`fit`, `transform`, `fit_transform`)
- [ ] **데이터 누수 방지 준수**: `fit`은 오직 Train 데이터만으로 수행, 어휘 사전 및 IDF 산출
- [ ] 어휘 사전 키를 알파벳 오름차순으로 정렬하여 열 인덱스 1:1 매핑
- [ ] TF 계산: 문서별 단어 빈도 행렬(`float64`) 생성
- [ ] Smooth IDF 계산: $\\text{idf}(t) = \\ln((1 + N)/(1 + \\text{df}(t))) + 1$
- [ ] **수치 안정성**: L2 Norm 계산 시 0으로 나누기 방어 코드 (`norm == 0`인 경우 영벡터 반환)
- [ ] `verify_tfidf.py` 단계별 정밀 검증 스크립트 작성:
  - [ ] Vocabulary 일치 및 Shape 일치 확인
  - [ ] DF 및 IDF 벡터 오차 검증
  - [ ] 최종 L2 정규화 TF-IDF 행렬 최대 오차 $\\max |X_{\\text{custom}} - X_{\\text{sklearn}}| < 1.0 \\times 10^{-6}$ 검증
  - [ ] 평균 오차(Mean Absolute Error) 로깅

### 🔍 수락 기준 (Acceptance Criteria)
- [ ] Sklearn과의 최대 오차가 $1.0 \\times 10^{-6}$ 미만으로 판정 통과(PASS) 로그 출력
- [ ] Test 문장에 대해 신규 어휘가 배제된 상태로 안정적 transform 수행 확인"""
    },
    {
        "title": "[Issue #3] [Search] 코사인 유사도 검색 엔진, 인덱스 직렬화 및 CLI 구축",
        "labels": ["feature", "phase-3", "cli"],
        "body": """### 🎯 목표
인덱스를 직렬화 저장/로드하고, 코사인 유사도($O(N \\cdot |V|)$) 및 `np.argpartition`을 이용한 고속 Top-K 문서 검색과 과제 규격 CLI를 구현합니다.

### 📝 작업 체크리스트
- [ ] 인덱스 영속화 파이프라인: Train 데이터 기반 인덱스 빌드 및 `artifacts/index.pkl` 저장/로드(`SearchEngine.load()`)
- [ ] 쿼리 벡터 변환: Train 단어 사전 및 IDF를 이용한 L2 정규화 쿼리 벡터 생성 (Zero-vector 방어)
- [ ] 코사인 유사도 벡터 연산: 정규화된 문서 행렬 $D$와 쿼리 벡터 $q$ 간 행렬 곱 ($O(N \\cdot |V|)$ dense 연산)
- [ ] Top-K 검색 가속: `np.argpartition`을 이용한 $O(N + K \\log K)$ 상위 5개 문서 인덱싱
- [ ] 검색 결과 포맷터: 유사도 스코어(소수점 4자리), 문서 ID, 150자 스니펫 출력
- [ ] 과제 필수 인터페이스 구현:
  - [ ] **(안 A) CLI**: `python main.py --query "..." --topk 5`
  - [ ] **(안 B) 함수 시그니처**: `search(query: str, top_k: int = 5) -> List[Tuple[float, int, str]]`

### 🔍 수락 기준 (Acceptance Criteria)
- [ ] 사전에 없는 단어만 입력되었을 때 에러 없이 유사도 0과 빈 스니펫/안내 반환
- [ ] CLI 및 함수 시그니처가 과제 요구사항 포맷과 100% 일치할 것"""
    },
    {
        "title": "[Issue #4] [ML/Evaluation] Logistic Regression 분류 모델 학습 및 BoW 한계 분석",
        "labels": ["machine-learning", "phase-4", "evaluation"],
        "body": """### 🎯 목표
Train TF-IDF 특성으로 `LogisticRegression`을 학습하고 Test 데이터셋에서 정확도/F1을 평가하며, 혼동 행렬 시각화 및 오분류 5건을 BoW 한계와 연계 분석합니다.

### 📝 작업 체크리스트
- [ ] 직접 구현한 TF-IDF로 변환된 $X_{\\text{train}}$으로 `LogisticRegression(random_state=42)` 학습
- [ ] Test 데이터 $X_{\\text{test}}$에 대해 Accuracy 및 Macro-averaged F1-Score 측정
- [ ] Scikit-learn `confusion_matrix` 및 `classification_report` 생성
- [ ] `matplotlib` / `seaborn` 기반 혼동 행렬 시각화 이미지(`assets/confusion_matrix.png`) 저장
- [ ] **오분류 사례 5건 정밀 분석**:
  - [ ] 실제 레이블 vs 예측 레이블 불일치 문서 5건 추출
  - [ ] BoW 본질적 한계(어순 무시, 문맥 소실, 도메인 중의어, 부정어 역전 등)와 연결하여 원인 서술

### 🔍 수락 기준 (Acceptance Criteria)
- [ ] Train/Test 분할 누수 없이 독립된 평가 지표 산출
- [ ] 혼동 행렬 이미지 생성 및 5건 이상의 구체적인 오분류 원인 분석 리포트 완성"""
    },
    {
        "title": "[Issue #5] [Bonus] TF 변형 비교, BM25 엔진 및 역색인(Inverted Index)",
        "labels": ["bonus", "advanced-ir"],
        "body": """### 🎯 목표
Zero-safe 규약을 적용한 TF 변형 비교, BM25 알고리즘 구현(Toy 예제 검증 포함), 그리고 역색인 구조의 동치성 검증 및 속도 벤치마크를 수행합니다.

### 📝 작업 체크리스트
- [ ] **1. 다양한 TF 변형 구현 & 정량 평가**:
  - [ ] Raw Count, Log Normalization $\\ln(1+\\text{tf})$, Double Normalization $0.5 + 0.5 \\frac{\\text{tf}}{\\max(\\text{tf})}$ 구현
  - [ ] **Zero-Safe 규약 준수**: Double Normalization 적용 시 $\\text{tf}=0$인 항목은 0으로 보존
  - [ ] 각 TF 방식별 분류 F1-Score 및 검색 랭킹 변화 정량 비교표 작성
- [ ] **2. BM25 알고리즘 직접 구현**:
  - [ ] 음수 방지형 Robertson-Spärck Jones IDF 및 $k_1=1.5, b=0.75$, $\\text{avgdl}$ 수식 적용
  - [ ] 3개 문서로 구성된 Toy example 수작업 계산 일치성 단위 테스트 통과
  - [ ] 동일 쿼리에 대해 TF-IDF vs BM25 상위 Top-5 검색 결과 비교 분석
- [ ] **3. 역색인(Inverted Index) 구현 & 동치성/성능 벤치마크**:
  - [ ] 단어 $\\rightarrow$ 포스팅 리스트(`doc_id`, `tf`) 구조 구현
  - [ ] **동치성 검증**: 역색인 기반 검색 결과가 전체 스캔(Full-scan) 검색과 동일한 점수·순위를 반환하는지 검증
  - [ ] **벤치마크 분리**: 인덱스 구축 시간과 단일 쿼리 실행 시간(ms)을 분리 측정하여 전체 스캔 대비 속도 비교

### 🔍 수락 기준 (Acceptance Criteria)
- [ ] Toy example 단위 테스트 통과 및 역색인-전체스캔 간 검색 결과 100% 일치
- [ ] TF 변형 비교 및 BM25 비교 분석 결과 리포트 수록"""
    },
    {
        "title": "[Issue #6] [Documentation] 필수 3단 구성 README.md 및 제출 패키징",
        "labels": ["documentation", "phase-5"],
        "body": """### 🎯 목표
과제 제출 기준인 필수 3단 구성(구현 요약, 검증/실험 결과, 한계와 개선 방향) 리포트와 재현 가능한 패키징(`requirements.txt`)을 완성합니다.

### 📝 작업 체크리스트
- [ ] `README.md` 과제 규격 필수 3단 구성 작성:
  - [ ] **1단: 구현 요약**: 전처리 파이프라인, TF-IDF 수학 공식, 검색/분류/보너스 아키텍처
  - [ ] **2단: 검증/실험 결과**: TF-IDF 검증 오차 로그, 분류 성능표, 혼동 행렬 시각화, BM25/역색인 실험표
  - [ ] **3단: 한계와 개선 방향**: BoW 한계 분석, 5개 오분류 원인, 사전학습 언어모델(Transformer) 후속 학습 방향
- [ ] 실행 방법 안내 (CLI 커맨드 예시 및 `search()` 함수 호출 예시)
- [ ] Python 3.10+ 구동용 의존성 명시 (`requirements.txt`)

### 🔍 수락 기준 (Acceptance Criteria)
- [ ] 과제 PDF 가이드라인의 모든 필수 문항 및 결과 양식이 누락 없이 완비될 것
- [ ] 새 가상환경에서 `pip install -r requirements.txt` 후 재현 실행이 완벽할 것"""
    }
]

def create_issue(issue_data):
    req = urllib.request.Request(
        API_URL,
        data=json.dumps({
            "title": issue_data["title"],
            "body": issue_data["body"],
            "labels": issue_data["labels"]
        }).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "A4_1-Issue-Bot"
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"✅ 등록 성공 (#{data['number']}): {data['title']}")
            print(f"   URL: {data['html_url']}")
            return True
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode()
        print(f"❌ 등록 실패: {e.code} - {err_msg}")
        return False
    except Exception as e:
        print(f"❌ 예외 발생: {e}")
        return False

print(f"🚀 https://github.com/{REPO} 에 총 {len(issues)}개 개정 이슈를 등록합니다...\n")
success_count = 0
for issue in issues:
    if create_issue(issue):
        success_count += 1

print(f"\n🎉 완료: 총 {len(issues)}개 중 {success_count}개 등록 완료!")
