# A4_1 실측 결과와 실행 체크표

현재 구현·검증·Git 관리의 검토 경로는 `main` 대상 통합 PR #12입니다. 이전 PR #9·#10은 전체 변경 포함을 확인한 뒤 대체 종료했습니다. Git 관리 회귀 검사를 포함한 전체 검사 50개를 통과했습니다. [통합 PR #12](https://github.com/ntt65/A4_1/pull/12), [Git 관리 실행 체크표](../git_workflow_repair_plan.md)

2026-11-12 로컬 검증 결과입니다. 작업 브랜치는 `feat/a4-1-nlp-verification`이며, 소스 복원과 문서 통합을 이슈 #7에서 추적합니다. 개인정보 전체 검수와 데이터 이용 조건 확인은 이슈 #8에 남아 있습니다. [구현 이슈 #7](https://github.com/ntt65/A4_1/issues/7), [데이터 검수 이슈 #8](https://github.com/ntt65/A4_1/issues/8)

## 1. 실측 수치

| 단계 | 항목 | 기준과 실측 결과 | 상태 |
| --- | --- | --- | --- |
| M1 | 데이터·분할 | 원본 3,894건, 빈 본문 107건 제외, Train 3,029 / Test 758 | 로컬 확인 |
| M2 | TF-IDF 대조 | 허용 기준 1e-6, Train 최대 오차 2.3315e-15, Test 8.3267e-16, 질의 0 | 통과 |
| M2 | 어휘·TF·DF·IDF | 어휘 33,297개, 열 순서·TF·DF 일치, IDF 최대 오차 0 | 통과 |
| M3 | 검색·Top-K | 실제 검색 Top-5 출력, 0점 제외, 동점 문서 ID 순서와 경계 검사 | 통과 |
| M3 | 저장·함수 API | 저장 인덱스 재로딩 후 Top-5·OOV 결과 확인 | 통과 |
| M4 | 분류 | Test 758건 중 701건 정답, Accuracy 92.4802%, Macro-F1 0.925329 | 측정 완료 |
| M4 | 혼동 행렬·오분류 | 이미지 생성, 실제 Test ID 3·5·13·19·25의 본문·기여도 분석 | 작성·확인 완료 |
| M6 | TF 변형 | Raw 92.4802% / Log 92.4802% / Double 91.9525% Accuracy | 측정 완료 |
| M6 | BM25 Toy | 독립 수작업 계산 최대 오차 2.2204e-16, 기준 1e-6 | 통과 |
| M6 | 역색인 | 7개 질의 최대 점수 오차 0, Top-5 순위 100% 일치 | 통과 |
| M6 | 시간 측정 | 구축 0.463321초, 우주 질의 dense 24.0746ms / 역색인 0.3283ms | 재측정 완료 |
| 회귀 검사 | 핵심·보너스·공개 산출물·Git 관리 | `unittest` 50개 | 통과 |
| 데이터 검수 | 인명·서명 잔존과 이용·재배포 조건 | 배포 조건과 자동 검사 확인, 전체 인명 검수 대기 | 이슈 #8 진행 중 |

시간은 BLAS 2스레드, 워밍업 후 3회 중앙값이며 토큰화·벡터화 시간을 제외한 점수 계산·Top-5 선택을 측정했습니다. 역색인 순위 일치는 명시된 7개 질의의 Top-5 기준입니다. [보너스 측정 근거](../../artifacts/bonus_metrics.json)

## 2. 실행 명령과 증거

| 명령 | 종료 코드 | 출력 |
| --- | --- | --- |
| `python verify_tfidf.py` | 0 | [TF-IDF 출력](../../artifacts/run_logs/verify_tfidf.txt) |
| `python main.py --query "space satellite launch orbit" --topk 5` | 0 | [검색 출력](../../artifacts/run_logs/search.txt) |
| `python main.py --train-classifier` | 0 | [분류 출력](../../artifacts/run_logs/classifier.txt) |
| `python main.py --bonus` | 0 | [보너스 출력](../../artifacts/run_logs/bonus.txt) |

[명령별 실행 기록](../../artifacts/verification_results.json)에 실행 시각, Python·패키지 버전, 종료 코드와 회귀 검사 결과를 기록했습니다. [데이터 메타데이터](../../artifacts/dataset_metadata.json), [TF-IDF 검증 JSON](../../artifacts/tfidf_verification.json), [분류 지표](../../artifacts/classifier_metrics.json), [혼동 행렬](../../artifacts/confusion_matrix.png)을 함께 대조할 수 있습니다.

## 3. 실행 체크표

- [x] 사용자 제공 Python 소스를 프로젝트 루트와 `src/`에 복원
- [x] 누락 import·패키지 초기화 파일명·실패 종료 코드 보완
- [x] Python 3.14.7 가상환경에서 의존성 로딩·호환성 확인
- [x] 헤더·푸터·인용 제거, 이메일·일반 전화번호 마스킹, 빈 본문 제외
- [x] 층화 8:2 분할과 `random_state=42`, 클래스 분포·코퍼스 해시 기록
- [x] Train 전용 어휘·IDF와 Test·질의 변환 분리 확인
- [x] 공통 토큰화·고정 불용어·정렬된 어휘 열과 sklearn 설정 대조
- [x] 빈 입력·불용어 전용 입력·OOV 영벡터 처리 확인
- [x] 양수 점수만 반환하고 Top-K 경계 동점과 문서 ID 순서 확인
- [x] CLI 검색과 `main.search()` 함수 호출 확인
- [x] `--build-index` 저장 후 재로딩 확인
- [x] 정확도·Macro-F1·클래스별 지표·혼동 행렬 생성
- [x] 실제 오분류 5건과 클래스 기여 단어·절편 효과 분석
- [x] Raw·Log·Double TF 분류와 검색 순위 변화 비교
- [x] BM25 Toy 수작업 대조와 같은 질의의 TF-IDF·BM25 결과 비교
- [x] 역색인 점수·Top-5 동치성과 구축·질의 시간 분리 측정
- [x] 핵심·보너스·공개 산출물·Git 관리 회귀 검사 50개 통과
- [x] README 필수 3단 구성과 하단 보너스 섹션 작성
- [x] 평가표 번호를 연결한 Q&A와 발표 개요 작성
- [x] 가상환경·데이터 캐시·생성 인덱스의 Git 제외 규칙 설정
- [ ] 전체 코퍼스와 산출물의 인명·서명 등 식별 정보 검수 완료
- [ ] 필요한 문서의 비식별화·제외 후 재검사와 수치 재측정
- [x] 실제 Figshare 배포 항목의 CC BY 4.0 표기·근거 URL·확인일 기록
- [x] 불용어 목록의 프로젝트 기준 버전·선정 근거·정규화 해시 기록

미완료 항목의 완료 기준과 후속 작업은 데이터 검수 이슈에 기록했습니다. [후속 체크표 #8](https://github.com/ntt65/A4_1/issues/8)

## 4. GitHub 전달 체크표

- [x] 작업 브랜치 `feat/a4-1-nlp-verification` 생성
- [x] 기존 단계별 이슈 #1~#6 확인
- [x] 소스 복원·실측 검증·문서 통합 이슈 #7 발행
- [x] 남은 데이터 검수와 이용 조건 확인 이슈 #8 발행
- [x] 개요·상세 계획·실행 체크표에 실제 수치와 이슈 연결
- [x] 검토할 코드·문서·검증 산출물 커밋 e3ff7e8
- [x] 작업 브랜치 원격 푸시
- [x] `main` 대상 PR #9 생성 및 이슈·검증 근거 연결

구현 커밋과 원격 브랜치, PR 생성 결과를 확인해 전달 항목을 완료 처리했습니다. 공개 측정 JSON과 검색 출력에는 원문 본문·스니펫·원본 기여 단어 필드를 제외했습니다. 전체 코퍼스 검수는 #8에서 추적합니다. [구현 커밋 e3ff7e8](https://github.com/ntt65/A4_1/commit/e3ff7e8caf528324e099eb580665c9a5756cb53a), [PR #9](https://github.com/ntt65/A4_1/pull/9)

[개요와 아키텍처](docs_01_system_architecture_and_plan.md), [상세 구현 계획](../../IMPLEMENTATION_PLAN.md)

## 5. 이슈 #8 후속 실행 체크표

- [x] 구현 수정 전 `feat/a4-1-data-review` 브랜치 생성
- [x] 실행 계획을 첫 커밋 8979244로 기록
- [x] 데이터 3,787건의 이메일·전화번호·대문자 조합 후보 자동 검사
- [x] 검토용 원문을 Git 제외 경로에 저장하고 공개 보고서에 집계만 기록
- [x] 분류·보너스 지표 생성기에 원문 비저장 스키마 2 반영
- [x] 분류·보너스 실제 재실행과 수치 일치 확인
- [x] 공개 필드 검사·패키지 검사·불용어 해시 회귀 검사
- [x] 데이터 출처와 라이선스 표기·변경 내용·불용어 채택 근거 기록
- [ ] 전체 인명·서명 검수와 필요한 비식별화·제외 후 재검사
- [x] 후속 구현 커밋 78c399c·푸시·PR #10 연결

자동 검사에서 이메일과 지정 전화번호 패턴은 0건이며, 인명 또는 제목 후보는 18,426개가 2,292개 문서에서 검출됐습니다. 제목·기관명 등이 포함되고 소문자 인명 등은 놓칠 수 있으므로 수동 검수 대상으로 관리합니다. [후속 계획](../A4_1_data_review_plan.md), [자동 검사 보고서](../../artifacts/privacy_audit.json), [출처와 이용 조건](../A4_1_data_sources.md)

후속 PR은 기존 구현 브랜치를 기준으로 생성했습니다. [PR #10](https://github.com/ntt65/A4_1/pull/10), [구현 커밋 78c399c](https://github.com/ntt65/A4_1/commit/78c399c83fae43a347ebc0f761c26e57fba6a3e7)
