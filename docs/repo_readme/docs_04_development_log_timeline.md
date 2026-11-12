# A4_1 개발 진행 이력

## 2026-11-12 단일 노트북 후속 작업

`feat/a4-1-single-notebook` 브랜치와 이슈 #13을 만든 뒤 개요·실행 계획·체크표를 첫 커밋 8b127f5로 기록했습니다. 선행 PR #12의 구현을 바탕으로 `A4_1_mission.ipynb`에 불용어·알고리즘·실험·설명을 내장했습니다. 노트북 파일 하나만 있는 임시 폴더와 별도 IPython 프로필의 새 커널에서 28개 셀을 모두 실행했고 최종 실행은 47.323초였습니다.

TF-IDF Train 최대 오차 2.3315e-15, Accuracy 92.4802%, Macro-F1 0.925329를 재계산했습니다. 혼동 행렬을 출력에 내장하고 실제 오분류 5건·Q1~Q5·TF 변형·BM25·두 역색인 검증을 포함했습니다. 전체 회귀 검사 59개와 저장 출력·실행 해시 검사를 통과했습니다. [계획과 실행 결과](../A4_1_notebook_plan.md), [실행 기록](../../artifacts/notebook_verification.json), [이슈 #13](https://github.com/ntt65/A4_1/issues/13)

## 2026-11-12 로컬 구현 및 실측 검증

| 단계 | 수행 내용 | 결과 |
| --- | --- | --- |
| 소스 확보 | 사용자 제공 텍스트에서 Python 소스와 불용어 목록 복원 | main.py, verify_tfidf.py, src 모듈, resources 구성 |
| 환경 복구 | 불완전한 가상환경 복구, uv 업데이트, 의존성 설치 | Python 3.14.7, uv 0.13.7, 의존성 호환성 확인 |
| 소스 보완 | import·초기화 파일명·리소스 경로·실패 종료 코드 수정 | CLI와 함수 API 정상 실행 |
| 데이터 준비 | 헤더·푸터·인용 제거, 메일·전화번호 마스킹, 빈 본문 제외 | 3,894건 중 107건 제외, Train 3,029 / Test 758 |
| TF-IDF 대조 | 어휘·TF·DF·IDF와 Train/Test/질의 벡터 비교 | Train 최대 절대 오차 2.3315e-15 |
| 검색 검증 | Top-K 동점·0점 처리, 실제 질의 실행, 저장 인덱스 재로딩 | Top-5와 함수 API 확인 |
| 분류 평가 | Logistic Regression 학습, 혼동 행렬과 실제 오분류 5건 분석 | Accuracy 92.4802%, Macro-F1 0.925329 |
| 보너스 검증 | Raw·Log·Double TF, BM25 수작업 비교, 역색인 비교 | Toy 오차 2.2204e-16, 7개 질의 Top-5 순위 일치 |
| 회귀 검사 | 전처리·TF-IDF·검색·인덱스·보너스 경계 조건 | 23개 통과 |
| 산출물 정리 | README, Q&A, 발표 개요, 실측 JSON·로그·혼동 행렬·소스 ZIP | 문서 링크·코드 예제·ZIP 검사 통과 |
| GitHub 작업 | 작업 브랜치, 통합·후속 검수 이슈, 구현 커밋, 원격 푸시와 PR 생성 | feat/a4-1-nlp-verification, #7, #8, e3ff7e8, PR #9 |

실행 시각과 환경, 명령별 결과는 [검증 기록](../../artifacts/verification_results.json)에 있습니다. 이전 AI Studio 문서에 기록된 97.56%, 0.9758 및 1.1102e-15는 선행 보고값으로 보존된 이력이며, 이번 로컬 실측값은 위 표와 README에 기록했습니다.

## 후속 작업

이슈 #8은 `feat/a4-1-data-review` 브랜치를 생성하고 실행 계획을 첫 커밋 8979244로 기록한 뒤 진행했습니다. 정확한 Figshare 배포 항목의 CC BY 4.0 표기와 불용어 목록의 프로젝트 선정 근거를 확인했습니다. 분류·보너스 공개 JSON 생성 경로와 ZIP 검사를 보완했고 후속 회귀 검사 30개를 통과했습니다.

사용 문서 3,787건의 자동 검사 집계를 기록했으며 전체 코퍼스의 인명·서명 수동 검수는 남아 있습니다. [후속 실행 계획](../A4_1_data_review_plan.md), [데이터 출처 보고서](../A4_1_data_sources.md), [자동 검사 집계](../../artifacts/privacy_audit.json), [검수 이슈 #8](https://github.com/ntt65/A4_1/issues/8)

커밋·푸시·PR 진행 상태는 [실행 체크표](docs_03_verification_and_checklists.md)에 기록합니다.

구현 커밋과 PR을 통해 변경 내역을 검토할 수 있습니다. [구현 커밋 e3ff7e8](https://github.com/ntt65/A4_1/commit/e3ff7e8caf528324e099eb580665c9a5756cb53a), [PR #9](https://github.com/ntt65/A4_1/pull/9)
