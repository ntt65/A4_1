# 공개 Git 이력 추적표

2026-11-12 정리 시작 시점의 `git log`, 변경 경로, 로컬·원격 참조를 대조했습니다. 커밋의 작성자 표시는 `ntt65`와 `mpegdj`로 나타납니다. 커밋 메시지와 작성자 정보만으로 실행한 클라이언트를 특정할 수 없어 도구의 작성 주체는 별도 확인 대상으로 기록합니다.

## 정리 시작 시점의 브랜치와 PR

| 참조 | 커밋 | 원격 대조 | 검토 경로 |
| --- | --- | --- | --- |
| main | ee4d315 | origin/main과 일치 | 기본 브랜치 |
| feat/a4-1-nlp-verification | 07f1915 | origin의 같은 브랜치와 일치 | PR #9 → main |
| feat/a4-1-data-review | 5e76937 | origin의 같은 브랜치와 일치 | PR #10 → feat/a4-1-nlp-verification |
| chore/a4-1-git-workflow | ee933fa | 첫 로컬 계획 커밋 | 이슈 #11의 정리 작업 |

복원한 Python 코드는 e3ff7e8에서 추가됐습니다. 정리 시작 시점의 기본 브랜치 ee4d315에는 해당 구현이 반영되기 전이었고, 두 구현 브랜치는 각각 원격과 동기화돼 있었습니다. [PR #9](https://github.com/ntt65/A4_1/pull/9), [PR #10](https://github.com/ntt65/A4_1/pull/10)

같은 날 AI Studio 프로젝트 화면에서는 `ntt65/A4_1`의 `main` 연결과 마지막 동기화 시각 07:27 KST, 푸시 대기 변경 파일 24개를 확인했습니다. Settings → GitHub → Disconnect로 해당 프로젝트의 저장소 연결을 해제한 뒤 `Sync to GitHub`·`Create new repository`가 표시되는 상태를 확인했습니다. GitHub 저장소 접근과 로컬·원격 참조 점검 결과를 함께 기록했습니다.

## 기본 브랜치의 초기 이력

| 커밋 | 당시 메시지 또는 요약 | 확인한 주요 변경 경로 | 추적할 목적 |
| --- | --- | --- | --- |
| dbb47d5 | Initial commit | README.md | 저장소 시작 |
| c246185 | feat: initialize project scaffolding | React/Vite 파일, 구현 계획, 이슈 생성 스크립트, 미션 PDF | 초기 화면·계획·도구 구성 |
| 6ff9d8a | docs: update implementation plan for A4_1 project | IMPLEMENTATION_PLAN.md, .aistudio 계획 파일 | 구현 계획 작성 |
| da87e8a | docs: update implementation plan and issue tracking | 구현 계획, .aistudio 계획 파일, create_github_issues.py | 계획과 이슈 정의 연결 |
| c15b234 | 구현 계획 요구사항 보완을 설명하는 장문 메시지 | IMPLEMENTATION_PLAN.md | 이슈 #5 문서화·#6 보너스 번호와 검증 요구사항 정리 |
| caa7388 | docs : create mission | docs/A4_1_mission.pdf | 미션 원문 확보 |
| cbf33e5 | docs : eval | docs/A4_1_eval.pdf | 평가 원문 확보 |
| cb0cac5 | reference site of github | docs/ref_github.md | 참고 저장소 링크 기록 |
| d31c75d | ai studio log | docs/repo_readme 문서 4개, 발표 개요 | 선행 설계·프롬프트·검증 요약·진행 이력 기록 |
| ee4d315 | docs : make readme.md requirement.txt mission_QnA.md | README.md, requirements.txt, docs/A4_1_mission_QnA.md | 실행·평가 문서 초안 작성 |

각 요약은 해당 커밋의 변경 파일을 기준으로 했습니다. `d31c75d`의 검증 요약에 있던 Accuracy 97.56% 등은 선행 문서의 보고값이며, 이후 실제 실행에서 확인한 수치는 README와 측정 산출물에 기록했습니다.

## 구현 복원 이후의 이슈 연결

| 커밋 | 작업 | 이슈·산출물 근거 |
| --- | --- | --- |
| e3ff7e8 | Python 파이프라인 복원, 실행 결함 수정, 실측 검증·문서 통합 | #7, 커밋 본문에서 #1~#6 참조, main.py·verify_tfidf.py·src·tests·artifacts |
| 07f1915 | PR #9와 전달 완료 체크 연결 | #7, 개요·계획·체크표·진행 이력 |
| 8979244 | 데이터 검수 구현 전 계획과 작업 순서 기록 | #8, AGENTS.md·데이터 검수 계획 |
| 78c399c | 공개 지표 생성 경로, 출처·이용 조건, 로컬 자동 검사 보완 | #8, audit_data.py·package_project.py·출처 보고서·공개 스키마 2 |
| 5e76937 | PR #10과 전달 결과 기록 | #8, 데이터 검수 계획·실행 체크표 |
| ee933fa | Git 관리 정리의 개요·계획·완료 기준 기록 | #11, git_workflow_repair_plan.md |

이슈 #1~#6의 번호는 기존 요구사항을 따라 유지합니다. #5는 문서·패키징, #6은 보너스입니다. 이슈 #8의 전체 인명·서명 검수는 미완료 상태로 추적합니다. [통합 이슈 #7](https://github.com/ntt65/A4_1/issues/7), [검수 이슈 #8](https://github.com/ntt65/A4_1/issues/8), [정리 이슈 #11](https://github.com/ntt65/A4_1/issues/11)

## 공개 이력 정리 상태

- [x] 브랜치별 기준 커밋과 원격 동기화 상태 대조
- [x] 초기 커밋의 실제 변경 파일과 목적 연결
- [x] 구현 이후 커밋을 이슈·산출물과 연결
- [x] AI Studio 프로젝트의 저장소 연결 해제와 화면 결과 기록
- [x] 기존 공개 이력을 보존하는 새 커밋·추적표 중심의 정리 범위 기록
- [ ] 통합 PR의 번호와 기존 PR 정리 결과 기록

현재 작업은 기존 공개 이력을 보존하고 새 변경을 이슈 번호가 연결된 커밋으로 기록합니다. `main` 대상 통합 PR이 생성되고 변경 반영을 확인한 뒤 기존 PR #9·#10의 대체 관계를 갱신합니다. 현재 정리 작업의 판단과 최종 PR은 [실행 계획](git_workflow_repair_plan.md)에 기록합니다. [로컬 Git 작업 절차](GIT_WORKFLOW.md)
