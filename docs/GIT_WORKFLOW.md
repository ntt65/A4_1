# 로컬 Git 작업과 이슈 추적

이 문서는 이슈 #11의 Git 관리 정리 기준입니다. 작업 브랜치는 `chore/a4-1-git-workflow`이며 실행 계획은 구현 수정 전에 커밋 ee933fa로 기록했습니다. [정리 이슈 #11](https://github.com/ntt65/A4_1/issues/11), [실행 계획](git_workflow_repair_plan.md)

## 관리 주체

| 대상 | 담당과 절차 |
| --- | --- |
| 코드·평가 문서·Git 변경 | 로컬 VS Code/Antigravity 작업 트리에서 변경 내용을 확인하고 이슈와 연결해 커밋 |
| AI Studio | 화면 UI 프로토타입과 아이디어 확인, 로컬로 가져올 소스의 파일별 변경 검토 |
| 이슈·PR | GitHub에서 요구사항·검증 근거·검토 상태 기록 |
| 같은 작업 트리의 Git 명령 | 현재 작업 담당자가 브랜치·스테이징·커밋·푸시를 수행 |

AI Studio의 GitHub 계정 연결과 자동 동기화 설정은 서비스의 연동 설정에서 관리합니다. 로컬 작업 지침과 후크는 해당 작업 트리의 Git 실행에 적용됩니다. 2026-11-12에 해당 프로젝트의 GitHub 저장소 연결을 해제하고 표시 상태를 확인했습니다. 구체적인 확인 내용은 아래에 기록했습니다.

## 작업 시작부터 PR까지

1. 현재 브랜치, 변경 파일, 원격 참조와 관련 이슈·PR을 읽기 전용으로 확인합니다.
2. 여러 파일에 걸친 작업은 구현·문서 수정 전에 작업 브랜치를 만듭니다.
3. 기존 이슈를 재사용하거나 허용된 범위에서 작업 이슈를 발행합니다.
4. 개요·실행 계획·완료 기준·체크표를 작성하고 첫 계획 커밋에 이슈 번호를 넣습니다.
5. 구현과 필요한 검증을 수행하고 확인된 결과로 체크표를 갱신합니다.
6. 파일별 변경 범위를 확인해 커밋·푸시하고 PR에 이슈와 실행 근거를 연결합니다.
7. PR 생성 결과를 확인한 뒤 계획과 전달 체크표에 실제 링크를 기록합니다.

작업 상태 확인에는 다음 명령을 사용합니다.

```bash
git status --short
git branch -vv
git log --graph --decorate --oneline --all -20
```

미추적 파일과 다른 작업의 변경은 먼저 소유와 목적을 확인합니다. 이 정리 작업을 시작할 때의 `gemini-code-1794436836670.sh`는 사용자 소유 미추적 파일로 기록했습니다.

## 커밋 메시지

제목은 `type(scope): 작업 내용` 또는 `type: 작업 내용`을 사용하고 제목이나 본문에 양의 정수 이슈 번호를 연결합니다. 기능 변경과 검증 목적을 읽을 수 있는 구체적인 내용을 적습니다.

```text
feat(tfidf): implement smooth IDF verification (#2)
fix(search): order boundary ties by document ID (#3)
docs: connect execution evidence to the repair plan (#11)
```

본문의 참조도 사용할 수 있습니다.

```text
test: verify numeric public artifact exports

Refs #8
```

허용하는 타입은 `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `build`, `ci`, `perf`, `style`, `revert`입니다. 선택적인 `(scope)`와 breaking-change 표시 `!`를 지원합니다. Git 주석 줄은 참조 판정에서 제외합니다.

Git이 생성하는 `Merge pull request #N`, `Merge branch`, `Merge remote-tracking branch`, `Merge tag` 형식도 처리합니다. 과거 커밋의 목적과 산출물은 [공개 이력 추적표](git_history_map.md)에 연결했습니다.

로컬 후크는 다음 명령으로 이 작업 트리에 설치합니다.

```bash
git config --local core.hooksPath .githooks
git config --local --get core.hooksPath
```

메시지 파일을 독립 검사할 수도 있습니다.

```bash
python scripts/check_commit_message.py COMMIT_MESSAGE.txt
```

검사 결과와 실제 후크 설정 여부는 [정리 실행 체크표](git_workflow_repair_plan.md)에서 확인합니다.

메시지 검사와 실제 `git hook run`의 정상·거부 동작은 다음 회귀 검사로 확인합니다. 후크 파일은 실행 가능 모드와 LF 개행을 적용해 버전 관리합니다.

```bash
python -m unittest discover -s tests -p test_commit_message.py -v
```

## 이슈 정의와 동기화

단계별 정의는 [github_issues.json](github_issues.json)에 있습니다. 번호는 #1=M1 데이터·전처리, #2=M2 TF-IDF, #3=M3 검색, #4=M4 분류, #5=M5 문서·패키징, #6=M6 보너스로 연결합니다.

기본 실행은 카탈로그를 오프라인으로 미리 봅니다.

```bash
python create_github_issues.py
```

GitHub에 반영할 때는 해당 저장소에 권한이 있는 `GITHUB_TOKEN` 환경변수를 준비한 뒤 명시적으로 실행합니다.

```bash
python create_github_issues.py --apply
```

도구는 전체 상태의 기존 이슈를 페이지별로 읽고 PR을 제외합니다. `<!-- a4-1:M1 -->` 같은 단계 표식을 먼저 대조하며, 기존 번호와 `[Issue #N]` 제목 접두사 또는 일치하는 제목으로 초기 이슈를 연결합니다. 일치하는 이슈는 변경된 제목·본문을 갱신하고, 누락된 단계는 표식을 포함해 생성합니다. 동일한 정의를 재실행하면 변경이 없는 이슈를 건너뜁니다.

상태·레이블·담당자는 GitHub의 실제 값을 사용합니다. 구현이 검증된 항목만 완료 체크하고, 전체 데이터 인명·서명 검수는 #8에 남깁니다. [데이터 검수 이슈 #8](https://github.com/ntt65/A4_1/issues/8)

## PR과 서비스 연결 점검

2026-11-12 정리 시작 시점에는 PR #9가 `main`을 대상으로 하고 PR #10이 #9의 구현 브랜치를 대상으로 했습니다. 각 브랜치는 원격과 동기화돼 있었습니다. 통합 검토 경로를 만든 뒤 기존 PR과의 관계를 실행 체크표에 기록합니다. [PR #9](https://github.com/ntt65/A4_1/pull/9), [PR #10](https://github.com/ntt65/A4_1/pull/10)

AI Studio 프로젝트의 Settings → GitHub에서 `ntt65/A4_1`의 `main` 연결, 마지막 동기화 시각 2026-11-12 07:27 KST, 푸시 대기 변경 파일 24개를 확인했습니다. 이 화면에서 해당 프로젝트의 저장소 연결에 `Disconnect`를 적용했습니다.

적용 후 `Sync to GitHub`와 `Create new repository`가 표시되고 기존 저장소·`main`·푸시 제어가 사라진 상태를 확인했습니다. GitHub 저장소에는 계속 접근할 수 있으며 로컬 참조와 원격의 동기화 상태도 대조했습니다. UI 프로토타입의 변경을 로컬로 가져올 때는 파일별 차이를 확인하고 로컬 작업 브랜치에서 이슈·커밋·PR로 연결합니다.

현재 정리 범위는 기존 공개 이력을 보존한 새 커밋과 이력 대조표입니다. `main` 대상 통합 PR을 확인한 후 PR #9·#10의 대체 관계를 기록하는 순서로 진행합니다.

- [x] 로컬·원격 브랜치의 동기화 상태 확인
- [x] 커밋·이슈·산출물의 추적 문서 작성
- [x] AI Studio 프로젝트의 실제 GitHub 저장소·브랜치·푸시 대기 상태 확인
- [x] 해당 프로젝트의 GitHub 저장소 연결 해제와 화면 결과 확인
- [x] 공개 이력의 정리 범위를 새 커밋과 추적 문서로 기록
- [ ] 통합 PR 생성과 기존 PR 관계 기록

`Loading file differences` 화면의 내부 원인 분석은 미확인 상태입니다. 이번 작업에서는 AI Studio 프로젝트의 원격 저장소·브랜치·푸시 제어가 해제된 화면과 로컬·원격 참조의 일치를 확인했습니다.
