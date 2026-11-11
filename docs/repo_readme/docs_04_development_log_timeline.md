# A4_1 개발 진행 이력

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
| GitHub 작업 | 작업 브랜치와 통합·후속 검수 이슈 생성 | feat/a4-1-nlp-verification, #7, #8 |

실행 시각과 환경, 명령별 결과는 [검증 기록](../../artifacts/verification_results.json)에 있습니다. 이전 AI Studio 문서에 기록된 97.56%, 0.9758 및 1.1102e-15는 선행 보고값으로 보존된 이력이며, 이번 로컬 실측값은 위 표와 README에 기록했습니다.

## 후속 작업

전체 코퍼스의 인명·서명 검수와 데이터 이용·재배포 조건 확인을 진행해야 합니다. 코드와 측정 산출물 검토는 통합 이슈에서, 남은 데이터 검수는 후속 이슈에서 추적합니다. [통합 이슈 #7](https://github.com/ntt65/A4_1/issues/7), [검수 이슈 #8](https://github.com/ntt65/A4_1/issues/8)

커밋·푸시·PR 진행 상태는 [실행 체크표](docs_03_verification_and_checklists.md)에 기록합니다.
