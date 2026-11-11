# A4_1 구현 개요와 아키텍처

2026-11-12 사용자 제공 Python 소스를 복원하고 로컬에서 실행한 상태를 반영했습니다. 작업 브랜치는 `feat/a4-1-nlp-verification`입니다. 소스 복원·실측 검증·문서 통합은 이슈 #7에서 추적하고, 개인정보 잔존 검수와 데이터 이용 조건 확인은 이슈 #8에서 추적합니다. [구현 이슈 #7](https://github.com/ntt65/A4_1/issues/7), [데이터 검수 이슈 #8](https://github.com/ntt65/A4_1/issues/8)

## 1. 프로젝트 개요

20 Newsgroups 네 카테고리의 텍스트를 NumPy TF-IDF로 표현해 코사인 검색과 Logistic Regression 분류를 수행합니다. 미션의 수치 검증 기준은 sklearn 대비 최대 절대 오차 `1e-6` 이내입니다.

| 항목 | 확인된 상태 |
| --- | --- |
| 데이터 | 원본 3,894건에서 빈 본문 107건 제외, 총 3,787건 |
| 분할 | 층화 8:2, `random_state=42`, Train 3,029건 / Test 758건 |
| 학습 어휘 | Train에서 학습한 33,297개 |
| TF-IDF 검증 | Train 최대 오차 2.3315e-15, Test 8.3267e-16 |
| 분류 | Accuracy 92.4802%, Macro-F1 0.925329 |
| 실행 환경 | Windows, Python 3.14.7, NumPy 2.5.4, scikit-learn 1.9.1 |
| 회귀 검사 | 29개 통과 |
| 추가 검수 | 배포 조건·불용어 채택 근거 확인, 전체 코퍼스 인명·서명 검수 대기 |

수치의 근거는 [데이터 메타데이터](../../artifacts/dataset_metadata.json), [TF-IDF 검증](../../artifacts/tfidf_verification.json), [분류 지표](../../artifacts/classifier_metrics.json), [명령별 실행 기록](../../artifacts/verification_results.json)에 있습니다.

## 2. 실제 모듈 구조

```text
src/dataset.py
    다운로드·캐시 → 헤더·푸터·인용 제거 → 이메일·전화번호 마스킹
    → 빈 본문 제외 → 층화 Train/Test 분할
                         ↓
src/preprocess.py       소문자화·정규식 토큰화·고정 불용어 제거
                         ↓
src/tfidf.py            Train 전용 어휘·DF·IDF 학습과 NumPy 변환
                         ↓
              ┌──────────┴────────────┐
src/search_engine.py              src/classifier.py
코사인·Top-K·저장·로드             Logistic Regression·Test 평가
              └──────────┬────────────┘
                       main.py
                  CLI와 search() 함수

verify_tfidf.py          sklearn 단계별 대조
src/bonus/              TF 변형·BM25·역색인 비교
```

전처리·벡터화·검색·분류를 나눠 같은 토큰 규칙과 벡터 표현을 재사용하고, 단계별 계산을 독립적으로 확인합니다. 실행 진입점은 [main.py](../../main.py)이며 함수 호출은 `from main import search`를 지원합니다. [상세 실행 방법](../../README.md)

## 3. 핵심 설계와 검증 원칙

1. Train에서 어휘·IDF를 학습하고 Test와 질의에는 `transform`만 적용합니다. 분할은 모든 비교 실험에서 같은 시드와 층화 조건을 사용합니다.
2. 양쪽 구현에서 `TextPreprocessor.tokenize`를 analyzer로 공유합니다. `smooth_idf=True`, `sublinear_tf=False`, `norm='l2'`, `float64`와 알파벳 순 어휘를 일치시킵니다.
3. `headers`, `footers`, `quotes`를 제거하고 이메일·일반 전화번호 형식을 마스킹합니다. 실제 결과에 인명·서명이 남아 있어 전체 개인정보 검수는 진행 대기 상태입니다.
4. 빈 문서·OOV의 L2 분모를 방어하고 영벡터를 유지합니다. Double Normalization도 0 빈도를 0으로 보존합니다.
5. Dense 점수 계산은 `O(N × |V|)`입니다. 양수 점수만 반환하고 점수 내림차순, 동점 문서 ID 오름차순을 적용합니다. 부분 선택에서 경계 동점을 포함해 순서를 확정합니다.
6. BM25는 `k1=1.5`, `b=0.75`와 양수 RSJ IDF를 사용합니다. 독립 수작업 Toy 계산과 비교하고 역색인은 같은 질의·Top-K 정책으로 전체 스캔과 대조합니다.
7. [requirements.txt](../../requirements.txt)에 검증 환경의 의존성을 고정하고, 코퍼스 해시·분할 분포·실행 시각·명령 종료 코드를 산출물에 기록합니다.
8. README는 구현 요약, 실험 결과, BoW 한계·실제 오분류 분석의 3단 구성을 유지합니다. 보너스는 README 하단의 별도 섹션에 배치합니다.

불용어는 [고정 파일](../../resources/stopwords_en.txt)의 123개를 사용합니다. 원출처·버전은 제공 자료에 없으며 `no`, `nor`, `not` 제거에 따른 의미 손실을 분석 문서에 기록했습니다. 원출처와 선정 근거 보완은 이슈 #8의 체크 항목입니다.

## 4. 실행 계획과 완료 상태

실행 순서는 `M1 → M2 → M3·M4 → M6 → M5`입니다. 기존 GitHub 이슈 번호는 유지하며 M5는 #5, M6는 #6에 대응합니다.

| 단계 | 실행 내용 | 산출물 | 상태 |
| --- | --- | --- | --- |
| M1 / #1 | 데이터·전처리·고정 분할 | `src/dataset.py`, `src/preprocess.py`, `resources/stopwords_en.txt`, 데이터 메타데이터 | 코드와 실측 완료, 데이터 검수 #8 진행 대기 |
| M2 / #2 | NumPy TF-IDF와 sklearn 대조 | `src/tfidf.py`, `verify_tfidf.py`, 검증 JSON·출력 | 완료 |
| M3 / #3 | 코사인 검색·CLI·함수·인덱스 저장 | `src/search_engine.py`, `main.py`, 검색 출력 | 완료 |
| M4 / #4 | 분류·혼동 행렬·실제 오분류 5건 | `src/classifier.py`, 분류 지표, 혼동 행렬, README 분석 | 완료 |
| M6 / #6 | Raw·Log·Double TF·BM25·역색인 | `src/bonus/`, 보너스 지표·출력 | 완료 |
| M5 / #5 | 평가 문서·실행 안내·패키지 | README, Q&A, 발표 개요, requirements | 로컬 작성·검증 완료 |
| GitHub 작업 / #7 | 작업 브랜치·이슈·커밋·PR 연결 | 구현 커밋 e3ff7e8, 작업 브랜치 원격 푸시, main 대상 PR #9 | 전달 완료, PR 검토 진행 |

기존 이슈 #1~#6은 단계별 요구사항의 근거이며, 해당 이슈의 모든 수락 기준을 완료했다고 일괄 판정하지 않습니다. 소스 통합 이슈 #7과 데이터 검수 이슈 #8을 연결해 실제 완료 근거를 추적합니다.

## 5. 검토할 문서

후속 작업은 `feat/a4-1-data-review` 브랜치에서 첫 구현 수정 전에 계획 커밋 8979244를 기록했습니다. 데이터 3,787건 자동 검사와 공개 지표 생성 경로를 보완하고 정확한 Figshare 배포 항목의 CC BY 4.0 표기를 확인했습니다. [후속 실행 계획](../A4_1_data_review_plan.md), [출처 보고서](../A4_1_data_sources.md), [자동 검사 집계](../../artifacts/privacy_audit.json)

구현과 실측 산출물의 변경 내역을 PR에서 검토할 수 있습니다. [구현 커밋 e3ff7e8](https://github.com/ntt65/A4_1/commit/e3ff7e8caf528324e099eb580665c9a5756cb53a), [PR #9](https://github.com/ntt65/A4_1/pull/9)

- [상세 구현 계획](../../IMPLEMENTATION_PLAN.md)
- [실행 체크표와 실측 결과](docs_03_verification_and_checklists.md)
- [README](../../README.md)
- [평가 대비 Q&A](../A4_1_mission_QnA.md)
- [발표 개요](../A4_1_presentation_outline.md)
