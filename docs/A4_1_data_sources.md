# A4_1 데이터와 불용어 출처 확인

확인일은 2026-11-12입니다. 이 문서는 이슈 #8에서 사용 데이터의 배포 메타데이터, 이용 조건과 고정 불용어 선정 근거를 기록합니다. [데이터 검수 이슈 #8](https://github.com/ntt65/A4_1/issues/8)

## 실제 다운로드 대상

로컬 scikit-learn 1.9.1의 `sklearn.datasets._twenty_newsgroups.ARCHIVE`와 같은 버전의 공식 소스를 대조했습니다. `fetch_20newsgroups`는 `20news-bydate.tar.gz`를 사용합니다. 전체 bydate 자료는 18,846개 게시물로 구성되며, 프로젝트는 네 카테고리를 선택한 후 별도의 고정 8:2 분할을 적용합니다. [scikit-learn 1.9.1 로더 소스](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.9.1/sklearn/datasets/_twenty_newsgroups.py)

| 확인 항목 | 값 |
| --- | --- |
| 파일명 | `20news-bydate.tar.gz` |
| sklearn 다운로드 URL | `https://ndownloader.figshare.com/files/5975967` |
| sklearn 기대 SHA-256 | `8f1b2514ca22a5ade8fbb9cfa5727df95fa587f4c87b786e15c759fa66d95610` |
| Figshare article / version | `3829962` / `2` |
| Figshare file ID | `5975967` |
| Figshare 파일 크기 | 14,464,277 bytes |
| Figshare supplied / computed MD5 | `d6e9e45cb8cb77ec5276dfa6dfc14318` |
| Figshare DOI | `10.6084/m9.figshare.3829962.v2` |
| Figshare 등록 저자 | Nelson Liu |
| Figshare 게시일 | 2016-09-15 |
| Figshare license 표시 | CC BY 4.0 |

위 Figshare 값은 공식 API 응답에서 확인했습니다. 파일 ID와 다운로드 URL이 설치된 sklearn의 대상과 일치합니다. [Figshare 공식 메타데이터](https://api.figshare.com/v2/articles/3829962), [배포 페이지](https://figshare.com/articles/dataset/20news-bydate_tar_gz/3829962)

## 배포 조건과 출처 표기

정확한 배포 항목의 메타데이터에서 CC BY 4.0 표시를 확인했습니다. Creative Commons의 공식 조건 설명은 적절한 출처 표기, 라이선스 링크 제공과 수정 여부 표시를 요구하며, 해당 조건 아래 공유와 변형을 허용합니다. [CC BY 4.0 공식 설명](https://creativecommons.org/licenses/by/4.0/)

프로젝트의 출처 표기는 다음 정보를 사용합니다.

> Nelson Liu, 2016, scikit-learn 20newsgroups dataset 20news-bydate.tar.gz, version 2, Figshare, DOI 10.6084/m9.figshare.3829962.v2, CC BY 4.0.

프로젝트에서 수행한 변경은 네 카테고리 선택, 헤더·푸터·인용 제거, 이메일·일반 전화번호 마스킹, 빈 본문 제외, 고정 Train/Test 재분할, 토큰화·불용어 제거 및 수치 변환입니다. 데이터 처리 결과의 코퍼스 해시와 분포는 [데이터 메타데이터](../artifacts/dataset_metadata.json)에 기록합니다.

CC BY 4.0의 공식 설명에는 개인정보·퍼블리시티 등 다른 권리가 이용에 영향을 줄 수 있다는 내용도 있습니다. 따라서 배포 라이선스 확인과 미션의 개인정보 미포함 검수는 각각 근거를 확보해야 합니다. [CC BY 4.0의 권리 관련 안내](https://creativecommons.org/licenses/by/4.0/)

## 다른 배포본과 확인 경로

UCI의 Twenty Newsgroups 항목은 20,000건과 `20_newsgroups.tar.gz`, `mini_newsgroups.tar.gz`를 제시하며 CC BY 4.0을 표시합니다. 현재 sklearn이 사용하는 파일명·건수와 다르므로, 이번 배포 조건 확인은 위 Figshare의 일치하는 file ID를 근거로 했습니다. [UCI Twenty Newsgroups](https://archive.ics.uci.edu/dataset/113/twenty%2Bnewsgroups)

원본 안내용 MIT URL은 `qwone.com`으로 리다이렉트합니다. 2026-11-12 웹 조회에서는 리다이렉트 이후 페이지가 시간 초과로 열리지 않았습니다. Figshare 웹 페이지도 웹 조회 도구에서 열리지 않아, 같은 날 Python `urllib.request`로 공식 API를 조회해 배포 항목과 라이선스 표시를 확인했습니다. [원본 안내 URL](https://people.csail.mit.edu/jrennie/20Newsgroups/), [확인한 Figshare API](https://api.figshare.com/v2/articles/3829962)

sklearn의 설명은 헤더·서명·인용에 있는 정보가 분류기 학습에 영향을 줄 수 있음을 명시합니다. 로더 소스에서는 푸터와 인용 제거가 휴리스틱이며 항상 정확하지 않을 수 있다고 설명합니다. 현재 개인정보 검수 범위와 미확인 사항은 이슈 #8에서 관리합니다. [sklearn 데이터셋 설명](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset), [로더 구현](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.9.1/sklearn/datasets/_twenty_newsgroups.py)

## 고정 불용어 선정 근거

[stopwords_en.txt](../resources/stopwords_en.txt)는 사용자가 제공한 Python 패키지 생성 코드의 문자열 목록을 복원한 파일입니다. 제공 코드에는 외부 불용어 사전의 이름·버전·원출처가 없었습니다. 프로젝트는 이 123개 단어를 재현 가능한 기준 목록 `a4-1-en-v1`로 채택합니다.

목록에는 관사, 대명사, 전치사, 접속사, be/do/have 계열과 일부 일반 부사가 포함됩니다. 문서 주제를 구분하는 힘이 낮은 기능어를 줄이는 기준으로 사용합니다. 실제 빈도 기반 최적화를 거친 목록이라는 근거는 없으며, 현재 성능은 이 목록을 고정한 조건에서 측정했습니다.

소문자 단어 123개가 알파벳 순서로 저장돼 있고 중복은 없습니다. 한 글자 토큰은 앞선 정규식 규칙에서 제외됩니다. 목록에 포함된 `no`, `nor`, `not`은 부정 표현을 잃게 할 수 있어 향후 비교 실험의 대상입니다. 목록을 변경하면 동일 분할에서 분류·검색 성능을 다시 측정하고 새 목록 버전과 해시를 기록해야 합니다.

| 목록 확인 항목 | 값 |
| --- | --- |
| 프로젝트 목록 버전 | `a4-1-en-v1` |
| 채택 기록일 | 2026-11-12 |
| 단어 수 / 고유 단어 수 | 123 / 123 |
| UTF-8, LF와 마지막 개행으로 정규화한 SHA-256 | `566c5909c1523a651dc2691ff0fa4fddabcc2d047daceccd2f4b962789af313d` |
| 확인 당시 Windows 파일 바이트 SHA-256 | `83f4744642d01cdb2a59f57349de689cabe8f6ae9b31ea85cc16826ad3c56a1b` |

목록 내용을 비교할 때는 UTF-8·LF 정규화 해시를 사용합니다. 외부 원출처 확인은 미확인으로 남기고 사용자 제공 목록을 프로젝트 기준으로 채택한 근거와 변경 이력을 기록했습니다.

## 확인 결과

- [x] 설치된 sklearn의 실제 다운로드 URL·파일명·기대 해시 확인
- [x] 같은 file ID의 Figshare 배포 항목·버전·라이선스 표시 확인
- [x] 출처 표기와 프로젝트 데이터 변경 내용 기록
- [x] 고정 불용어 목록의 프로젝트 채택 근거·버전·해시 기록
- [ ] 사용자 제공 불용어 목록의 외부 원출처 확인
- [ ] 전체 사용 데이터의 인명·서명 등 식별 정보 검수 완료
