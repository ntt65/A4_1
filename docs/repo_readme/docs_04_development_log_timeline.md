# [A4_1] 전체 개발 진행 이력 및 액션 타임라인

```
2026-11-12
├── [기획/설계] A4_1 요구사항 명세 분석 및 8대 의사결정 수립
│   └── 데이터 누수 방지, Sklearn 1:1 오차 목표 설정
├── [데이터셋] 20 Newsgroups 4개 카테고리 로더 및 PII 전처리 파이프라인
│   └── 3,792건 8:2 고정 분할 및 stopwords_en.txt 구축
├── [코어 엔진] NumPy TF-IDF 벡터라이저 구현
│   └── Vocabulary 알파벳 정렬, Smooth IDF, L2 Norm 구현
│   └── verify_tfidf.py: Sklearn 1:1 오차 1.1102e-15 달성
├── [검색/분류] 코사인 유사도 검색기 & 로지스틱 회귀 모델
│   └── CLI 검색 인터페이스 및 main.py search() 함수 규격 확정
│   └── 분류 모델 학습 (정확도 97.56%), 혼동 행렬 시각화
├── [보너스] 3종 확장 기능 구현
│   └── Double Norm TF, BM25 엔진, 역색인(Inverted Index)
│   └── Toy Example 오차 5.55e-17, 역색인 동치성 2.78e-17 검증
└── [패키징/진단] README.md 작성 및 AI Studio 실행 환경 진단
    └── 로컬 Git 워크스페이스 연동 및 docs 문서화 번들링
```
