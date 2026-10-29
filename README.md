# OIBC 2026 — 태양광 집합자원 구성·예측

[ruby9915/OIBC](https://github.com/ruby9915/OIBC)를 포크한 저장소입니다. 작업은 브랜치별로 나눠 두었고, **각 브랜치마다 무슨 일을 했는지 브랜치 이름과 같은 md 파일에 정리해 두었습니다.**

| 브랜치 | 정리 문서 | 내용 |
|---|---|---|
| [`ugi/penalty-analysis-2026-10-28`](https://github.com/UGI1621/OIBC/tree/ugi/penalty-analysis-2026-10-28) | [ugi-penalty-analysis-2026-10-28.md](https://github.com/UGI1621/OIBC/blob/ugi/penalty-analysis-2026-10-28/ugi-penalty-analysis-2026-10-28.md) | 대회 점수식(벌금) 채점 도구, 벌금 원인 분석, MLP 에폭 평균, GBDT 입력 추가, 학습 시작일 2021-01. 혼합 벌금 201.3 → 191.4백만 원 |
| [`ugi/penalty-analysis-2026-10-29`](https://github.com/UGI1621/OIBC/tree/ugi/penalty-analysis-2026-10-29) | [ugi-penalty-analysis-2026-10-29.md](https://github.com/UGI1621/OIBC/blob/ugi/penalty-analysis-2026-10-29/ugi-penalty-analysis-2026-10-29.md) | 28 브랜치 위에 이어서. GBDT 용량 가중치, 팀 구성(지점 나누기), 여러 연도 검증 도구, 다른 연도(2023-24)로 과적합 검증 |

- 29 브랜치는 28 브랜치를 포함합니다. 최신 상태는 29 브랜치입니다.
- main은 원본 저장소의 `958452d`와 같습니다. 원래 README(프로젝트 소개, 데이터 준비, 실행 환경)는 원본 저장소와 각 브랜치에 그대로 있습니다.
- 대회 원본 데이터와 모델·예측 파일은 저장소에 없습니다. 데이터는 `OIBC_2026_DATA/handover 2/01_participant_dataset/`에 두고 실행합니다.
