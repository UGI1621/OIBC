# OIBC 2026 — 태양광 집합자원 구성·예측

[ruby9915/OIBC](https://github.com/ruby9915/OIBC)를 포크한 저장소입니다. 작업은 브랜치별로 나눠 두었고, **각 브랜치마다 무슨 일을 했는지 브랜치 이름과 같은 md 파일에 정리해 두었습니다.** 파일 이름은 브랜치 이름의 `/`를 `-`로 바꾼 것입니다.

## 작업 브랜치 (본 줄기)

| 브랜치 | 정리 문서 | 내용 |
|---|---|---|
| [`ugi/penalty-analysis-2026-10-28`](https://github.com/UGI1621/OIBC/tree/ugi/penalty-analysis-2026-10-28) | [ugi-penalty-analysis-2026-10-28.md](https://github.com/UGI1621/OIBC/blob/ugi/penalty-analysis-2026-10-28/ugi-penalty-analysis-2026-10-28.md) | 대회 점수식(벌금) 채점 도구, 벌금 원인 분석, MLP 에폭 평균, GBDT 입력 추가, 학습 시작일 2021-01. 혼합 벌금 201.3 → 191.4백만 원 |
| [`ugi/penalty-analysis-2026-10-29`](https://github.com/UGI1621/OIBC/tree/ugi/penalty-analysis-2026-10-29) | [ugi-penalty-analysis-2026-10-29.md](https://github.com/UGI1621/OIBC/blob/ugi/penalty-analysis-2026-10-29/ugi-penalty-analysis-2026-10-29.md) | 28 브랜치 위에 이어서. GBDT 용량 가중치, 팀 구성(지점 나누기), 여러 연도 검증 도구, 다른 연도(2023-24)로 과적합 검증, GBDT 이웃 시각 입력. 혼합 벌금 188.7백만 원 |

## 실험 브랜치

실험은 브랜치를 따로 만들어 시험합니다. **채택한 것만 본 줄기에 합치고, 버린 것은 브랜치에 기록으로 남깁니다.**

| 브랜치 | 정리 문서 | 내용 | 판정 |
|---|---|---|---|
| [`ugi/exp-noise-injection`](https://github.com/UGI1621/OIBC/tree/ugi/exp-noise-injection) | [ugi-exp-noise-injection.md](https://github.com/UGI1621/OIBC/blob/ugi/exp-noise-injection/ugi-exp-noise-injection.md) | MLP 입력에 예보 오차만큼 잡음 주입. 2023-24 +3.5%, 2024-25 +2.0% | 버림 |
| [`ugi/exp-neighbor-hours`](https://github.com/UGI1621/OIBC/tree/ugi/exp-neighbor-hours) | [ugi-exp-neighbor-hours.md](https://github.com/UGI1621/OIBC/blob/ugi/exp-neighbor-hours/ugi-exp-neighbor-hours.md) | GBDT에 1시간 앞뒤 예보와 그날 예보 요약 14개 추가. 세 연도 GBDT −2.1~−2.9%, 혼합 −1.0~−1.1% | 채택, 29에 합침 |

## 브랜치 의존성

```
main (958452d + 이 README)
 └─ ugi/penalty-analysis-2026-10-28        main 958452d 에서 분기
     └─ ugi/penalty-analysis-2026-10-29    28 의 e1fe692 에서 분기
         ├─ ugi/exp-noise-injection        29 의 5531629 에서 분기 (버림, 합치지 않음)
         └─ ugi/exp-neighbor-hours         29 의 b4991ec 에서 분기 → 29 에 합침 (448b367)
```

- **최신 상태는 29 브랜치입니다.** 28 브랜치 작업과 채택한 실험을 모두 포함하므로 29 브랜치만 받으면 됩니다.
- 28 브랜치의 정리 문서 커밋(`46362b7`)은 29 브랜치에 같은 내용으로 복사해 두었습니다(`0a403c2`).
- 어느 브랜치도 main 에는 합치지 않았습니다. main 은 원본 저장소 상태(`958452d`)에 이 README 만 바꾼 것입니다.
- 원래 README(프로젝트 소개, 데이터 준비, 실행 환경)는 원본 저장소와 각 브랜치에 그대로 있습니다.
- 대회 원본 데이터와 모델·예측 파일은 저장소에 없습니다. 데이터는 `OIBC_2026_DATA/handover 2/01_participant_dataset/`에 두고 실행합니다.
