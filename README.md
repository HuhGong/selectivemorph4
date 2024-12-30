# Selective Morph

Selective Morph는 선택적 스타일 트랜스퍼와 실시간 세그멘테이션을 결합한 혁신적인 이미지 변환 플랫폼입니다.

## 주요 기능

- 선택적 스타일 트랜스퍼: 이미지의 특정 영역에만 원하는 스타일을 적용
- 실시간 세그멘테이션: BiSeNet을 활용한 고성능 실시간 이미지 분할
- 직관적인 UI: 사용자 친화적인 인터페이스로 쉽게 스타일 적용
- 커스텀 스타일: 자신만의 스타일 이미지 업로드 가능

## 기술 스택

### Frontend
- React
- CSS3
- JavaScript

### Backend
- BiSeNet V1 & V2 세그멘테이션 모델[1][2]
- Style Transfer 알고리즘
- FastAPI
- PyTorch

## 성능

BiSeNet 모델은 Cityscapes 데이터셋에서 다음과 같은 성능을 보여줍니다:

| 모델 | mIOU | FPS |
|------|------|-----|
| BiSeNetV1 | 68.4% | 105 |
| BiSeNetV2 | 72.6% | 156 |

## 설치 및 실행

자세한 백엔드 설정과 모델 학습 방법은 `backend/README.md`를 참조하세요.

## 팀 멤버

- 김준래 - Frontend
- 윤현준 - AI Model
- 이제우 - Backend
- 정세찬 - Backend

## 라이선스

MIT License

## 참조

더 자세한 기술적 구현과 모델 학습 방법은 back_end 폴더의 README를 참조하세요.

Citations:
[1] https://arxiv.org/abs/1808.00897
[2] https://arxiv.org/abs/2004.02147