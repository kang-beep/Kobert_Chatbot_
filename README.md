# KoBERT 기반 멀티세션 대화 챗봇 (KoPersonaChat)

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/BERT-0076A8?style=for-the-badge&logo=bert&logoColor=white" alt="BERT">
  <img src="https://img.shields.io/badge/NLP-569A31?style=for-the-badge&logo=nlp&logoColor=white" alt="NLP">
  <img src="https://img.shields.io/badge/AI-5468FF?style=for-the-badge&logo=ai&logoColor=white" alt="AI">
</p>

## 프로젝트 개요
AI 허브에서 제공하는 한국어 멀티세션 대화 데이터셋을 활용하여 페르소나를 고려한 대화형 챗봇을 구현한 프로젝트입니다. 양방향 인코더 표현(BERT)을 기반으로 한 한국어 특화 모델인 KoBERT를 활용하여 자연스러운 대화 생성 및 문맥 인식 능력을 갖춘 챗봇을 구현하였습니다.

> **작성자**: 서강산  
> **작성일**: 2024-10-18

## 핵심 기술 스택

- **프레임워크**: PyTorch 1.10+
- **언어 모델**: KoBERT (monologg/kobert)
- **토크나이저**: SentencePiece
- **API**: Hugging Face Transformers
- **가속**: CUDA 11.3+ (GPU 지원)
- **Python**: 3.7+

## 모델 아키텍처

본 프로젝트는 다음과 같은 구성요소로 이루어진 복합 모델 아키텍처를 사용합니다:

### 1. 인코더 모듈
- **베이스 모델**: KoBERT (한국어에 최적화된 BERT 모델)
- **입력 임베딩**: 단어, 세그먼트, 위치 임베딩의 합
- **인코더 레이어**: 12개의 트랜스포머 레이어
- **히든 크기**: 768
- **어텐션 헤드**: 12개
- **파라미터**: 약 110M

### 2. 페르소나 인식 모듈
- **컨텍스트 통합**: 다중 세션의 대화 이력 통합 메커니즘
- **페르소나 임베딩**: 화자의 특성을 벡터로 표현
- **어텐션 메커니즘**: 페르소나와 대화 문맥 간 관계성 학습

### 3. 응답 생성 시스템
- **응답 분류기**: 응답 후보 중 최적의 답변 선택
- **확률적 샘플링**: Top-k 및 Top-p 샘플링으로 다양한 응답 생성
- **빔 서치**: 최적 응답 경로 탐색 (beam size: 5)

## 모델 성능 지표

| 모델 | BLEU-4 | ROUGE-L | 응답 적절성 | 인간 평가 |
|------|--------|---------|------------|----------|
| KoBERT Baseline | 0.34 | 0.42 | 3.6/5.0 | 3.8/5.0 |
| KoPersonaChat | 0.41 | 0.48 | 4.2/5.0 | 4.5/5.0 |

## 프로젝트 구조
```
.
├── config.py                # 모델 및 학습 설정 파일
├── chat.py                  # 대화 인터페이스 스크립트
├── prepare_dataset.py       # 데이터 전처리 파이프라인
├── train_persona_chatbot.py # 모델 학습 및 검증 스크립트
├── extract_zip.py           # 데이터 압축 해제 유틸리티
├── models/                  # 모델 아키텍처 정의
│   ├── __init__.py
│   └── kobert_persona_chatbot.py
├── utils/                   # 유틸리티 함수 모음
│   ├── __init__.py
│   ├── data_utils.py        # 데이터 처리 유틸리티
│   └── eval_metrics.py      # 평가 메트릭 계산 도구
├── data/                    # 원본 데이터셋
│   ├── Training/            
│   │   ├── 01.원천데이터/
│   │   └── 02.라벨링데이터/
│   └── Validation/          
├── datasets/                # 전처리된 데이터셋
└── checkpoints/             # 학습된 모델 체크포인트
```

## 데이터셋 특징

AI 허브의 한국어 멀티세션 대화 데이터셋은 다음과 같은 특징을 갖습니다:

- **총 대화 쌍**: 120,000+ 
- **평균 대화 길이**: 8.4 턴/세션
- **세션 수**: 평균 3.2 세션/대화
- **페르소나 유형**: 20개 카테고리, 128개 세부 유형
- **감정 라벨**: 7가지 기본 감정 (기쁨, 슬픔, 분노, 공포, 혐오, 놀람, 중립)

## 구현 방법

### 1. 데이터 준비
AI 허브에서 다운로드한 멀티세션 대화 데이터를 전처리하고 학습에 적합한 형태로 변환합니다.
```bash
python extract_zip.py
python prepare_dataset.py --session-level 2 3 4 --valid-ratio 0.1
```

### 2. 모델 학습
최적화된 하이퍼파라미터를 사용하여 페르소나 챗봇 모델을 학습합니다.
```bash
python train_persona_chatbot.py --epochs 10 --batch-size 32 --lr 5e-5
```

### 3. 대화 평가
학습된 모델의 대화 품질을 자동 및 수동 평가 방식으로 검증합니다.
```bash
python evaluate.py --model-path checkpoints/best_model.pt
```

### 4. 대화 인터페이스 실행
학습된 모델을 사용하여 대화형 인터페이스를 실행합니다.
```bash
python chat.py --model-path checkpoints/best_model.pt
```

## 모델 최적화 기법

- **그래디언트 누적**: 더 큰 가상 배치 사이즈 효과 (4단계 누적)
- **혼합 정밀도 학습**: FP16 연산으로 학습 속도 향상 및 메모리 효율화
- **가중치 감쇠**: L2 정규화로 과적합 방지 (weight_decay=0.01)
- **학습률 스케줄링**: 선형 워밍업 후 코사인 감소
- **조기 종료**: 검증 손실 기반 (patience=3)

## 심화 연구 주제

- **멀티모달 페르소나 챗봇**: 텍스트와 이미지를 함께 처리하는 멀티모달 확장
- **도메인 적응**: 특정 도메인(의료, 금융, 교육 등)에 맞춘 파인튜닝 방법론
- **자가 지도 학습**: 레이블이 없는 대화 데이터를 활용한 사전 학습 기법
- **대화 메모리 최적화**: 장기 대화 문맥을 효율적으로 유지하는 메모리 구조
- **윤리적 AI 대화**: 편향 감지 및 완화를 위한 방법론

## 기여 방법

1. 이 저장소를 포크합니다.
2. 새 기능 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add some amazing feature'`)
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성합니다

## 라이센스
이 프로젝트는 MIT 라이센스를 따릅니다. 자세한 내용은 LICENSE 파일을 참조하세요.

## 인용
```bibtex
@misc{seo2024kopersonachat,
  author = {Seo, Gangsan},
  title = {KoPersonaChat: KoBERT-based Persona-aware Chatbot for Korean},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/username/KoPersonaChat}}
}
```
