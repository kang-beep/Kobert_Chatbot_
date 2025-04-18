# KoBERT 기반 AI 허브 멀티세션 대화 챗봇

## 프로젝트 개요
AI 허브에서 제공하는 한국어 멀티세션 대화 데이터셋을 활용하여 페르소나를 고려한 챗봇을 구현한 프로젝트입니다.

작성자: 서강산
작성일: 2025-04-18

## 프로젝트 구조
```
.
├── config.py                # 설정 파일
├── chat.py                  # 대화 테스트 스크립트
├── prepare_dataset.py       # 데이터셋 준비 스크립트
├── train_persona_chatbot.py # 모델 학습 스크립트
├── extract_zip.py           # 압축 파일 해제 유틸리티
├── models/                  # 모델 정의
│   ├── __init__.py
│   └── kobert_persona_chatbot.py
├── utils/                   # 유틸리티 함수
│   ├── __init__.py
│   └── data_utils.py
├── data/                    # 데이터 디렉토리
│   ├── Training/            # 학습 데이터
│   │   ├── 01.원천데이터/
│   │   └── 02.라벨링데이터/
│   └── Validation/          # 검증 데이터
├── datasets/                # 전처리된 데이터셋 (자동 생성)
└── models/                  # 학습된 모델 저장 (자동 생성)
```

## 사용 방법

### 1. 데이터 준비
AI 허브에서 다운로드한 멀티세션 대화 데이터를 `data` 디렉토리에 압축 해제합니다.
```
python extract_zip.py
```

### 2. 데이터셋 전처리
라벨링 데이터를 전처리하여 학습용 데이터셋을 생성합니다.
```
python prepare_dataset.py --session-level 2 3 4 --valid-ratio 0.1
```

### 3. 모델 학습
페르소나 챗봇 모델을 학습합니다.
```
python train_persona_chatbot.py
```

### 4. 대화 테스트
학습된 모델을 사용하여 대화를 테스트합니다.
```
python chat.py
```

## 주요 기능
- **페르소나 인식**: 사용자와 챗봇의 페르소나 정보를 활용한 대화 생성
- **멀티세션 대화**: 여러 세션에 걸친 대화 문맥 유지
- **GPU 가속**: CUDA를 활용한 GPU 학습 지원

## 기술 스택
- Python 3.6+
- PyTorch
- Transformers (Hugging Face)
- KoBERT (monologg/kobert)

## 라이센스
이 프로젝트는 MIT 라이센스를 따릅니다. 자세한 내용은 LICENSE 파일을 참조하세요. 
