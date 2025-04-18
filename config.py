"""
AI 허브 멀티세션 대화 데이터 학습 설정
작성자: 서강산
작성일: 2024-10-08
"""

import os
from pathlib import Path

# 프로젝트 루트 경로
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# 데이터 경로
DATA_DIR = ROOT_DIR / 'data'
TRAIN_DIR = DATA_DIR / 'Training'
VALID_DIR = DATA_DIR / 'Validation'

# 원천 데이터 및 라벨링 데이터 경로
SOURCE_DATA_DIR = TRAIN_DIR / '01.원천데이터'
LABEL_DATA_DIR = TRAIN_DIR / '02.라벨링데이터'

# 추출된 데이터 경로
EXTRACTED_SOURCE_DIR = {
    'session2': SOURCE_DATA_DIR / 'TS_session2_extracted',
    'session3': SOURCE_DATA_DIR / 'TS_session3_extracted',
    'session4': SOURCE_DATA_DIR / 'TS_session4_extracted'
}

EXTRACTED_LABEL_DIR = {
    'session2': LABEL_DATA_DIR / 'TL_session2_extracted',
    'session3': LABEL_DATA_DIR / 'TL_session3_extracted',
    'session4': LABEL_DATA_DIR / 'TL_session4_extracted'
}

# 모델 저장 경로
MODEL_DIR = ROOT_DIR / 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# 학습 설정
TRAIN_CONFIG = {
    'batch_size': 16,
    'epochs': 10,
    'learning_rate': 3e-5,
    'weight_decay': 0.01,
    'max_seq_length': 128,
    'warmup_steps': 500,
    'seed': 42,
    'save_steps': 1000
}

# 토크나이저 설정
TOKENIZER_CONFIG = {
    'pretrained_model_name': 'monologg/kobert',
    'max_length': 128,
    'padding': 'max_length',
    'truncation': True
}

# 멀티세션 대화 학습 설정
MULTISESSION_CONFIG = {
    'use_session_level': [2, 3, 4],  # 학습에 사용할 세션 레벨 (2, 3, 4)
    'use_session_history': True,     # 이전 세션 히스토리 사용 여부
    'use_persona_summary': True,     # 페르소나 요약 정보 사용 여부
    'max_dialog_history': 5          # 최대 대화 기록 사용 수
} 