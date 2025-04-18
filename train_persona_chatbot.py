"""
KoBERT 기반 페르소나 챗봇 학습 스크립트
작성자: 서강산
작성일: 2024-10-08
수정일: 2024-10-08
"""

import os
import json
import logging
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

# 프로젝트 모듈 import
import config
from models.kobert_persona_chatbot import (
    PersonaChatDataset, 
    KobertPersonaChatbot,
    train_model,
    save_model_with_answers,
    generate_response
)
from prepare_dataset import prepare_datasets

# 로깅 설정
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def load_dataset(file_path):
    """
    데이터셋 파일을 로드합니다.
    """
    questions = []
    answers = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    questions.append(parts[0])
                    answers.append(parts[1])
    
    return list(zip(questions, answers))

def load_answers(file_path):
    """
    응답 목록 파일을 로드합니다.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def train_persona_chatbot(args):
    """
    페르소나 챗봇 모델을 학습합니다.
    """
    logger.info(f"{'='*20} 페르소나 챗봇 학습 시작 {'='*20}")
    
    # 데이터셋 준비
    if not os.path.exists(args.train_data):
        logger.info("데이터셋 준비 중...")
        dataset_info = prepare_datasets(
            session_level=config.MULTISESSION_CONFIG['use_session_level'],
            valid_ratio=0.1
        )
        train_data_path = dataset_info['train_path']
        valid_data_path = dataset_info['valid_path']
        answer_path = dataset_info['answer_path']
    else:
        train_data_path = args.train_data
        valid_data_path = args.valid_data
        answer_path = args.answer_file
    
    # 데이터 로드
    logger.info(f"학습 데이터 로드 중: {train_data_path}")
    train_pairs = load_dataset(train_data_path)
    
    logger.info(f"검증 데이터 로드 중: {valid_data_path}")
    valid_pairs = load_dataset(valid_data_path)
    
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"학습 장치: {device}")
    
    # 토크나이저 로드
    logger.info("토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.TOKENIZER_CONFIG['pretrained_model_name'], 
        trust_remote_code=True
    )
    
    # 데이터셋 및 데이터로더 설정
    logger.info("데이터로더 설정 중...")
    train_dataset = PersonaChatDataset(
        train_pairs, 
        tokenizer, 
        max_len=config.TOKENIZER_CONFIG['max_length']
    )
    valid_dataset = PersonaChatDataset(
        valid_pairs, 
        tokenizer, 
        max_len=config.TOKENIZER_CONFIG['max_length']
    )
    
    # 고유 응답 목록 가져오기
    unique_answers = train_dataset.get_unique_answers()
    logger.info(f"고유 응답 수: {len(unique_answers)}")
    
    # 응답 목록 저장
    answer_path = os.path.join(os.path.dirname(args.answer_file), "answers.json")
    os.makedirs(os.path.dirname(answer_path), exist_ok=True)
    with open(answer_path, 'w', encoding='utf-8') as f:
        json.dump(unique_answers, f, ensure_ascii=False, indent=2)
    logger.info(f"답변 목록이 '{answer_path}'에 저장되었습니다.")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.TRAIN_CONFIG['batch_size'], 
        shuffle=True
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=config.TRAIN_CONFIG['batch_size'], 
        shuffle=False
    )
    
    # 모델 초기화
    logger.info("모델 초기화 중...")
    model = KobertPersonaChatbot(
        pretrained_model_name=config.TOKENIZER_CONFIG['pretrained_model_name'],
        num_answers=len(unique_answers)
    ).to(device)
    
    # 옵티마이저 및 스케줄러 설정
    optimizer = AdamW(
        model.parameters(), 
        lr=config.TRAIN_CONFIG['learning_rate'], 
        weight_decay=config.TRAIN_CONFIG['weight_decay']
    )
    
    total_steps = len(train_loader) * config.TRAIN_CONFIG['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.TRAIN_CONFIG['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # 모델 학습
    logger.info("모델 학습 시작...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=config.TRAIN_CONFIG['epochs'],
        save_dir=config.MODEL_DIR
    )
    
    # 최종 모델 저장
    logger.info("최종 모델 저장 중...")
    final_model_path = os.path.join(config.MODEL_DIR, "kobert_persona_chatbot_final.pt")
    save_model_with_answers(model, tokenizer, unique_answers, final_model_path)
    
    # 테스트 예측
    if args.test_input:
        logger.info(f"테스트 예측: '{args.test_input}'")
        response = generate_response(model, tokenizer, args.test_input, unique_answers, device)
        logger.info(f"응답: '{response}'")
    
    logger.info(f"{'='*20} 페르소나 챗봇 학습 완료 {'='*20}")
    return model, tokenizer, unique_answers

def main():
    parser = argparse.ArgumentParser(description='KoBERT 기반 페르소나 챗봇 학습')
    parser.add_argument('--train-data', type=str, default='datasets/persona_train.txt',
                        help='학습 데이터 파일 경로')
    parser.add_argument('--valid-data', type=str, default='datasets/persona_valid.txt',
                        help='검증 데이터 파일 경로')
    parser.add_argument('--answer-file', type=str, default='datasets/answers.json',
                        help='응답 목록 파일 경로')
    parser.add_argument('--test-input', type=str, default=None,
                        help='학습 후 테스트할 입력 텍스트')
    args = parser.parse_args()
    
    train_persona_chatbot(args)

if __name__ == "__main__":
    main() 