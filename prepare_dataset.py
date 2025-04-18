"""
AI 허브 멀티세션 대화 데이터 전처리 스크립트
작성자: 서강산
작성일: 2024-10-08
"""

import os
import json
import argparse
from typing import Dict, List, Tuple
import logging
from pathlib import Path

# 프로젝트 모듈 import
from utils.data_utils import (
    get_session_files,
    process_multisession_data,
    split_data_train_valid,
    save_dataset
)
import config

# 로깅 설정
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def create_dataset_dir():
    """데이터셋 저장 디렉토리를 생성합니다."""
    dataset_dir = os.path.join(config.ROOT_DIR, 'datasets')
    os.makedirs(dataset_dir, exist_ok=True)
    return dataset_dir

def prepare_datasets(session_level=[2, 3, 4], valid_ratio=0.1):
    """
    AI 허브 멀티세션 대화 데이터를 전처리하여 학습용 데이터셋을 생성합니다.
    
    Args:
        session_level: 사용할 세션 레벨 목록
        valid_ratio: 검증 데이터 비율
    """
    logger.info(f"{'='*20} 데이터셋 준비 시작 {'='*20}")
    logger.info(f"세션 레벨: {session_level}")
    
    dataset_dir = create_dataset_dir()
    
    # 라벨링 데이터 처리
    all_dialog_data = []
    
    for level in session_level:
        label_dir = config.EXTRACTED_LABEL_DIR.get(f'session{level}')
        if not os.path.exists(label_dir):
            logger.warning(f"세션 {level}의 라벨링 데이터 디렉토리가 존재하지 않습니다: {label_dir}")
            continue
            
        logger.info(f"세션 {level} 라벨링 데이터 처리 중...")
        dialog_data = process_multisession_data(label_dir, [level])
        all_dialog_data.extend(dialog_data)
        logger.info(f"세션 {level} 처리 완료: {len(dialog_data)}개 대화 쌍")
    
    # 학습/검증 데이터 분할
    train_pairs, valid_pairs = split_data_train_valid(all_dialog_data, valid_ratio)
    
    # 데이터셋 저장
    train_path = os.path.join(dataset_dir, "persona_train.txt")
    valid_path = os.path.join(dataset_dir, "persona_valid.txt")
    
    save_dataset(train_pairs, train_path)
    save_dataset(valid_pairs, valid_path)
    
    # 답변 목록 저장 (추론 시 필요)
    answers = [answer for _, answer in train_pairs]
    unique_answers = list(set(answers))
    
    answer_path = os.path.join(dataset_dir, "answers.json")
    with open(answer_path, 'w', encoding='utf-8') as f:
        json.dump(unique_answers, f, ensure_ascii=False, indent=2)
    
    logger.info(f"고유 답변 수: {len(unique_answers)}")
    logger.info(f"답변 목록이 '{answer_path}'에 저장되었습니다.")
    logger.info(f"{'='*20} 데이터셋 준비 완료 {'='*20}")
    
    return {
        'train_path': train_path,
        'valid_path': valid_path,
        'answer_path': answer_path,
        'num_train': len(train_pairs),
        'num_valid': len(valid_pairs),
        'num_answers': len(unique_answers)
    }

def main():
    parser = argparse.ArgumentParser(description='AI 허브 멀티세션 대화 데이터 전처리')
    parser.add_argument('--session-level', type=int, nargs='+', default=[2, 3, 4],
                        help='사용할 세션 레벨 목록 (예: 2 3 4)')
    parser.add_argument('--valid-ratio', type=float, default=0.1,
                        help='검증 데이터 비율 (기본값: 0.1)')
    args = parser.parse_args()
    
    prepare_datasets(session_level=args.session_level, valid_ratio=args.valid_ratio)

if __name__ == "__main__":
    main() 