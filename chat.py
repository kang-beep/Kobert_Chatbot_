"""
학습된 페르소나 챗봇 모델을 로드하여 대화 테스트
작성자: 서강산
작성일: 2024-10-08
"""

import os
import json
import argparse
import torch
import logging

# 프로젝트 모듈 import
import config
from models.kobert_persona_chatbot import (
    load_model_with_answers,
    generate_response
)

# 로깅 설정
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def chat_with_model(model_path=None):
    """
    학습된 모델을 로드하여 대화를 테스트합니다.
    """
    if not model_path:
        model_path = os.path.join(config.MODEL_DIR, "kobert_persona_chatbot_final.pt")
    
    if not os.path.exists(model_path):
        logger.error(f"모델 파일 '{model_path}'이 존재하지 않습니다.")
        return
    
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"실행 장치: {device}")
    
    # 모델과 토크나이저, 응답 목록 로드
    model, tokenizer, answers = load_model_with_answers(model_path, device)
    
    logger.info(f"모델 로드 완료. 응답 종류: {len(answers)}개")
    logger.info("대화를 시작합니다. '종료'를 입력하면 대화가 종료됩니다.")
    
    context = []
    
    while True:
        # 사용자 입력
        user_input = input("\n사용자: ")
        if user_input.lower() in ('종료', 'quit', 'exit'):
            logger.info("대화를 종료합니다.")
            break
        
        # 페르소나 정보 추가 (실제로는 사용자 정보를 활용)
        if not context:
            persona_prefix = "[페르소나 정보] 나는 40대 남성이다. 나는 기술에 관심이 많다.\n[주제] 일상대화\n[질문] "
            augmented_input = f"{persona_prefix}{user_input}"
        else:
            augmented_input = user_input
        
        # 응답 생성
        response = generate_response(model, tokenizer, augmented_input, answers, device)
        print(f"챗봇: {response}")
        
        # 대화 기록 추가
        context.append((user_input, response))

def main():
    parser = argparse.ArgumentParser(description='페르소나 챗봇 대화 테스트')
    parser.add_argument('--model-path', type=str, default=None,
                        help='학습된 모델 파일 경로')
    args = parser.parse_args()
    
    chat_with_model(args.model_path)

if __name__ == "__main__":
    main() 