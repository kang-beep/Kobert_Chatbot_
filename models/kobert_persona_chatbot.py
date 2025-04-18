"""
KoBERT 기반 페르소나 챗봇 모델
작성자: 서강산
작성일: 2024-10-08
수정일: 2024-10-08
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import os
import json
import logging
from typing import List, Dict, Tuple, Optional

# 로깅 설정
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 대화 데이터셋 클래스
class PersonaChatDataset(Dataset):
    def __init__(self, dialog_pairs, tokenizer, max_len=128):
        """
        페르소나 대화 데이터셋
        
        Args:
            dialog_pairs: (question, answer) 형태의 대화 쌍 리스트
            tokenizer: HuggingFace 토크나이저
            max_len: 최대 시퀀스 길이
        """
        self.questions, self.answers = zip(*dialog_pairs) if dialog_pairs else ([], [])
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.unique_answers = list(set(self.answers))  # 고유 응답 목록
        self.answer_to_idx = {answer: idx for idx, answer in enumerate(self.unique_answers)}
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        
        # 질문 토큰화
        question_encoding = self.tokenizer(
            question,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 응답 인덱스 (고유 응답 리스트 내 인덱스)
        answer_idx = self.answer_to_idx.get(answer, 0)
        
        # 토큰화된 텐서 추출
        input_ids = question_encoding['input_ids'].squeeze(0)
        attention_mask = question_encoding['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'question': question,
            'answer': answer,
            'labels': torch.tensor(answer_idx)  # 고유한 응답 인덱스를 레이블로 사용
        }
    
    def get_unique_answers(self):
        """고유 응답 목록을 반환합니다."""
        return self.unique_answers

# KoBERT 기반 페르소나 챗봇 모델
class KobertPersonaChatbot(nn.Module):
    def __init__(self, pretrained_model_name, num_answers, dropout_rate=0.3):
        """
        KoBERT 기반 페르소나 챗봇 모델
        
        Args:
            pretrained_model_name: 사전 학습된 모델 이름 ('monologg/kobert')
            num_answers: 응답 클래스 수
            dropout_rate: 드롭아웃 비율
        """
        super(KobertPersonaChatbot, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_answers)
        
    def forward(self, input_ids, attention_mask):
        """
        순전파
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits

# 모델 학습 함수
def train_model(model, train_loader, valid_loader, optimizer, scheduler, device, epochs=5, save_dir=None):
    """
    모델 학습 함수
    
    Args:
        model: 학습할 모델
        train_loader: 학습 데이터로더
        valid_loader: 검증 데이터로더
        optimizer: 옵티마이저
        scheduler: 스케줄러
        device: 학습 장치
        epochs: 학습 에포크 수
        save_dir: 모델 저장 디렉토리
    """
    logger.info(f"{'='*20} 학습 시작 {'='*20}")
    logger.info(f"학습 장치: {device}")
    logger.info(f"배치 크기: {train_loader.batch_size}, 전체 데이터: {len(train_loader.dataset)}")
    
    # 레이블 범위 검증
    num_classes = model.classifier.out_features
    logger.info(f"모델 출력 클래스 수: {num_classes}")
    
    # 손실 함수
    criterion = nn.CrossEntropyLoss()
    
    # 최적의 모델 저장을 위한 변수
    best_valid_loss = float('inf')
    
    # GPU 메모리 모니터링
    if device.type == 'cuda':
        logger.info(f"학습 전 GPU 메모리 사용량: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    
    for epoch in range(epochs):
        # 학습 모드
        model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 레이블 유효성 검사 - 범위를 벗어나는 레이블이 있는지 확인
            if torch.any(labels >= num_classes):
                invalid_labels = labels[labels >= num_classes]
                logger.warning(f"범위를 벗어나는 레이블 발견: {invalid_labels.tolist()} (최대 클래스 수: {num_classes})")
                # 유효한 범위로 클램핑
                labels = torch.clamp(labels, 0, num_classes - 1)
            
            # 그래디언트 초기화
            optimizer.zero_grad()
            
            # 순전파
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            # 역전파
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            # 배치 진행상황 출력 (100배치마다)
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} | 배치 {batch_idx}/{len(train_loader)} | 손실: {loss.item():.4f}")
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 검증
        model.eval()
        valid_loss = 0
        
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # 레이블 유효성 검사
                if torch.any(labels >= num_classes):
                    # 유효한 범위로 클램핑
                    labels = torch.clamp(labels, 0, num_classes - 1)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                valid_loss += loss.item()
        
        avg_valid_loss = valid_loss / len(valid_loader)
        
        # 결과 출력
        logger.info(f"Epoch {epoch+1}/{epochs} 완료 | 학습 손실: {avg_train_loss:.4f} | 검증 손실: {avg_valid_loss:.4f}")
        
        # 최적의 모델 저장
        if avg_valid_loss < best_valid_loss and save_dir:
            best_valid_loss = avg_valid_loss
            
            # 모델 저장
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"kobert_persona_chatbot_epoch_{epoch+1}.pt")
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'valid_loss': avg_valid_loss,
            }, model_path)
            
            logger.info(f"모델 저장됨: {model_path}")
        
        # GPU 메모리 모니터링
        if device.type == 'cuda':
            logger.info(f"GPU 메모리 사용량: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    
    logger.info(f"{'='*20} 학습 완료 {'='*20}")
    
    # 최종 메모리 사용량 출력
    if device.type == 'cuda':
        logger.info(f"학습 후 GPU 메모리 사용량: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        logger.info(f"GPU 메모리 캐시: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
    
    return model

# 응답 생성 함수
def generate_response(model, tokenizer, question, answer_list, device, max_len=128):
    """
    질문에 대한 응답 생성
    
    Args:
        model: 학습된 모델
        tokenizer: 토크나이저
        question: 질문 텍스트
        answer_list: 응답 후보 리스트
        device: 추론 장치
        max_len: 최대 시퀀스 길이
    """
    model.eval()
    
    # 질문 토큰화
    inputs = tokenizer(
        question,
        truncation=True,
        max_length=max_len,
        padding='max_length',
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        predicted_idx = torch.argmax(outputs, dim=1).item()
    
    if predicted_idx < len(answer_list):
        return answer_list[predicted_idx]
    else:
        return "죄송합니다, 답변을 찾을 수 없습니다."

# 모델 저장 함수
def save_model_with_answers(model, tokenizer, answer_list, save_path):
    """
    모델과 답변 목록을 함께 저장
    """
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
    # 모델 저장
    model_data = {
        'model_state_dict': model.state_dict(),
        'answers': answer_list
    }
    
    torch.save(model_data, save_path)
    
    # 토크나이저도 저장
    tokenizer_path = os.path.join(os.path.dirname(save_path), "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    
    logger.info(f"모델과 답변 목록이 '{save_path}'에 저장되었습니다.")

# 모델 로딩 함수
def load_model_with_answers(model_path, device):
    """
    모델과 답변 목록을 함께 로드
    """
    if not os.path.exists(model_path):
        logger.error(f"모델 파일 '{model_path}'이 존재하지 않습니다.")
        return None, None
    
    # 모델 로드
    model_data = torch.load(model_path, map_location=device)
    answer_list = model_data.get('answers', [])
    
    # 토크나이저 로드
    tokenizer_path = os.path.join(os.path.dirname(model_path), "tokenizer")
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
    
    # 모델 초기화
    model = KobertPersonaChatbot(
        pretrained_model_name="monologg/kobert", 
        num_answers=len(answer_list)
    ).to(device)
    
    # 모델 가중치 로드
    model.load_state_dict(model_data['model_state_dict'])
    
    logger.info(f"모델과 답변 목록이 '{model_path}'에서 로드되었습니다.")
    return model, tokenizer, answer_list 