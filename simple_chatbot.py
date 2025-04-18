"""
KoBERT 기반 간단한 챗봇 모델
작성자: 서강산
작성일: 2024-10-05
수정일: 2024-10-05 - GPU 학습 지원 추가
수정일: 2024-10-08 - GPU 사용 확인 및 출력 개선, CUDA 검증 추가
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer
import os
import sys
import subprocess

# CUDA 환경 검증
print("="*50)
print("CUDA 환경 확인")
print("="*50)
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"CUDA 버전(PyTorch): {torch.version.cuda if hasattr(torch.version, 'cuda') else '없음'}")

# NVIDIA-SMI 실행 시도
try:
    nvidia_smi_output = subprocess.check_output("nvidia-smi", shell=True).decode('utf-8')
    print("\n[NVIDIA-SMI 정보]")
    print(nvidia_smi_output[:500] + "..." if len(nvidia_smi_output) > 500 else nvidia_smi_output)
except:
    print("nvidia-smi 명령을 실행할 수 없습니다.")

# PyTorch가 CPU 버전으로 설치된 경우 안내문 출력
if not torch.cuda.is_available() and subprocess.call("nvidia-smi", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
    print("\n경고: GPU가 감지되었지만 PyTorch가 GPU를 인식하지 못합니다.")
    print("PyTorch CPU 버전이 설치되었거나 CUDA 드라이버 문제일 수 있습니다.")
    print("GPU 버전의 PyTorch를 설치하려면 다음 명령을 실행하세요:")
    print("pip uninstall torch -y")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("(CUDA 11.8 버전용 명령입니다. 다른 CUDA 버전은 https://pytorch.org/get-started/locally/ 참조)")
    print("\n계속 진행하시겠습니까? (GPU 없이 CPU로 실행됩니다) [y/n]")
    user_input = input().strip().lower()
    if user_input != 'y':
        print("실행을 중단합니다. GPU 버전의 PyTorch를 설치하고 다시 시도하세요.")
        sys.exit(0)
    print("CPU 모드로 계속 실행합니다.")

# CUDA 설정 최적화
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # GPU 오류 디버깅 도움
torch.backends.cudnn.benchmark = True     # GPU 연산 속도 향상

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

# GPU 정보 출력 - 수정된 부분
if device.type == 'cuda':
    print(f"GPU 사용 중: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    print(f"GPU 가용 메모리: {torch.cuda.get_device_properties(0).total_memory/1024**2:.2f} MB")
else:
    print("GPU를 사용할 수 없어 CPU로 실행합니다.")

# KoBERT 모델 및 토크나이저 로드
print("KoBERT 모델 로딩 중...")
try:
    tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
    base_model = AutoModel.from_pretrained("monologg/kobert").to(device)  # 바로 GPU로 로드
    print(f"모델 로딩 완료 - 현재 모델 위치: {next(base_model.parameters()).device}")
except Exception as e:
    print(f"모델 로딩 중 오류 발생: {e}")
    raise

# 간단한 대화 데이터셋 예시
class ChatbotDataset(Dataset):
    def __init__(self, questions, answers, tokenizer, max_len=64):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        
        # 질문 토큰화
        inputs = self.tokenizer(
            question,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 토큰화된 텐서 추출
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        
        # 응답 레이블 - 예시에서는 간단히 인덱스를 반환
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(idx)  # 실제로는 응답 자체를 생성하거나 분류할 수 있음
        }

# 챗봇 모델 정의
class KobertChatbot(nn.Module):
    def __init__(self, bert_model, num_answers):
        super(KobertChatbot, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_answers)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits

# 간단한 학습 함수
def train_chatbot(model, train_loader, optimizer, device, epochs=5):
    model.train()
    print(f"{'='*20} 학습 시작 {'='*20}")
    print(f"학습 장치: {device}")
    print(f"배치 크기: {train_loader.batch_size}, 전체 데이터: {len(train_loader.dataset)}")
    
    # GPU 정보 출력
    if device.type == 'cuda':
        print(f"학습 전 GPU 메모리 사용량: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 그래디언트 초기화
            optimizer.zero_grad()
            
            # 순전파
            outputs = model(input_ids, attention_mask)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            
            # 역전파
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # 배치 진행상황 출력 (4배치마다)
            if batch_idx % 4 == 0:
                print(f"Epoch {epoch+1}/{epochs} | 배치 {batch_idx}/{len(train_loader)} | 손실: {loss.item():.4f}")
            
        avg_loss = total_loss/len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} 완료 | 평균 손실: {avg_loss:.4f}")
        
        # GPU 메모리 모니터링
        if device.type == 'cuda':
            print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    
    print(f"{'='*20} 학습 완료 {'='*20}")
    
    # 최종 메모리 사용량 출력
    if device.type == 'cuda':
        print(f"학습 후 GPU 메모리 사용량: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        # 메모리 캐시 정보도 출력
        print(f"GPU 메모리 캐시: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")

# 응답 생성 함수
def generate_response(model, tokenizer, question, answers, device):
    model.eval()
    inputs = tokenizer(
        question,
        truncation=True,
        max_length=64,
        padding='max_length',
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        predicted_idx = torch.argmax(outputs, dim=1).item()
    
    return answers[predicted_idx]

# 메인 함수
def main():
    # 예시 데이터
    questions = [
        "안녕하세요?",
        "오늘 날씨가 어때요?",
        "KoBERT는 무엇인가요?",
        "챗봇 개발은 어렵나요?",
        "인공지능에 대해 알려주세요",
        "파이썬은 어떤 언어인가요?",
        "딥러닝이란 무엇인가요?",
        "자연어 처리는 무엇인가요?",
        "한국어 모델의 특징은 무엇인가요?",
        "BERT 모델이란 무엇인가요?"
    ]
    
    answers = [
        "안녕하세요! 무엇을 도와드릴까요?",
        "오늘 날씨는 맑고 화창합니다.",
        "KoBERT는 한국어 BERT 모델로, 한국어 자연어 처리에 사용됩니다.",
        "챗봇 개발은 기본 원리를 이해하면 시작할 수 있어요. 다양한 학습 데이터가 중요합니다.",
        "인공지능은 인간의 지능을 모방하는 컴퓨터 기술로, 기계학습과 딥러닝이 주요 기술입니다.",
        "파이썬은 간결하고 읽기 쉬운 문법을 가진 고수준 프로그래밍 언어입니다. 인공지능 개발에 널리 사용됩니다.",
        "딥러닝은 여러 층의 인공 신경망을 사용하여 데이터로부터 패턴을 학습하는 기계학습의 한 분야입니다.",
        "자연어 처리(NLP)는 컴퓨터가 인간의 언어를 이해하고 처리할 수 있게 하는 기술입니다.",
        "한국어 모델은 교착어인 한국어의 특성을 고려한 토큰화와 형태소 분석이 중요합니다.",
        "BERT는 양방향 인코더 표현을 사용하는 트랜스포머 모델로, 문맥을 고려한 단어 표현을 학습합니다."
    ]
    
    # 데이터셋 및 데이터로더 설정
    dataset = ChatbotDataset(questions, answers, tokenizer)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)  # 배치 크기 증가
    
    # 모델 초기화 (바로 GPU로)
    model = KobertChatbot(base_model, len(answers)).to(device)
    
    # 옵티마이저 설정 (학습률 약간 감소)
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)  # 가중치 감쇠 추가
    
    # 모델 학습
    train_chatbot(model, train_loader, optimizer, device, epochs=30)  # 에포크 증가
    
    # 모델 저장
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'kobert_chatbot_model.pt')
    print("모델이 'kobert_chatbot_model.pt'로 저장되었습니다.")
    
    # 대화 테스트
    print("\n챗봇과 대화를 시작합니다. '종료'를 입력하면 대화가 종료됩니다.")
    while True:
        user_input = input("\n사용자: ")
        if user_input.lower() == '종료':
            print("대화를 종료합니다. 감사합니다!")
            break
        
        response = generate_response(model, tokenizer, user_input, answers, device)
        print(f"챗봇: {response}")

if __name__ == "__main__":
    main() 