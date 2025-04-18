"""
KoBERT 모델 설정 및 테스트 파일
작성자: 서강산
작성일: 2024-10-05
"""
import torch
from transformers import AutoModel, AutoTokenizer

# 필요한 라이브러리가 설치되어 있는지 확인
try:
    import sentencepiece as spm
    print("SentencePiece 라이브러리가 정상적으로 로드되었습니다.")
except ImportError:
    print("SentencePiece 라이브러리를 설치해주세요: pip install sentencepiece")
    
try:
    # KoBERT 모델 및 토크나이저 로드
    model = AutoModel.from_pretrained("monologg/kobert")
    tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
    
    # 테스트 문장으로 모델 동작 확인
    test_sentence = "안녕하세요. KoBERT 모델을 테스트합니다."
    encoded_input = tokenizer(test_sentence, return_tensors='pt')
    
    with torch.no_grad():
        output = model(**encoded_input)
    
    print("\n모델 로드 및 추론 성공!")
    print(f"테스트 문장: '{test_sentence}'")
    print(f"임베딩 shape: {output.last_hidden_state.shape}")
    
except Exception as e:
    print(f"오류 발생: {e}")
    print("필요한 모든 라이브러리가 설치되었는지 확인해주세요.")
