"""
AI 허브 멀티세션 대화 데이터 처리 유틸리티
작성자: 서강산
작성일: 2024-10-08
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def load_json_file(file_path: str) -> Dict:
    """
    JSON 파일을 로드합니다.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"파일 '{file_path}' 로드 중 오류 발생: {e}")
        return {}

def get_session_files(data_dir: str, session_level: Optional[List[int]] = None) -> List[str]:
    """
    지정된 디렉토리에서 세션 레벨에 맞는 JSON 파일 목록을 반환합니다.
    
    Args:
        data_dir: 데이터 디렉토리 경로
        session_level: 세션 레벨 목록 (예: [2, 3, 4])
    """
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    if not session_level:
        return json_files
    
    # 세션 레벨에 맞는 파일만 필터링
    filtered_files = []
    for file_path in json_files:
        try:
            data = load_json_file(file_path)
            file_session_level = int(data.get("FileInfo", {}).get("sessionLevel", "0"))
            if file_session_level in session_level:
                filtered_files.append(file_path)
        except Exception as e:
            print(f"파일 '{file_path}' 처리 중 오류 발생: {e}")
    
    return filtered_files

def extract_dialog_data(json_data: Dict) -> List[Tuple[str, str, Dict]]:
    """
    JSON 데이터에서 대화 데이터를 추출합니다.
    
    Returns:
        List of (question, answer, metadata) tuples
    """
    dialog_data = []
    
    # 페르소나 정보 추출
    persona_info = {
        "speaker1": json_data.get("personaInfo", {}).get("clInfo", {}).get("personaFeatures", []),
        "speaker2": json_data.get("personaInfo", {}).get("cpInfo", {}).get("personaFeatures", [])
    }
    
    # 세션별 대화 추출
    session_info_list = json_data.get("sessionInfo", [])
    for session_info in session_info_list:
        dialog = session_info.get("dialog", [])
        
        # 각 세션의 페르소나 요약 정보
        session_persona_summary = session_info.get("sessionPersonaSummary", {})
        
        # 대화 쌍 구성
        for i in range(0, len(dialog) - 1, 2):
            if i + 1 >= len(dialog):
                break
                
            question = dialog[i]["utterance"]
            answer = dialog[i + 1]["utterance"]
            
            # 메타데이터 구성
            metadata = {
                "session_id": session_info.get("sessionID", ""),
                "topic": json_data.get("topicInfo", {}).get("topicTitle", ""),
                "persona": persona_info,
                "session_persona_summary": session_persona_summary,
                "question_summary": dialog[i].get("summary", ""),
                "answer_summary": dialog[i + 1].get("summary", ""),
                "date": dialog[i].get("date", ""),
                "terminate": dialog[i + 1].get("terminate", "false") == "true"
            }
            
            dialog_data.append((question, answer, metadata))
    
    return dialog_data

def process_multisession_data(data_dir: str, session_level: List[int]) -> List[Tuple[str, str, Dict]]:
    """
    지정된 디렉토리의 멀티세션 대화 데이터를 처리하여 학습용 데이터셋을 구성합니다.
    
    Args:
        data_dir: 라벨링 데이터 디렉토리 경로
        session_level: 세션 레벨 목록 (예: [2, 3, 4])
    """
    all_dialog_data = []
    
    # 세션 레벨에 맞는 파일 목록 가져오기
    json_files = get_session_files(data_dir, session_level)
    print(f"처리할 파일 수: {len(json_files)}")
    
    # 각 파일마다 대화 데이터 추출
    for file_path in json_files:
        json_data = load_json_file(file_path)
        if not json_data:
            continue
            
        dialog_data = extract_dialog_data(json_data)
        all_dialog_data.extend(dialog_data)
    
    print(f"총 추출된 대화 쌍 수: {len(all_dialog_data)}")
    return all_dialog_data

def augment_with_persona(question: str, metadata: Dict, use_persona: bool = True) -> str:
    """
    페르소나 정보를 사용하여 질문을 보강합니다.
    """
    if not use_persona:
        return question
        
    # 화자 페르소나 정보
    speaker1_persona = metadata["persona"]["speaker1"]
    persona_text = f"[페르소나 정보] {' '.join(speaker1_persona)}"
    
    # 주제 정보
    topic_text = f"[주제] {metadata['topic']}"
    
    # 세션 페르소나 요약 정보
    session_persona_summary = metadata["session_persona_summary"].get("speaker1", [])
    summary_text = f"[대화 요약] {' '.join(session_persona_summary)}"
    
    # 보강된 질문
    augmented_question = f"{persona_text}\n{topic_text}\n{summary_text}\n[질문] {question}"
    return augmented_question

def split_data_train_valid(dialog_data: List[Tuple[str, str, Dict]], valid_ratio: float = 0.1) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    데이터를 학습용과 검증용으로 분할합니다.
    """
    import random
    random.shuffle(dialog_data)
    
    valid_size = int(len(dialog_data) * valid_ratio)
    train_data = dialog_data[:-valid_size] if valid_size > 0 else dialog_data
    valid_data = dialog_data[-valid_size:] if valid_size > 0 else []
    
    # 간단히 (question, answer) 쌍으로 변환
    train_pairs = [(augment_with_persona(q, meta), a) for q, a, meta in train_data]
    valid_pairs = [(augment_with_persona(q, meta), a) for q, a, meta in valid_data]
    
    return train_pairs, valid_pairs

def save_dataset(data_pairs: List[Tuple[str, str]], file_path: str) -> None:
    """
    데이터셋을 파일로 저장합니다.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for question, answer in data_pairs:
            f.write(f"{question}\t{answer}\n")
    
    print(f"데이터셋이 '{file_path}'에 저장되었습니다. 총 {len(data_pairs)}개 대화 쌍") 