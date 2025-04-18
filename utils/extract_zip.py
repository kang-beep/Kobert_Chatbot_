"""
AI 허브 데이터셋 압축 파일 해제 스크립트
작성자: 서강산
작성일: 2024-10-08
수정일: 2024-10-08 - 경로 수정
"""

import os
import zipfile
import sys
import shutil
from pathlib import Path

def check_zip_file(zip_path):
    """
    압축 파일의 유효성을 확인합니다.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print(f"압축 파일 정보: {zip_path}")
            print(f"압축 파일 크기: {os.path.getsize(zip_path) / (1024*1024):.2f} MB")
            print(f"파일 목록: {len(zip_ref.namelist())} 개의 파일")
            
            # 압축 파일 내 첫 10개 파일 표시
            for i, name in enumerate(zip_ref.namelist()[:10]):
                print(f"  - {name}")
            
            if len(zip_ref.namelist()) > 10:
                print(f"  - ... 그 외 {len(zip_ref.namelist()) - 10}개 파일")
            
            print("압축 파일 테스트 중...")
            test_result = zip_ref.testzip()
            if test_result is None:
                print("압축 파일 검증 완료: 문제 없음")
                return True
            else:
                print(f"압축 파일 손상 발견: {test_result}")
                return False
    except zipfile.BadZipFile:
        print(f"잘못된 ZIP 파일 형식입니다: {zip_path}")
        return False
    except Exception as e:
        print(f"압축 파일 확인 중 오류 발생: {e}")
        return False

def extract_zip_file(zip_path, extract_path):
    """
    압축 파일을 지정된 경로에 해제합니다.
    """
    try:
        # 이미 존재하는 폴더가 있으면 삭제
        if os.path.exists(extract_path):
            print(f"기존 폴더 {extract_path} 삭제 중...")
            shutil.rmtree(extract_path)
        
        # 추출 폴더 생성
        os.makedirs(extract_path, exist_ok=True)
        
        print(f"압축 해제 시작: {zip_path} -> {extract_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        print(f"압축 해제 완료: {extract_path}")
        
        # 해제된 파일 목록 출력
        extracted_files = list(Path(extract_path).rglob('*'))
        print(f"해제된 파일/폴더 수: {len(extracted_files)}")
        
        return True
    except Exception as e:
        print(f"압축 해제 중 오류 발생: {e}")
        return False

def main():
    # 압축 파일 위치 (현재 프로젝트 기준 경로로 수정)
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Training")
    source_data_path = os.path.join(base_path, "01.원천데이터")
    labeled_data_path = os.path.join(base_path, "02.라벨링데이터")
    
    print(f"데이터 기본 경로: {base_path}")
    print(f"원천 데이터 경로: {source_data_path}")
    print(f"라벨링 데이터 경로: {labeled_data_path}")
    
    # 압축 파일 목록
    source_zip_files = [
        os.path.join(source_data_path, "TS_session2.zip"),
        os.path.join(source_data_path, "TS_session3.zip"),
        os.path.join(source_data_path, "TS_session4.zip")
    ]
    
    labeled_zip_files = [
        os.path.join(labeled_data_path, "TL_session2.zip"),
        os.path.join(labeled_data_path, "TL_session3.zip"),
        os.path.join(labeled_data_path, "TL_session4.zip")
    ]
    
    print("=" * 50)
    print("AI 허브 데이터셋 압축 파일 확인 및 해제 도구")
    print("=" * 50)
    
    # 원천 데이터 확인 및 해제
    print("\n[원천 데이터 압축 파일 확인]")
    for zip_file in source_zip_files:
        if os.path.exists(zip_file):
            is_valid = check_zip_file(zip_file)
            if is_valid:
                print(f"\n{zip_file} 파일을 해제하시겠습니까? (y/n): ", end="")
                response = input().strip().lower()
                if response == 'y':
                    extract_path = os.path.join(source_data_path, os.path.basename(zip_file).replace(".zip", "_extracted"))
                    extract_zip_file(zip_file, extract_path)
            print("-" * 50)
        else:
            print(f"파일이 존재하지 않습니다: {zip_file}")
    
    # 라벨링 데이터 확인 및 해제
    print("\n[라벨링 데이터 압축 파일 확인]")
    for zip_file in labeled_zip_files:
        if os.path.exists(zip_file):
            is_valid = check_zip_file(zip_file)
            if is_valid:
                print(f"\n{zip_file} 파일을 해제하시겠습니까? (y/n): ", end="")
                response = input().strip().lower()
                if response == 'y':
                    extract_path = os.path.join(labeled_data_path, os.path.basename(zip_file).replace(".zip", "_extracted"))
                    extract_zip_file(zip_file, extract_path)
            print("-" * 50)
        else:
            print(f"파일이 존재하지 않습니다: {zip_file}")

if __name__ == "__main__":
    main() 