"""
data_loader.py
원본 데이터(예: namuwiki-extracted 데이터셋)를 읽어와
표준화된 문서 리스트(dict)로 변환하는 역할만 담당.
"""

from pathlib import Path
from typing import List, Dict


def load_documents(source_path: Path) -> List[Dict]:
    """
    원본 소스에서 문서를 읽어 표준 포맷으로 반환.
    반환 형식: [{"id": str, "title": str, "text": str, "source": str}, ...]

    TODO: 실제 데이터 소스에 맞게 구현
    - 로컬 텍스트/JSON 파일이면 여기서 파싱
    - HuggingFace datasets 쓰면 load_dataset() 결과를 이 포맷으로 매핑
    """
    documents = []

    # 예시: JSON 파일 로딩 로직 (실제 데이터 형식에 맞춰 수정 필요)
    # import json
    # with open(source_path, "r", encoding="utf-8") as f:
    #     raw = json.load(f)
    # for item in raw:
    #     documents.append({
    #         "id": item["id"],
    #         "title": item.get("title", ""),
    #         "text": item["text"],
    #         "source": str(source_path),
    #     })

    return documents


def load_from_huggingface(dataset_name: str, split: str = "train") -> List[Dict]:
    """
    HuggingFace datasets에서 직접 로딩할 때 사용.
    예: load_from_huggingface("heegyu/namuwiki-extracted")
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)

    documents = []
    for i, row in enumerate(dataset):
        documents.append({
            "id": str(i),
            "title": row.get("title", ""),
            "text": row.get("text", ""),
            "source": dataset_name,
        })

    return documents
