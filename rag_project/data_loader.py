from pathlib import Path


def load_markdown(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return {"text": text, "metadata": {"source": filepath, "format": "markdown"}}


def load_document(filepath):
    if filepath.endswith(".md"):
        return load_markdown(filepath)
    raise ValueError(f"Unsupported file format: {filepath}")


def load_documents_from_folder(folder_path):
    folder = Path(folder_path)
    docs = []
    for path in sorted(folder.glob("*.md")):
        docs.append(load_document(str(path)))
    return docs
