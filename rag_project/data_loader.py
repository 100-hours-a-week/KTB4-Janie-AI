from datasets import load_dataset
from langchain_core.documents import Document
from config import DATASET_NAME

def row_to_text(row):
    popular = "인기 있는" if row["popularity"] >= 50 else "숨겨진"

    if row["energy"] > 0.6 and row["valence"] > 0.6:
        mood = "신나고 밝은"
    elif row["energy"] > 0.6 and row["valence"] <= 0.6:
        mood = "격렬하지만 어두운"
    elif row["energy"] <= 0.6 and row["valence"] > 0.6:
        mood = "잔잔하지만 밝은"
    else:
        mood = "차분하고 우울한"

    dance = "춤추기 좋은" if row["danceability"] > 0.65 else "춤추기보단 감상용인"
    acoustic = "어쿠스틱한" if row["acousticness"] > 0.6 else "전자음 중심의"
    live = "현장감 있는" if row["liveness"] >= 0.65 else "스튜디오 녹음 느낌의"
    vocal = "보컬 없는 연주곡" if row["instrumentalness"] > 0.5 else "보컬이 있는 곡"
    rap = ", 랩/스포큰워드 비중이 높은" if row["speechiness"] > 0.33 else ""

    return (
        f"곡: {row['track_name']} - 아티스트: {row['artists']}\n"
        f"장르: {row['track_genre']}\n"
        f"특징: {popular}, {mood}, {dance}, {acoustic}, {live}, {vocal}{rap}\n"
        f"템포: {row['tempo']:.0f}BPM"
    )


def load_spotify_documents():
    spotify_data = load_dataset(DATASET_NAME)
    docs = [
        Document(
            page_content=row_to_text(row),
            metadata={'genre': row['track_genre'], 'artists': row['artists']}
        )
        for row in spotify_data['train']
    ]
    print(len(docs))
    return docs

