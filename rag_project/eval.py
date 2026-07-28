import os
from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate
from main import music_search

load_dotenv()

os.environ['LANGSMITH_TRACING'] = "true"
os.environ['LANGSMITH_PROJECT'] = "music_rag_eval"


def build_dataset(client: Client):
    dataset = client.create_dataset(dataset_name='music_rag_eval')
    examples = [
        {'inputs': {'question': '비 오는 날 듣기 좋은 노래 추천해줘'}, 'outputs': {'expected_source': 'spotify'}},
        {'inputs': {'question': 'Olivia Rodrigo 노래 추천해줘'}, 'outputs': {'expected_source': 'spotify'}},
        {'inputs': {'question': '소수빈 라이브 영상 추천해줘'}, 'outputs': {'expected_source': 'youtube'}},
        {'inputs': {'question': '클래식 연주 영상 추천해줘'}, 'outputs': {'expected_source': 'youtube'}},
        {"inputs": {"question": "5SOS 노래 추천해줘"}, "outputs": {"expected_source": "spotify"}},
        {"inputs": {"question": "5 Seconds of Summer 노래 추천해줘"}, "outputs": {"expected_source": "spotify"}}
    ]
    client.create_examples(dataset_id=dataset.id, examples=examples)
    return dataset

# routing accuracy
def routing_accuracy(run, example):
    predicted = run.outputs.get("source")
    expected = example.outputs.get("expected_source")
    return {'key':'routing_accuracy', 'score': 1 if predicted == expected else 0}

def target(inputs: dict) -> dict:
    return music_search(inputs['question'])

if __name__ == '__main__':
    client = Client()

    try:
        existing = client.read_dataset(dataset_name='music_rag_eval')
        client.delete_dataset(dataset_id=existing.id)
    except Exception:
        pass

    build_dataset(client)
    evaluate(
        target,
        data='music_rag_eval',
        evaluators=[routing_accuracy]
    )

