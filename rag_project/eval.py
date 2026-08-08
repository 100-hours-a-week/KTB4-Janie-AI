import os
from dotenv import load_dotenv

load_dotenv()

os.environ['LANGSMITH_TRACING'] = "true"
os.environ['LANGSMITH_PROJECT'] = "music_rag_eval"
os.environ['LANGSMITH_ENDPOINT']= 'https://apac.api.smith.langchain.com'

from langsmith import Client
from langsmith import traceable
from langsmith.evaluation import evaluate
from main import music_search
from generator import extract_search_params, llm
from retriever import spotify_search_with_check
from vectorstore import vector_store
import asyncio
from main import music_search_stream

# routing accuracy
def routing_accuracy(run, example):
    predicted = run.outputs.get("source")
    expected = example.outputs.get("answer")
    return {'key':'routing_accuracy', 'score': 1 if predicted == expected else 0}

@traceable(name='music_search_eval_target')
def target(inputs: dict) -> dict:
    return music_search(inputs['question'])

@traceable(name='routing_only_eval_target')
def target_routing_only(inputs: dict) -> dict:
    params = extract_search_params(inputs['question'], llm)

    if params.get('intent') == 'out_of_scope':
        return {'source': 'none', 'answer': ''}
    if params.get('intent') == 'youtube_direct':
        return {'source': 'youtube', 'answer': ''}
    
    _, need_fallback = spotify_search_with_check(inputs['question'], params, vector_store)
    return {'source': 'youtube' if need_fallback else 'spotify', 'answer': ''}

def target_streaming(inputs: dict) -> dict:
    async def collect():
        source = None
        full_text = ''
        async for event in music_search_stream(inputs['question']):
            if event['type'] == 'meta':
                source = event['source']
            elif event['type'] in ('token', 'final'):
                full_text = event.get('text', full_text) if event['type'] == 'final' else full_text + event['text']
        return {'source': source, 'answer': full_text}

    return asyncio.run(collect())

if __name__ == '__main__':
    client = Client()

    # # 비스트리밍 (기존)
    # evaluate(
    #     target,
    #     data='music_rag_eval',
    #     evaluators=[routing_accuracy],
    #     experiment_prefix='music-non-streaming'
    # )

    # 스트리밍 (신규)
    evaluate(
        target_streaming,
        data='music_rag_eval',
        evaluators=[routing_accuracy],
        experiment_prefix='music-streaming'
    )


