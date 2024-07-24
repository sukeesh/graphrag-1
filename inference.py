import os
import csv
import redis
import base64
import json
from typing import List, Optional
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
import streamlit as st
import boto3
import logging
from botocore.exceptions import ClientError
import sys
import botocore

acki = ''
sak = ''
sesstoken = ''


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def generate_message(bedrock_runtime, model_id, system_prompt, messages, max_tokens):
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": messages
    })

    response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
    response_body = json.loads(response.get('body').read())
    return response_body


redis_client = redis.Redis(host='localhost', port=6379, db=0)

def embed_text_google(texts: List[str], model_name: str = "text-embedding-004", task: str = "RETRIEVAL_DOCUMENT", dimensionality: Optional[int] = 256) -> List[List[float]]:
    """Embeds texts with a pre-trained Google model."""
    model = TextEmbeddingModel.from_pretrained(model_name)


    def process_batch(batch_texts: List[str]) -> List[List[float]]:
        inputs = [TextEmbeddingInput(text, task) for text in batch_texts]
        kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
        embeddings = model.get_embeddings(inputs, **kwargs)
        return [embedding.values for embedding in embeddings]

    # Split texts into batches of 50
    batch_size = 50
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = process_batch(batch_texts)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings

def generate_embedding(text: str) -> List[float]:
    return embed_text_google([text])[0]

def cosine_similarity(vec1, vec2):
    dot_product = sum(p*q for p, q in zip(vec1, vec2))
    magnitude = (sum([val**2 for val in vec1]) * sum([val**2 for val in vec2])) ** 0.5
    if not magnitude:
        return 0
    return dot_product / magnitude

def get_top_k_closest(query_embedding, k=5):
    keys = redis_client.keys('*')
    records = []
    for key in keys:
        record = redis_client.get(key)
        if record:
            record = json.loads(record)
            if record['vector'] is not None and not record['is_master_node']:
                similarity = cosine_similarity(query_embedding, record['vector'])
                records.append((key, similarity))
    records.sort(key=lambda x: x[1], reverse=True)
    return [record[0] for record in records[:k]]

def bfs_traverse(keys, depth=2):
    visited = set()
    queue = [(key, 0) for key in keys]
    results = []

    while queue:
        current_key, current_depth = queue.pop(0)
        if current_depth > depth:
            break
        if current_key not in visited:
            visited.add(current_key)
            record = redis_client.get(current_key)
            if record:
                record = json.loads(record)
                results.append(record)
                for edge_key in record['edges']:
                    if edge_key not in visited:
                        queue.append((edge_key, current_depth + 1))

    return results

st.title('Legal Search')

query = st.text_input("Enter your query:")

if query:
    query_embedding = generate_embedding(query)
    top_keys = get_top_k_closest(query_embedding, k=1)
    records = bfs_traverse(top_keys, depth=1)

    context = ' '.join(record['text'] for record in records if record['text'])

    try:
        # Initialize the boto3 client with AWS credentials
        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            aws_access_key_id=acki,
            aws_secret_access_key=sak,
            aws_session_token=sesstoken
        )

        model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
        system_prompt = """You are a helpful assistant whose job is to answer queries regarding RBI guidelines.
                           Relevant context is passed below for you to refer. The latest circular or notification can supersede the earlier notification.
                           So, please make sure the get the latest content.
                           Please answer the user query. Please refer this context: """ + context
        max_tokens = 2000

        user_message = {"role": "user", "content": query}
        messages = [user_message]

        response = generate_message(bedrock_runtime, model_id, system_prompt, messages, max_tokens)
        st.markdown("### Answer:", unsafe_allow_html=True)
        answer = response['content'][0]['text']
        answer = answer.replace('\n', '<br>')
        st.markdown(f"<div style='word-wrap: break-word;'>{answer}</div>", unsafe_allow_html=True)
                
        st.markdown("### References:", unsafe_allow_html=True)

        # Using a set to remove duplicate links
        unique_references = set()

        for record in records:
            if 'current_link' in record:
                unique_references.add(record['current_link'])

        # Formatting the references
        if unique_references:
            references_html = "<ul>"
            for link in unique_references:
                references_html += f"<li><a href='{link}' target='_blank'>{link}</a></li>"
            references_html += "</ul>"
            
            st.markdown(references_html, unsafe_allow_html=True)
        
    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occurred: " + format(message))
