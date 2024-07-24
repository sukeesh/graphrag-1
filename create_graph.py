import os
import csv
import redis
import base64
import json
from typing import List, Optional
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel


redis_client = redis.Redis(host='localhost', port=6379, db=0)

def embed_text_google(texts: List[str], model_name: str = "text-embedding-004", task: str = "RETRIEVAL_DOCUMENT", dimensionality: Optional[int] = 256) -> List[List[float]]:


    model = TextEmbeddingModel.from_pretrained(model_name)

    # Helper function to process a batch of texts
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


def create_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def process_record(record, csv_file_path):
    current_link, title, date, supersedes_links, notification_user, notification_id = record
    file_path = os.path.join(notification_user, f"{notification_id}.txt")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Generate the master key
    master_key = base64.urlsafe_b64encode(f"{current_link}_master".encode()).decode()

    # Check if the record has already been processed
    if redis_client.exists(master_key):
        print(f"Record for {current_link} has already been processed. Skipping.")
        return

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    chunks = create_chunks(content, chunk_size=3000, overlap=100)

    # Prepare edges from supersedes_links
    supersede_links_list = supersedes_links.split('; ')
    edges = [base64.urlsafe_b64encode(link.encode()).decode() for link in supersede_links_list]

    # Store the master node with edges to both supersede nodes and chunk nodes
    master_node = {
        "vector": None,
        "text": None,
        "edges": edges,
        "current_link": current_link,
        "is_master_node": True
    }

    chunk_keys = []
    for idx, chunk in enumerate(chunks):
        chunk_key = base64.urlsafe_b64encode(f"{current_link}_{idx}".encode()).decode()
        chunk_node = {
            "vector": generate_embedding(chunk),
            "text": chunk,
            "edges": [master_key],
            "current_link": current_link,
            "is_master_node": False
        }
        redis_client.set(chunk_key, json.dumps(chunk_node))
        chunk_keys.append(chunk_key)
        print(f"Stored chunk {idx} for {current_link}")

    # Add chunk keys to the master node's edges
    master_node["edges"].extend(chunk_keys)
    redis_client.set(master_key, json.dumps(master_node))

def process_csv(csv_file_path):
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip header row
        for record in csv_reader:
            process_record(record, csv_file_path)

def main():
    # Define the paths to your CSV files
    notifications_csv_path = "notifications_metadata.csv"
    master_directions_csv_path = "master_directions_metadata.csv"

    # Process each CSV file
    process_csv(notifications_csv_path)
    process_csv(master_directions_csv_path)

if __name__ == "__main__":
    main()