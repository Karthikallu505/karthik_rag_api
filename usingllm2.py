import scrapy
from scrapy.crawler import CrawlerProcess
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import openai
import json
import os  


embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


openai.api_key = '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'  


def chunk_text(text, chunk_size=200):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def query_embedding(query):
    return embedding_model.encode([query])[0]


def search_vector_database(query, k=5):
    query_vec = np.array([query_embedding(query)])
    distances, indices = index.search(query_vec, k)
    return indices, distances


def generate_response_with_openai(query, context):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"Answer the following question based on the context: {context}\nQuestion: {query}"}
        ]
    )
    return response['choices'][0]['message']['content']

class UniversitySpider(scrapy.Spider):
    name = 'university_spider'
    start_urls = [
        "https://www.uchicago.edu/",
        "https://www.washington.edu/",
        "https://www.stanford.edu/",
        "https://und.edu/"
    ]

    def parse(self, response):
        # Extract text from paragraphs
        text = ' '.join(response.css('p::text').getall())
        yield {'url': response.url, 'content': text}


if not os.path.exists('scraped_data.json'):
    process = CrawlerProcess(settings={
        'FEED_FORMAT': 'json',
        'FEED_URI': 'scraped_data.json',
    })

    process.crawl(UniversitySpider)
    process.start()  


with open('scraped_data.json') as f:
    scraped_data = json.load(f)


chunked_data = {}
embeddings = []
metadata = []

for item in scraped_data:
    url = item['url']
    content = item['content']
    if content:
        chunks = chunk_text(content)
        chunk_embeddings = embedding_model.encode(chunks)
        embeddings.extend(chunk_embeddings)
        metadata.extend([{"url": url, "chunk": i} for i in range(len(chunks))])
        chunked_data[url] = chunks

embeddings = np.array(embeddings)


dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


user_query = input("Enter your query: ")
retrieved_indices, _ = search_vector_database(user_query)


retrieved_chunks = [metadata[i] for i in retrieved_indices[0]]
retrieved_text = "\n ".join([chunked_data[chunk['url']][chunk['chunk']] for chunk in retrieved_chunks])


response = generate_response_with_openai(user_query, retrieved_text)


print("Retrieved Text:")
print(retrieved_text)
print("\nGenerated Response:")
print(response)

# Optionally, print both responses together
print("\n--- Summary ---")
print(f"General Response: {retrieved_text}")
print(f"Generated Response: {response}")
