import scrapy
from scrapy.crawler import CrawlerProcess
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import openai
import json
import os  # Import os for file checking

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Set your OpenAI API key
openai.api_key = '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'  # Replace with your actual API key

# Function to chunk text
def chunk_text(text, chunk_size=200):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Function to query embedding
def query_embedding(query):
    return embedding_model.encode([query])[0]

# Function to perform similarity search in FAISS
def search_vector_database(query, k=5):
    query_vec = np.array([query_embedding(query)])
    distances, indices = index.search(query_vec, k)
    return indices, distances

# Function to generate response using OpenAI API
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

# Run the Scrapy spider only if scraped_data.json doesn't exist
if not os.path.exists('scraped_data.json'):
    process = CrawlerProcess(settings={
        'FEED_FORMAT': 'json',
        'FEED_URI': 'scraped_data.json',
    })

    process.crawl(UniversitySpider)
    process.start()  # The script will block here until the crawling is finished

# Load the scraped data
with open('scraped_data.json') as f:
    scraped_data = json.load(f)

# Chunk and embed data
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

# Store embeddings in FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# User query
user_query = input("Enter your query: ")
retrieved_indices, _ = search_vector_database(user_query)

# Retrieve relevant chunks
retrieved_chunks = [metadata[i] for i in retrieved_indices[0]]
retrieved_text = "\n ".join([chunked_data[chunk['url']][chunk['chunk']] for chunk in retrieved_chunks])

# Generate response using OpenAI
response = generate_response_with_openai(user_query, retrieved_text)

# Print results
print("Retrieved Text:")
print(retrieved_text)
print("\nGenerated Response:")
print(response)

# Optionally, print both responses together
print("\n--- Summary ---")
print(f"General Response: {retrieved_text}")
print(f"Generated Response: {response}")