from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# Query
query = "I want to buy cheap ticket to Hawaii"

# Sample documents
documents = ['0.txt', '1.txt', '2.txt', '3.txt']

# Function to read the documents
def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

# Read the documents
documents = [read_file(doc) for doc in documents]

'''
Fuzzy Matching
'''
from rapidfuzz import fuzz, process

scores = process.extract(query, documents, scorer=fuzz.token_sort_ratio)

# Normalize the scores from 0-100 to 0-1
for each in scores:
    print(f'score: {each[1]/100}: index: {each[2]}')

'''
TF-IDF
'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, process, utils

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Combine documents and query for the purpose of TF-IDF fitting
all_texts = documents + [query]

# Fit the vectorizer on the combined text and transform
tfidf_matrix = vectorizer.fit_transform(all_texts)

# Separate the TF-IDF matrix into document vectors and query vector
doc_vectors = tfidf_matrix[:-1]
query_vector = tfidf_matrix[-1]

# Compute cosine similarity between the query vector and document vectors
cosine_similarities = cosine_similarity(query_vector, doc_vectors).flatten()

# Display the cosine similarities
for idx, similarity in enumerate(cosine_similarities):
    print(f"Document {idx}: {similarity:.4f}")

'''
BM25
'''
from rank_bm25 import BM25Okapi
from rapidfuzz import utils

def fuzz_full_process(text):
    return utils.default_process(text)

# Transform all characters into lower case
processed_documents = [fuzz_full_process(doc).split() for doc in documents]

bm25 = BM25Okapi(processed_documents)

# Tokenization
fuzz_processed_query = fuzz_full_process(query).split()

doc_scores = bm25.get_scores(fuzz_processed_query)

print(doc_scores)

'''
Embedding Similarity
'''

client = OpenAI(api_key='YOUR_OPENAI_API_KEY')

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def cosine_similarity_score(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

# Get embeddings
embeddings = [get_embedding(doc) for doc in documents]

# Calculate similarity score
similarity_scores = [cosine_similarity_score(embedding, get_embedding(query)) for embedding in embeddings]

# Print out similarity scores
print(similarity_scores)