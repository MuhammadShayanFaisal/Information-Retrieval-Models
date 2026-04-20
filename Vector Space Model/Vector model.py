import os                                                                              
import math
import json
import re
from collections import Counter, defaultdict
import nltk
from nltk.stem import WordNetLemmatizer

def clean_doc_id(filename):
    name = filename.replace(".txt", "")
    name = name.lower().replace("speech", "").replace("_", "").replace(" ", "")
    return name

def tokenize(text):
    """Basic tokenization: lowercase + remove punctuation only (no stopword removal, no lemmatization)"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text.split()

def preprocess(text, stopwords):
    """Full preprocessing for inverted index: tokenize + stopword removal + lemmatization"""
    words = tokenize(text)
    lemmatizer = WordNetLemmatizer()
    result = []
    for word in words:
        if word not in stopwords:
            result.append(lemmatizer.lemmatize(word))
    return result

def load_documents(folder_path, stopwords):
    """Load documents in two forms: raw tokens (for TF-IDF) and preprocessed (for inverted index)"""
    raw_docs = {}       # for TF-IDF: tokenized only
    clean_docs = {}     # for inverted index: stopwords removed + lemmatized
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), 'r') as f:
            text = f.read()
            clean_id = clean_doc_id(filename)
            raw_docs[clean_id] = tokenize(text)
            clean_docs[clean_id] = preprocess(text, stopwords)
    return raw_docs, clean_docs

def build_vocabulary(docs):
    vocab = set()
    for words in docs.values():
        vocab.update(words)
    return sorted(list(vocab))

def compute_tf(docs):
    return {doc: Counter(words) for doc, words in docs.items()}

def compute_df(docs, vocab):
    return {term: sum(1 for d in docs if term in docs[d]) for term in vocab}

def compute_idf(df, N):
    return {term: math.log(N / df[term]) if df[term] else 0 for term in df}

def build_vectors(tf, idf, vocab):
    vectors = {}
    for doc, counts in tf.items():
        vectors[doc] = [counts.get(term, 0) * idf[term] for term in vocab]
    return vectors

def build_inverted_index(clean_docs):
    """Built from preprocessed docs (stopwords removed + lemmatized)"""
    index = defaultdict(list)
    for doc, words in clean_docs.items():
        for word in set(words):
            index[word].append(doc)
    return dict(index)

def cosine_similarity(v1, v2):
    dot = sum(a*b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a*a for a in v1))
    mag2 = math.sqrt(sum(b*b for b in v2))
    return dot/(mag1*mag2) if mag1 and mag2 else 0

def query_vector(query, vocab, idf):
    """Query uses raw tokenization to match TF-IDF vocabulary"""
    words = tokenize(query)
    counts = Counter(words)
    return [counts.get(term, 0) * idf.get(term, 0) for term in vocab]

def search(query, vectors, vocab, idf, alpha=0.005):
    q_vec = query_vector(query, vocab, idf)
    results = []
    for doc, vec in vectors.items():
        score = cosine_similarity(vec, q_vec)
        if score >= alpha:
            results.append((doc, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def save_indexes(vocab, inverted_index, vectors):
    os.makedirs("indexes", exist_ok=True)
    
    with open("indexes/vocab.txt", "w", encoding='utf-8') as f:
        f.write("\n".join(sorted(vocab)))
    
    with open("indexes/inverted_index.txt", "w", encoding='utf-8') as f:
        for term in sorted(inverted_index.keys()):
            f.write(f"{term}: {', '.join(sorted(inverted_index[term]))}\n")
    
    with open("indexes/vectors.json", "w") as f:
        json.dump(vectors, f)

    print("💾 Indexes saved in 'indexes/' folder as .txt files\n")

def display_results(query, results):
    doc_ids = [doc for doc, _ in results]
    print(f"Length={len(doc_ids)}")
    print(set(doc_ids))

print("Loading system...\n")
stopwords = ["a", "is", "the", "of", "all", "and", "to", "can", "be", "as","once", "for", "at", "am", "are", "has", "have", "had", "up", 
    "his", "her", "in", "on", "no", "we", "do"]

# Load two versions of docs
raw_docs, clean_docs = load_documents("Trump Speeches", stopwords)

# TF-IDF pipeline uses raw_docs (tokenized only)
vocab = build_vocabulary(raw_docs)
tf = compute_tf(raw_docs)
df = compute_df(raw_docs, vocab)
idf = compute_idf(df, len(raw_docs))
vectors = build_vectors(tf, idf, vocab)

# Inverted index uses clean_docs (stopwords removed + lemmatized)
inverted_index = build_inverted_index(clean_docs)

save_indexes(vocab, inverted_index, vectors)

print("System Ready (type 'exit' to quit)\n")
while True:
    query = input("🔍 Enter Query: ")
    if query.lower() == "exit":
        break
    results = search(query, vectors, vocab, idf)
    display_results(query, results)