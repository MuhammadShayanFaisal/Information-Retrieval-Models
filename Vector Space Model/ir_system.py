import os
import math
import json
from collections import Counter, defaultdict

def clean_doc_id(filename):
    # remove extension
    name = filename.replace(".txt", "")
    
    # remove 'speech', '_', spaces
    name = name.lower().replace("speech", "").replace("_", "").replace(" ", "")
    
    return name
# -------------------------------
# 🔷 LOAD STOPWORDS
# -------------------------------
def load_stopwords(path):
    with open(path, 'r') as f:
        return set(word.strip() for word in f.readlines())

# -------------------------------
# 🔷 PREPROCESSING
# -------------------------------
def preprocess(text, stopwords):
    words = text.lower().split()
    words = [w for w in words if w.isalpha()]
    words = [w for w in words if w not in stopwords]
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    for word in words:
        print(word+' -> '+lemmatizer.lemmatize(word))
    return words

# -------------------------------
# 🔷 LOAD DOCUMENTS
# -------------------------------
def load_documents(folder_path, stopwords):
    docs = {}
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
            clean_id = clean_doc_id(filename)
            docs[clean_id] = preprocess(f.read(), stopwords)
    return docs
# -------------------------------
# 🔷 BUILD VOCABULARY
# -------------------------------
def build_vocabulary(docs):
    vocab = set()
    for words in docs.values():
        vocab.update(words)
    return sorted(list(vocab))  # alphabetical

# -------------------------------
# 🔷 TF, DF, IDF
# -------------------------------
def compute_tf(docs):
    return {doc: Counter(words) for doc, words in docs.items()}

def compute_df(docs, vocab):
    return {term: sum(1 for d in docs if term in docs[d]) for term in vocab}

def compute_idf(df, N):
    return {term: math.log(N / df[term]) if df[term] else 0 for term in df}

# -------------------------------
# 🔷 BUILD VECTORS
# -------------------------------
def build_vectors(tf, idf, vocab):
    vectors = {}
    for doc, counts in tf.items():
        vectors[doc] = [counts.get(term, 0) * idf[term] for term in vocab]
    return vectors

# -------------------------------
# 🔷 INVERTED INDEX
# -------------------------------
def build_inverted_index(docs):
    index = defaultdict(list)
    for doc, words in docs.items():
        for word in set(words):
            index[word].append(doc)  # already clean now
    return dict(index)

# -------------------------------
# 🔷 COSINE SIMILARITY
# -------------------------------
def cosine_similarity(v1, v2):
    dot = sum(a*b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a*a for a in v1))
    mag2 = math.sqrt(sum(b*b for b in v2))
    return dot/(mag1*mag2) if mag1 and mag2 else 0

# -------------------------------
# 🔷 QUERY VECTOR
# -------------------------------
def query_vector(query, vocab, idf, stopwords):
    words = preprocess(query, stopwords)
    counts = Counter(words)
    return [counts.get(term, 0) * idf.get(term, 0) for term in vocab]

# -------------------------------
# 🔷 SEARCH
# -------------------------------
def search(query, vectors, vocab, idf, stopwords, alpha=0):
    q_vec = query_vector(query, vocab, idf, stopwords)
    
    results = []
    for doc, vec in vectors.items():
        score = cosine_similarity(vec, q_vec)
        if score >= alpha:
            results.append((doc, score))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# -------------------------------
# 🔷 SAVE INDEXES
# -------------------------------
def save_indexes(vocab, inverted_index, vectors):
    os.makedirs("indexes", exist_ok=True)
    
    with open("indexes/vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)
    
    with open("indexes/inverted_index.json", "w") as f:
        json.dump(inverted_index, f, indent=2)
    
    with open("indexes/vectors.json", "w") as f:
        json.dump(vectors, f)

    print("💾 Indexes saved in 'indexes/' folder\n")

# -------------------------------
# 🔷 FORMAT OUTPUT (IMPORTANT)
# -------------------------------
def display_results(query, results):
    doc_ids = [doc for doc, _ in results]  # already clean
    print(f"Length={len(doc_ids)}")
    print(results)
    # print(set(doc_ids))

# -------------------------------
# 🔷 MAIN
# -------------------------------
if __name__ == "__main__":
    
    print("🔄 Loading system...\n")
    
    stopwords = load_stopwords("Stopword-List.txt")
    docs = load_documents("Trump Speeches", stopwords)
    
    vocab = build_vocabulary(docs)
    tf = compute_tf(docs)
    df = compute_df(docs, vocab)
    idf = compute_idf(df, len(docs))
    
    vectors = build_vectors(tf, idf, vocab)
    inverted_index = build_inverted_index(docs)
    
    # save indexes
    save_indexes(vocab, inverted_index, vectors)
    
    print("✅ System Ready (type 'exit' to quit)\n")
    
    while True:
        query = input("🔍 Enter Query: ")
        
        if query.lower() == "exit":
            break
        
        results = search(query, vectors, vocab, idf, stopwords)
        display_results(query, results)