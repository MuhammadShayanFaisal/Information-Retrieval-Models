import os                                                                              
import math
import json
import re
from collections import Counter, defaultdict
import nltk
from nltk.stem import WordNetLemmatizer

def clean_doc_id(filename):
    name = filename.replace(".txt", "")                                                                                               # eliminate the .txt name in file
    name = name.lower().replace("speech", "").replace("_", "").replace(" ", "")                                                       # eliminate the 'speech' in filename to get only file number
    return name

def preprocess(text, stopwords):
    text = text.lower()                                                                                                              # case fold
    text = re.sub(r'[^a-z0-9\s]', ' ', text)                                                                                         # remove punctuation
    words = text.split() 
    result = []      
    lemmatizer = WordNetLemmatizer()                   
    for word in words:
        if word not in stopwords:                                                                                                    # checking if word is not a stopword then go for other preprocessing steps
            result.append(lemmatizer.lemmatize(word))                                                                                        # lemmatization
    return result                                                                                                                           # tokenize

def load_documents(folder_path, stopwords):                                                                                        # load the document from 56 Trump speeches folder
    docs = {}
    for filename in os.listdir(folder_path):                                                                                      # look for every file in Folder
        with open(os.path.join(folder_path, filename), 'r') as f:                                                                  # Read the file content
            clean_id = clean_doc_id(filename)                                                                                      # extracting the file number
            docs[clean_id] = preprocess(f.read(), stopwords)                                                        # preprocess the content by case folding, remove stop words and lemmatization
    return docs

def build_vocabulary(docs):
    vocab = set()                                                                                                                 # make set of unique words in vacabulary 
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

def build_inverted_index(docs):
    index = defaultdict(list)
    for doc, words in docs.items():
        for word in set(words):
            index[word].append(doc)  # already clean now
    return dict(index)

def cosine_similarity(v1, v2):
    dot = sum(a*b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a*a for a in v1))
    mag2 = math.sqrt(sum(b*b for b in v2))
    return dot/(mag1*mag2) if mag1 and mag2 else 0

def query_vector(query, vocab, idf, stopwords):
    words = preprocess(query, stopwords)
    counts = Counter(words)
    return [counts.get(term, 0) * idf.get(term, 0) for term in vocab]

def search(query, vectors, vocab, idf, stopwords, alpha=0.005):
    q_vec = query_vector(query, vocab, idf, stopwords)
    
    results = []
    for doc, vec in vectors.items():
        score = cosine_similarity(vec, q_vec)
        if score >= alpha:
            results.append((doc, score))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def save_indexes(vocab, inverted_index, vectors):
    os.makedirs("indexes", exist_ok=True)
    
    with open("indexes/vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)
    
    with open("indexes/inverted_index.json", "w") as f:
        json.dump(inverted_index, f, indent=2)

    print("💾 Indexes saved in 'indexes/' folder\n")

def display_results(query, results):
    doc_ids = [doc for doc, _ in results]  # already clean
    print(f"Length={len(doc_ids)}")
    print(set(doc_ids))

print("Loading system...\n")
stopwords = ["a", "is", "the", "of", "all", "and", "to", "can", "be", "as","once", "for", "at", "am", "are", "has", "have", "had", "up", 
    "his", "her", "in", "on", "no", "we", "do" ]
docs = load_documents("Trump Speeches", stopwords)
vocab = build_vocabulary(docs)
tf = compute_tf(docs)
df = compute_df(docs, vocab)
idf = compute_idf(df, len(docs))
    
vectors = build_vectors(tf, idf, vocab)
inverted_index = build_inverted_index(docs)
    
save_indexes(vocab, inverted_index, vectors)
    
print("✅ System Ready (type 'exit' to quit)\n")
while True:
    query = input("🔍 Enter Query: ")
        
    if query.lower() == "exit":
        break
        
    results = search(query, vectors, vocab, idf, stopwords)
    display_results(query, results)