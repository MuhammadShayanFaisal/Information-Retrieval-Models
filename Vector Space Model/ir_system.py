import os
import math
import tkinter as tk
from collections import defaultdict, Counter

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
    words = text.lower().split()  # tokenization + lowercase
    words = [w for w in words if w.isalpha()]  # remove punctuation
    words = [w for w in words if w not in stopwords]  # remove stopwords
    return words

# -------------------------------
# 🔷 LOAD DOCUMENTS
# -------------------------------
def load_documents(folder_path, stopwords):
    docs = {}
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
            text = f.read()
            docs[filename] = preprocess(text, stopwords)
    return docs

# -------------------------------
# 🔷 BUILD VOCABULARY (ALPHABETICAL)
# -------------------------------
def build_vocabulary(docs):
    vocab = set()
    for words in docs.values():
        vocab.update(words)
    return sorted(list(vocab))  # alphabetical

# -------------------------------
# 🔷 COMPUTE TF
# -------------------------------
def compute_tf(docs):
    tf = {}
    for doc, words in docs.items():
        tf[doc] = Counter(words)
    return tf

# -------------------------------
# 🔷 COMPUTE DF
# -------------------------------
def compute_df(docs, vocab):
    df = {}
    for term in vocab:
        df[term] = sum(1 for doc in docs if term in docs[doc])
    return df

# -------------------------------
# 🔷 COMPUTE IDF
# -------------------------------
def compute_idf(df, N):
    idf = {}
    for term, freq in df.items():
        if freq > 0:
            idf[term] = math.log(N / freq)
        else:
            idf[term] = 0
    return idf

# -------------------------------
# 🔷 BUILD TF-IDF VECTORS
# -------------------------------
def build_vectors(tf, idf, vocab):
    vectors = {}
    for doc, counts in tf.items():
        vector = []
        for term in vocab:
            tf_val = counts.get(term, 0)
            vector.append(tf_val * idf[term])
        vectors[doc] = vector
    return vectors

# -------------------------------
# 🔷 COSINE SIMILARITY
# -------------------------------
def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot / (mag1 * mag2)

# -------------------------------
# 🔷 QUERY VECTOR
# -------------------------------
def query_vector(query, vocab, idf, stopwords):
    words = preprocess(query, stopwords)
    counts = Counter(words)
    
    vector = []
    for term in vocab:
        tf_val = counts.get(term, 0)
        vector.append(tf_val * idf.get(term, 0))
    return vector

# -------------------------------
# 🔷 SEARCH FUNCTION
# -------------------------------
def search(query, vectors, vocab, idf, stopwords, alpha=0.005):
    q_vec = query_vector(query, vocab, idf, stopwords)
    
    scores = []
    for doc, d_vec in vectors.items():
        sim = cosine_similarity(d_vec, q_vec)
        if sim >= alpha:
            scores.append((doc, sim))
    
    # sort descending
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

# -------------------------------
# 🔷 GUI (Tkinter)
# -------------------------------
class SearchGUI:
    def __init__(self, root, vectors, vocab, idf, stopwords):
        self.vectors = vectors
        self.vocab = vocab
        self.idf = idf
        self.stopwords = stopwords
        
        root.title("VSM Search Engine")
        
        self.label = tk.Label(root, text="Enter Query:")
        self.label.pack()
        
        self.entry = tk.Entry(root, width=50)
        self.entry.pack()
        
        self.button = tk.Button(root, text="Search", command=self.run_search)
        self.button.pack()
        
        self.text = tk.Text(root, height=20, width=60)
        self.text.pack()
    
    def run_search(self):
        query = self.entry.get()
        results = search(query, self.vectors, self.vocab, self.idf, self.stopwords)
        
        self.text.delete(1.0, tk.END)
        
        if not results:
            self.text.insert(tk.END, "No relevant documents found.\n")
        else:
            for doc, score in results:
                self.text.insert(tk.END, f"{doc} → {score:.4f}\n")

# -------------------------------
# 🔷 MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    
    # Load stopwords
    stopwords = load_stopwords("stopwords.txt")
    
    # Load documents
    docs = load_documents("speeches", stopwords)
    
    # Build vocabulary
    vocab = build_vocabulary(docs)
    
    # Compute TF, DF, IDF
    tf = compute_tf(docs)
    df = compute_df(docs, vocab)
    idf = compute_idf(df, len(docs))
    
    # Build document vectors
    vectors = build_vectors(tf, idf, vocab)
    
    # Launch GUI
    root = tk.Tk()
    app = SearchGUI(root, vectors, vocab, idf, stopwords)
    root.mainloop()