import os                                                                                       # for file/folder operations
import math                                                                                     # for log and sqrt in TF-IDF and cosine
import json                                                                                     # for saving/loading vectors as JSON
import re                                                                                       # for regex-based punctuation removal
from collections import Counter, defaultdict                                                    # Counter for word freq, defaultdict for index
import nltk                                       
from nltk.stem import WordNetLemmatizer        
import tkinter as tk                                                                            # for GUI components
from tkinter import scrolledtext                                                                

def clean_doc_id(filename):
    name = filename.replace(".txt", "")                                                          # eliminate the .txt name in file
    name = name.lower().replace("speech", "").replace("_", "").replace(" ", "")                  # # eliminate the 'speech' in filename to get only file number
    return name                                                           

def tokenize(text):
    text = text.lower()                                                                          # convert all characters to lowercase (case folding)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)                                                     # replace punctuation/special chars with spaces
    return text.split()                            

def preprocess(text, stopwords):
    words = tokenize(text)                         # first apply basic tokenization
    lemmatizer = WordNetLemmatizer()               # initialize lemmatizer (converts e.g. 'running' → 'run')
    result = []
    for word in words:
        if word not in stopwords:                                                                # checking if word is not a stopword then go for other preprocessing steps
            result.append(lemmatizer.lemmatize(word))                                            # lemmatization
    return result                                                                                # return cleaned, meaningful token list

def load_documents(folder_path, stopwords):                                                     # load the document from 56 Trump speeches folder
    raw_docs = {}                                                                               # doc_id → list of raw tokens (for TF-IDF)
    clean_docs = {}                                                                             # doc_id → list of preprocessed tokens (for inverted index)

    for filename in os.listdir(folder_path):                                                    # look for every file in Folder
        with open(os.path.join(folder_path, filename), 'r') as f:                               # Read the file content
            text = f.read()                                                                     # read entire file as a single string
            clean_id = clean_doc_id(filename)                                                   # extracting the file number
            raw_docs[clean_id] = tokenize(text)                                                 # store raw tokens for TF-IDF
            clean_docs[clean_id] = preprocess(text, stopwords)                                  # store clean tokens for inverted index

    return raw_docs, clean_docs                                                                 # return both doc representations

def build_vocabulary(docs):
    vocab = set()                                                                               # make set of unique words in vacabulary
    for words in docs.values():
        vocab.update(words)                                                                     # add all tokens from each document
    return sorted(list(vocab))                                                                  # return sorted list for consistent vector ordering

def compute_tf(docs):
    return {doc: Counter(words) for doc, words in docs.items()}                                 # count each term per document

def compute_df(docs, vocab):
    return {
        term: sum(1 for d in docs if term in docs[d])                                           # count docs containing each term
        for term in vocab
    }

def compute_idf(df, N):
    return {
        term: math.log(N / df[term]) if df[term] else 0                                         # idf=log(N/df)
        for term in df
    }

def build_vectors(tf, idf, vocab):
    vectors = {}
    for doc, counts in tf.items():                                                             # build TF-IDF vector for each document: [tf(term) * idf(term) for term in vocab]
        vectors[doc] = [counts.get(term, 0) * idf[term] for term in vocab]
    return vectors                                                                             # returns {doc_id → [float, float, ...]}

def build_inverted_index(clean_docs):
    index = defaultdict(list)                                                                  
    for doc, words in clean_docs.items():
        for word in set(words):                                                                # use set() to avoid duplicate doc entries per term
            index[word].append(doc)                                                            # map each term to the document it appears in
    return dict(index)                                                                         # convert defaultdict to regular dict before returning

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))                                                    # dot product of the two vectors
    mag1 = math.sqrt(sum(a * a for a in v1))                                                    # magnitude of vector 1
    mag2 = math.sqrt(sum(b * b for b in v2))                                                    # magnitude of vector 2
    return dot / (mag1 * mag2) if mag1 and mag2 else 0                                          # cosine = dot / (|v1|.|v2|)

def query_vector(query, vocab, idf):
    words = tokenize(query)                                                                     # tokenize query (lowercase + punctuation removal only)
    counts = Counter(words)                                                                     # count term frequency in the query
    return [counts.get(term, 0) * idf.get(term, 0) for term in vocab]                           # build query vector aligned to vocab

def search(query, vectors, vocab, idf, alpha=0.005):
    q_vec = query_vector(query, vocab, idf)                                                     # build the TF-IDF query vector

    results = []
    for doc, vec in vectors.items():
        score = cosine_similarity(vec, q_vec)                                                   # compute cosine similarity between query vector and document vector
        if score >= alpha:                                                                      # apply minimum relevance threshold to filter out very low scores
            results.append((doc, score))                                                       

    results.sort(key=lambda x: x[1], reverse=True)                                              # sort by score, highest first
    return results                                                                              # return ranked list of (doc_id, score)

def save_indexes(vocab, inverted_index, vectors):
    os.makedirs("indexes", exist_ok=True)                                                       # create 'indexes/' folder if it doesn't exist

    with open("indexes/vocab.txt", "w") as f:                                                   # write vocab to a text file, one term per line
        f.write("\n".join(sorted(vocab)))                                                       # sort vocab 

    with open("indexes/inverted_index.txt", "w") as f:                                          # write inverted index to a text file in format: term: doc1, doc2, ...
        for term in sorted(inverted_index.keys()):
            f.write(f"{term}: {', '.join(sorted(inverted_index[term]))}\n")                     # sort doc IDs

    print("Indexes saved in 'indexes/' folder\n") 

def run_gui(vectors, vocab, idf):                                                              # create a simple Tkinter GUI for searching the document collection
    root = tk.Tk()
    root.title("Document Search Engine")
    root.geometry("500x600")

    def perform_search():                                                                     # function to execute when search button is clicked or 'Enter' is pressed in the query entry field
        query = entry.get()
        if not query:
            return
        
        results = search(query, vectors, vocab, idf)                                          # perform search and get ranked results
        
        result_area.delete(1.0, tk.END)                                                       # clear previous results from the text area
        result_area.insert(tk.END, f"Search Results for: '{query}'\n")                        # insert a header for the results section
        result_area.insert(tk.END, "="*40 + "\n\n")                                           # insert a separator line for better readability
        
        if not results:
            result_area.insert(tk.END, "No documents found.")
        else:
            result_area.insert(tk.END, f"Found {len(results)} document(s):\n\n")
            for doc_id, score in results:
                result_area.insert(tk.END, f"ID: {doc_id.ljust(10)} | Score: {score:.4f}\n")  # insert each result with document ID and its cosine similarity score upto 4 decimal places, formatted for readability

    tk.Label(root, text="Enter Search Query:", font=("Arial", 10, "bold")).pack(pady=10)      # label prompting user to enter search query
    entry = tk.Entry(root, width=50)
    entry.pack(pady=5)
    entry.bind('<Return>', lambda event: perform_search())                                    # Allow pressing 'Enter' to search
    
    search_btn = tk.Button(root, text="Search", command=perform_search, bg="#dddddd")       # search button to trigger the search function when clicked, with a light gray background for better visibility
    search_btn.pack(pady=10)
    
    result_area = scrolledtext.ScrolledText(root, width=55, height=25)
    result_area.pack(pady=10)

    root.mainloop()

print("Loading system...")                                                                    
stopwords = ["a", "is", "the", "of", "all", "and", "to", "can", "be", "as", "once","for", "at", "am", "are", "has", "have", "had", "up", "his", "her", "in", "on", "no", "we", "do"]
raw_docs, clean_docs = load_documents("Trump Speeches", stopwords)                           # load documents and preprocess them, getting both raw token lists for TF-IDF and cleaned token lists for inverted index
vocab = build_vocabulary(raw_docs)                                                           # build the vocabulary from the raw token lists (includes stopwords, punctuation-removed tokens, but not lemmatized and stop word removal)
tf = compute_tf(raw_docs)
df = compute_df(raw_docs, vocab)
idf = compute_idf(df, len(raw_docs))
vectors = build_vectors(tf, idf, vocab)
inverted_index = build_inverted_index(clean_docs)
    
save_indexes(vocab, inverted_index, vectors)
    
print("System Ready. Launching GUI...")
run_gui(vectors, vocab, idf)                                                               # Launch the GUI for interactive searching of the document collection