import os
import re
import json
import nltk
nltk.download('punkt')
from collections import defaultdict

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
def preprocess(text, stopwords):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    words = text.split()
    
    result = []
    for word in words:
        if word not in stopwords:
            result.append(stemmer.stem(word))
    
    return result

def load_stopwords(path):
    with open(path, 'r') as f:
        return set(f.read().split())

def build_indexes(folder_path, stopwords):
    
    inverted_index = defaultdict(set)
    positional_index = defaultdict(lambda: defaultdict(list))
    
    # IMPORTANT: sort files
    files = sorted(os.listdir(folder_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))
       
    for doc_id, filename in enumerate(files):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
            text = f.read()
        
        words = preprocess(text, stopwords)
        
        for pos, word in enumerate(words):
            inverted_index[word].add(str(doc_id))
            positional_index[word][str(doc_id)].append(pos)
    
    return inverted_index, positional_index


# =========================
# SAVE / LOAD INDEXES
# =========================

def save_indexes(inv_index, pos_index):
    
    # ---------- Inverted Index ----------
    with open("inverted_index.txt", "w") as f:
        for term in sorted(inv_index.keys()):
            docs = sorted(inv_index[term])
            f.write(f"{term} : {', '.join(docs)}\n")
    
    
    # ---------- Positional Index ----------
    with open("positional_index.txt", "w") as f:
        for term in sorted(pos_index.keys()):
            
            line = f"{term} : "
            parts = []
            
            for doc in sorted(pos_index[term].keys()):
                positions = ",".join(map(str, pos_index[term][doc]))
                parts.append(f"{doc}:[{positions}]")
            
            line += " ; ".join(parts)
            f.write(line + "\n")
    
    print("Indexes saved in readable format!")

def load_indexes():
    
    inv_index = {}
    pos_index = {}
    
    # ---------- Load Inverted Index ----------
    with open("inverted_index.txt", "r") as f:
        for line in f:
            term, docs = line.strip().split(" : ")
            inv_index[term] = set(docs.split(", "))
    
    
    # ---------- Load Positional Index ----------
    with open("positional_index.txt", "r") as f:
        for line in f:
            
            term, rest = line.strip().split(" : ")
            pos_index[term] = {}
            
            docs_parts = rest.split(" ; ")
            
            for part in docs_parts:
                doc, positions = part.split(":")
                positions = positions.strip("[]")
                
                if positions:
                    pos_index[term][doc] = list(map(int, positions.split(",")))
                else:
                    pos_index[term][doc] = []
    
    return inv_index, pos_index
# =========================
# BOOLEAN OPERATIONS
# =========================

def AND(post1, post2):
    return post1 & post2

def OR(post1, post2):
    return post1 | post2

def NOT(post, all_docs):
    return all_docs - post


# =========================
# BOOLEAN QUERY PROCESSOR
# =========================

def process_boolean_query(query, inv_index, total_docs, stopwords):
    
    tokens = query.split()
    tokens = [t.lower() for t in tokens]
    
    all_docs = set(str(i) for i in range(total_docs))
    
    # preprocess only terms (not operators)
    processed = []
    for t in tokens:
        if t not in ["and", "or", "not"]:
            term = preprocess(t, stopwords)
            processed.append(term[0] if term else "")
        else:
            processed.append(t)
    
    # Handle NOT first
    i = 0
    while i < len(processed):
        if processed[i] == "not":
            term = processed[i+1]
            postings = inv_index.get(term, set())
            processed[i:i+2] = [NOT(postings, all_docs)]
        else:
            i += 1
    
    # Handle AND
    i = 0
    while i < len(processed):
        if processed[i] == "and":
            left = processed[i-1]
            right = processed[i+1]
            
            if isinstance(left, str):
                left = inv_index.get(left, set())
            if isinstance(right, str):
                right = inv_index.get(right, set())
            
            processed[i-1:i+2] = [AND(left, right)]
            i = 0
        else:
            i += 1
    
    # Handle OR
    i = 0
    while i < len(processed):
        if processed[i] == "or":
            left = processed[i-1]
            right = processed[i+1]
            
            if isinstance(left, str):
                left = inv_index.get(left, set())
            if isinstance(right, str):
                right = inv_index.get(right, set())
            
            processed[i-1:i+2] = [OR(left, right)]
            i = 0
        else:
            i += 1
    
    result = processed[0]
    
    if isinstance(result, str):
        result = inv_index.get(result, set())
    
    return result


# =========================
# POSITIONAL INTERSECT
# =========================

def positional_intersect(p1, p2, k):
    answer = set()
    
    for doc in p1:
        if doc in p2:
            list1 = p1[doc]
            list2 = p2[doc]
            
            i = 0
            j = 0
            
            while i < len(list1):
                while j < len(list2):
                    
                    if abs(list1[i] - list2[j]) <= k:
                        answer.add(doc)
                        break
                    elif list2[j] > list1[i]:
                        break
                    else:
                        j += 1
                i += 1
    
    return answer


# =========================
# PHRASE / PROXIMITY QUERY
# =========================

def process_phrase_query(query, pos_index, stopwords):
    
    # Example: "develop solution /3"
    parts = query.split("/")
    
    terms_part = parts[0].strip()
    k = int(parts[1].strip())
    
    terms = terms_part.split()
    
    # preprocess terms
    t1 = preprocess(terms[0], stopwords)[0]
    t2 = preprocess(terms[1], stopwords)[0]
    
    p1 = pos_index.get(t1, {})
    p2 = pos_index.get(t2, {})
    
    return positional_intersect(p1, p2, k)
    # =========================
# MAIN SEARCH ENGINE FUNCTION
# =========================

def run_search_engine(folder_path, stopwords_path):
    
    # Step 1: Load stopwords
    stopwords = load_stopwords(stopwords_path)
    
    print("Building indexes...")
    inv_index, pos_index = build_indexes(folder_path, stopwords)
    
    # Step 2: Save indexes
    save_indexes(inv_index, pos_index)
    
    print("Loading indexes...")
    inv_index, pos_index = load_indexes()
    
    total_docs = len(set(doc for docs in inv_index.values() for doc in docs))
    
    print("\nSearch Engine Ready! 🚀")
    print("Type 'exit' to quit\n")
    
    # Step 3: Query loop
    while True:
        query = input("Enter Query: ").strip()
        
        if query.lower() == "exit":
            print("Exiting...")
            break
        
        # Detect query type
        if "/" in query:
            print("Detected: Phrase / Proximity Query")
            result = process_phrase_query(query, pos_index, stopwords)
        
        else:
            print("Detected: Boolean Query")
            result = process_boolean_query(query, inv_index, total_docs, stopwords)
        
        print("Result Docs:", len(result),sorted(result))
        print("-" * 40)
run_search_engine(r"C:\Users\DELL\Documents\IR Assignment 01\Trump Speeches", "Stopword-List.txt")