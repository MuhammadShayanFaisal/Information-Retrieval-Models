import os
import re

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

def build_indexes(folder_path, stopwords):
    
    inverted_index = defaultdict(set)
    positional_index = defaultdict(lambda: defaultdict(list))
    
    files = sorted(os.listdir(folder_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    for doc_id, filename in enumerate(files):
        with open(os.path.join(folder_path, filename), 'r') as f:
            text = f.read()

        words = preprocess(text, stopwords)
        for pos, word in enumerate(words):
          docs = inverted_index[word]
          docs.add(str(doc_id))
          doc_positions = positional_index[word][str(doc_id)]
          doc_positions.append(pos)
    return inverted_index, positional_index

def save_indexes(inv_index, pos_index):
    
    with open("inverted_index.txt", "w") as f:
        for term in inv_index.keys():
            docs = sorted(inv_index[term])
            f.write(f"{term} : {', '.join(docs)}\n")
    print("Inverted Indexes saved")
    
    with open("positional_index.txt", "w") as f:
        for term in pos_index.keys():
            line = f"{term} : "
            parts = []

            for doc in sorted(pos_index[term].keys()):
                positions = ",".join(map(str, pos_index[term][doc]))
                parts.append(f"{doc}:[{positions}]")
            line += " ; ".join(parts)
            f.write(line + "\n")
    print("Positional Indexes saved")

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
# SIMPLE QUERY (OR behavior)
# =========================

def simple_query(query, index, stopwords):
    
    words = query.split()
    result = set()
    
    for word in words:
        processed = preprocess(word, stopwords)
        if processed:
            result |= index.get(processed[0], set())
    
    return result


# =========================
# BOOLEAN QUERY (ADVANCED)
# =========================

def process_boolean_query(query, index, all_docs,stopwords):
    
    query = query.replace("_set", "")
    
    tokens = re.findall(r'\(|\)|AND|OR|NOT|\w+', query.upper())
    
    def get_postings(term):
        processed = preprocess(term.lower(), stopwords)
        if not processed:
            return set()
        return index.get(processed[0], set())
    
    expression = []
    
    for token in tokens:
        
        if token == "AND":
            expression.append("&")
        elif token == "OR":
            expression.append("|")
        elif token == "NOT":
            expression.append("all_docs -")
        elif token in ["(", ")"]:
            expression.append(token)
        else:
            expression.append(f"get_postings('{token}')")
    
    expr = " ".join(expression)
    
    try:
        return eval(expr)
    except:
        print("Error in query!")
        return set()

def proximity_query(term1, term2, k, pos_index):
    result = set()
    postings1 = pos_index.get(term1, {})
    postings2 = pos_index.get(term2, {})
    common_docs = set(postings1.keys()) & set(postings2.keys())
    
    for doc in common_docs:
        pos1 = postings1[doc]
        pos2 = postings2[doc]
        
        i, j = 0, 0
        
        while i < len(pos1) and j < len(pos2):
            
            # correct ordered distance
            if (pos2[j] - pos1[i])== k+1:
                result.add(doc)
                break
            
            elif pos2[j] <= pos1[i]:
                j += 1
            else:
                i += 1
    
    return result

def phrase_query(query, pos_index, stopwords):
    
    terms = query.split()
    terms = [preprocess(t, stopwords)[0] for t in terms]
    
    if len(terms) < 2:
        return set()
    
    # start with first term postings
    result_docs = set(pos_index.get(terms[0], {}).keys())
    
    for i in range(1, len(terms)):
        next_term = terms[i]
        temp_docs = set()
        
        postings1 = pos_index.get(terms[i-1], {})
        postings2 = pos_index.get(next_term, {})
        
        common_docs = result_docs & set(postings2.keys())
        
        for doc in common_docs:
            pos1 = postings1[doc]
            pos2 = postings2[doc]
            
            # check adjacency (k = 0 → difference = 1)
            for p1 in pos1:
                if (p1 + 1) in pos2:
                    temp_docs.add(doc)
                    break
        
        result_docs = temp_docs
    
    return result_docs

def detect_query_type(query):
    q = query.upper()
    words = query.strip().split()
    if "/" in query:
        return "PROXIMITY"
    elif "AND" in q or "OR" in q or "NOT" in q or "(" in q:
        return "BOOLEAN"
    elif len(words) > 1:
        return "PHRASE"
    else:
        return "SIMPLE"

def handle_query(query, inv_index, pos_index, all_docs, stopwords):
    qtype = detect_query_type(query)
    print("Query Type:", qtype)
    if qtype == "PROXIMITY":
        try:
            parts = query.split("/")
            k = int(parts[1].strip())
            terms = parts[0].split()
            term1 = preprocess(terms[0], stopwords)[0]
            term2 = preprocess(terms[1], stopwords)[0]
            return proximity_query(term1, term2, k, pos_index)  
        except:
            print("Invalid Proximity Query Format!")
            return set() 
    elif qtype == "BOOLEAN":
        return process_boolean_query(query, inv_index,all_docs, stopwords)
    elif qtype == "PHRASE":
        return phrase_query(query, pos_index, stopwords)
    else:
        return simple_query(query, inv_index, stopwords)

def main():
    folder_path = input("Enter the path to the folder which contain Trump Speeches (56 files): ")
    stopwords = ["a", "is", "the", "of", "all", "and", "to", "can", "be", "as", 
                "once", "for", "at", "am", "are", "has", "have", "had", "up", 
                "his", "her", "in", "on", "no", "we", "do" ]
    print("Building indexes...")
    inv_index, pos_index = build_indexes(folder_path, stopwords)
    save_indexes(inv_index, pos_index)
    
    all_docs = set()
    for v in inv_index.values():
        all_docs|=v
    print("System Ready for Queries")
    
    while True:
        query = input("\nEnter Query (or exit): ")
        if query.lower() == "exit":
            break
        result = handle_query(query, inv_index, pos_index,all_docs, stopwords)
        print("Result:", len(result),result)

main()