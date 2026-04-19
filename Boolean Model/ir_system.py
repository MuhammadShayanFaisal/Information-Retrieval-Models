import os
import re
from collections import defaultdict
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
def preprocess(text, stopwords):                                                                                                     # prerocess function for my documents and user query
    text = text.lower()                                                                                                              # case fold
    text = re.sub(r'[^a-z0-9\s]', ' ', text)                                                                                         # remove punctuation
    words = text.split()                                                                                                             # tokenize
    result = []                         
    for word in words:
        if word not in stopwords:                                                                                                    # checking if word is not a stopword then go for other preprocessing steps
            result.append(stemmer.stem(word))                                                                                        # Porter stemming
    return result

def build_indexes(folder_path, stopwords):                                                                                           # build indexes
     
    inverted_index = defaultdict(set)                                                                                               # set is used to avoid duplicate document IDs for the same term
    positional_index = defaultdict(lambda: defaultdict(list))                                                                       # nested defaultdict to store term, doc_id,  positions
    
    files = sorted(os.listdir(folder_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))                                      # sort files based on numeric ID in filename
    for doc_id, filename in enumerate(files):                                                                                       # enumerate is used for counter and content in file 
        with open(os.path.join(folder_path, filename), 'r') as f:                                                                   # Open file
            text = f.read()                                                                                                         # Read file content

        words = preprocess(text, stopwords)                                                                                        # Preprocess the text
        for pos, word in enumerate(words):                                                                                         # here enumerate is used for position and word in each file words list
          docs = inverted_index[word]                                                                                              
          docs.add(str(doc_id))                                                                                                    # Add document ID to inverted index
          doc_positions = positional_index[word][str(doc_id)]                                                                      
          doc_positions.append(pos)                                                                                                # Add position to positional index
    return inverted_index, positional_index

def save_indexes(inv_index, pos_index):                                                                                            # Save indexes to text files
    
    with open("inverted_index.txt", "w") as f:                                                                                     # open file for writing inverted index
        for term in inv_index.keys():                                                                                              # writing each term in inverted index with its document IDs
            docs = sorted(inv_index[term])
            f.write(f"{term} : {', '.join(docs)}\n")
    print("Inverted Indexes saved")
    
    with open("positional_index.txt", "w") as f:                                                                                  # open file for writing positional index
        for term in pos_index.keys():                                                                                             # writing each term in positional index with its document IDs and positions                                                                                         
            line = f"{term} : "
            parts = []

            for doc in sorted(pos_index[term].keys()):
                positions = ",".join(map(str, pos_index[term][doc]))
                parts.append(f"{doc}:[{positions}]")
            line += " ; ".join(parts)
            f.write(line + "\n")
    print("Positional Indexes saved")

def load_indexes():                                                                                                              # Load indexes from text files
    inv_index = {}
    pos_index = {}
    
    with open("inverted_index.txt", "r") as f:                                                                                   # open file for reading inverted index
        for line in f:                                                                                                           # for each line in file split term and document IDs and store in dictionary
            term, docs = line.strip().split(" : ")
            inv_index[term] = set(docs.split(", "))                                                                              # store document IDs
    
    with open("positional_index.txt", "r") as f:                                                                                # open file for reading positional index
        for line in f:                                                                                                          # for each line in file split term and document IDs with positions and store in nested dictionary
            term, rest = line.strip().split(" : ")
            pos_index[term] = {}
            
            docs_parts = rest.split(" ; ")
            
            for part in docs_parts:                                                                                             # split document ID and positions
                doc, positions = part.split(":")
                positions = positions.strip("[]")
                
                if positions:
                    pos_index[term][doc] = list(map(int, positions.split(",")))
                else:
                    pos_index[term][doc] = []
    
    return inv_index, pos_index

def simple_query(query, index, stopwords):                                                                                   # simple query processing for single term queries
    word = query.strip().split()[0] if query.strip() else ""                                                                 # ersae extra spaces and get the first word as term
    if not word:
        return set()
    processed = preprocess(word, stopwords)
    if processed:
        return index.get(processed[0], set())                                                                               # return the set of document IDs for the processed term from the index
    return set()

def process_boolean_query(query, index, all_docs,stopwords):                                                                # process boolean queries with AND, OR, NOT and parentheses
    
    tokens = re.findall(r'\(|\)|AND|OR|NOT|\w+', query.upper())                                                             # tokenize the query into operators and terms, convert to uppercase for uniformity
    
    def get_postings(term):                                                                                                 # helper function to get postings for a term, preprocess the term and return the set of document IDs from the index
        processed = preprocess(term.lower(), stopwords)
        if not processed:
            return set()
        return index.get(processed[0], set())
    
    expression = []
    
    for token in tokens:                                                                                                   # convert boolean operators to Python set operations and terms to get_postings calls
        
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
        return eval(expr)                                                                                                  # evaluate the expression to get the final set of document IDs matching the boolean query
    except:
        print("Error in query!")
        return set()

def proximity_query(term1, term2, k, pos_index):                                                                            # process proximity queries for two terms with a specified distance k using the positional index
    result = set()
    postings1 = pos_index.get(term1, {})                                                                                    
    postings2 = pos_index.get(term2, {})
    common_docs = set(postings1.keys()) & set(postings2.keys())                                                             # find common documents that contain both terms
    
    for doc in common_docs:                                                                                                 # for each common document, check the positions of the two terms and see if they are within k words of each other
        pos1 = postings1[doc]
        pos2 = postings2[doc]
        
        i, j = 0, 0
        
        while i < len(pos1) and j < len(pos2):                                                                               # use two variables to traverse the position lists of both terms
            
            if abs(pos2[j] - pos1[i])== k+1:                                                                                 # if the difference in positions is exactly k+1 (k words in between), then add the document to the result set
                result.add(doc)
                break
            
            elif pos2[j] <= pos1[i]:
                j += 1
            else:
                i += 1
    
    return result

def phrase_query(query, pos_index, stopwords):                                                                              # process phrase queries by checking if the terms appear in the exact order and adjacent to each other in the documents using the positional index
    
    terms = query.split()                                                                                                   # split the query into terms
    terms = [preprocess(t, stopwords)[0] for t in terms]                                                                    # preprocess each term in the phrase query
    
    if len(terms) < 2:
        return set()
    
    result_docs = set(pos_index.get(terms[0], {}).keys())
    
    for i in range(1, len(terms)):
        next_term = terms[i]
        temp_docs = set()
        
        postings1 = pos_index.get(terms[i-1], {})
        postings2 = pos_index.get(next_term, {})
        
        common_docs = result_docs & set(postings2.keys())                                                                    # find common documents that contain the next term in the phrase query
        
        for doc in common_docs:                                                                                              # for each common document, check the positions of the current term and the next term to see if they are adjacent (difference of 1 in positions)
            pos1 = postings1[doc]
            pos2 = postings2[doc]
            
            for p1 in pos1:                                                                                                 # check if there is a position in pos2 that is exactly 1 greater than p1, which would indicate that the terms appear in the correct order and are adjacent
                if (p1 + 1) in pos2:
                    temp_docs.add(doc)
                    break
        
        result_docs = temp_docs
    
    return result_docs

def detect_query_type(query):                                                                                             # detect the type of query based on the user input
    q = query.upper()                                                                                                     
    words = query.strip().split()                                                                                        # removing extra spaces and split the query into words
    if "/" in query:                                                                                                     # declare proximity query if "/" is present in the query
        return "PROXIMITY"
    elif "AND" in q or "OR" in q or "NOT" in q or "(" in q:                                                              # declare boolean query if any boolean operator or parentheses is present in the query
        return "BOOLEAN"
    elif len(words) > 1:                                                                                                 # declare phrase query if there are multiple words in the query (after stripping extra spaces)
        return "PHRASE"
    else:                                                                                                                # for one word queries, declare simple query
        return "SIMPLE"

def handle_query(query, inv_index, pos_index, all_docs, stopwords):                                                      # deal with the user query
    qtype = detect_query_type(query)                                                                                     # checks the type of query
    print("Query Type:", qtype)
    if qtype == "PROXIMITY":                                                                                             # process proximity query, extract terms and distance k from the query, preprocess the terms, and call the proximity_query function to get the matching documents
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
    elif qtype == "BOOLEAN":                                                                                            # process boolean query by calling the process_boolean_query function
        return process_boolean_query(query, inv_index,all_docs, stopwords)
    elif qtype == "PHRASE":                                                                                            # process phrase query by calling the phrase_query function
        return phrase_query(query, pos_index, stopwords)
    else:                                                                                                               # for simple queries, call the simple_query function to get the matching documents
        return simple_query(query, inv_index, stopwords)

def main():
    folder_path = input("Enter the path to the folder which contain Trump Speeches (56 files): ")                                            # Folder's path which contain 56 files
    stopwords = ["a", "is", "the", "of", "all", "and", "to", "can", "be", "as",                                                              # as provided in StopwordList.txt
                "once", "for", "at", "am", "are", "has", "have", "had", "up", 
                "his", "her", "in", "on", "no", "we", "do" ]
    print("Building indexes...")
    inv_index, pos_index = build_indexes(folder_path, stopwords)                                                                            # Build Inverted and Positional Indexes
    save_indexes(inv_index, pos_index)                                                                                                      # Save Inverted and Positional Indexes as text files
    
    all_docs = set()                                                                                                                        # All documents set
    for v in inv_index.values():
        all_docs.update(v)
    print("----------------------System Ready for Queries-------------------------------")
    
    while True:
        query = input("\nEnter Query (or exit): ")                                                                                           # User Query Input
        if query.lower() == "exit":
            break
        result = handle_query(query, inv_index, pos_index,all_docs, stopwords)                                                               # My Model handle the query and return the result
        print("Matched Documents :", len(result))                                                                                            # This line show the number of matched documents
        print("Matched Document IDs :", result)                                                                                              # This line show the matched document IDs.
        print("-------------------------------------------------------------------------")

main()                                                                                                                                       # My Code starts from this main function                   