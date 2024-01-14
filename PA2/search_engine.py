import os
from collections import defaultdict, Counter
import pickle
import math
import operator
import json

from tqdm import tqdm
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from datasets import load_dataset
import nltk
nltk.download('wordnet')

class Indexer:
    dbfile = "./ir.idx"  # This is the index file you will create and manage for indexing
    dbfilemytest="./mir.idx"

    def __init__(self):
        self.tok2idx = {}                       # map (token to id)
        self.idx2tok = {}                       # map (id to token)
        self.postings_lists = {}                # postings for each word
        self.docs = []                          # encoded document list
        self.raw_ds = None                      # raw documents for search results
        self.corpus_stats = {'avgdl': 0}        # any corpus-level statistics
        self.stopwords = stopwords.words('english')
        
        if os.path.exists(self.dbfile):
            # Load data from the existing index file if it exists
            with open(self.dbfile, 'rb') as f:
                self.tok2idx, self.idx2tok, self.postings_lists, self.docs, self.corpus_stats = pickle.load(f)
        else:
            # Load CNN/DailyMail dataset, preprocess and create postings lists.
            #ds = load_dataset("C:/Users/Sydney/WebCrawling/PA2/cnn.jsonl",data_files="C:/Users/Sydney/WebCrawling/PA2/cnn.jsonl")
            #data_list=[json.loads(line) for line in (open(json_file, encoding="utf8"))]
            json_file="C:/Users/Sydney/WebCrawling/PA2/cnn.jsonl"
            raw_list =[json.loads(line) for line in(open(json_file, encoding="utf8"))]
            from datasets import Dataset
            pageId=[]
            url=[]
            title=[]
            article=[]
            for i in range(len(raw_list)):
                pageId.append(raw_list[i]['pageid'])
                url.append(raw_list[i]['url'])
                title.append(raw_list[i]['title'])
                article.append(raw_list[i]['article'])
            
            
            ds_list=({'pageid': pageId, 'url': url, 'title': title, 'article': article})
            myds=Dataset.from_dict(ds_list)
            with open('newdataset.pickle', 'wb') as output:
                pickle.dump(myds, output)

            data_files = {
            "train": "C:/Users/Sydney/WebCrawling/PA2/train.tsv",
            "validation": "C:/Users/Sydney/WebCrawling/PA2/dev.tsv",
            "test": "C:/Users/Sydney/WebCrawling/PA2/test.tsv",
}
            #ds=load_dataset("C:\\Users\\Sydney\\WebCrawling\\PA2",data_files=data_files,sep="\t")
            self.raw_ds = myds['article']
            self.clean_text(self.raw_ds)
            self.create_postings_lists()

    def clean_text(self, lst_text, query=False):
        tokenizer = RegexpTokenizer(r'\w+')
        lemmatizer = WordNetLemmatizer()

        cleaned_text = []

        for doc in lst_text:
        # Ensure the document is a string before applying lower()
            if isinstance(doc, str):
                # Tokenize, lemmatize, and preprocess the text
                tokens = tokenizer.tokenize(doc.lower())
                tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in self.stopwords]
                cleaned_text.append(tokens)

        # Use cleaned_text for further processing
        if not query:
            self.docs = cleaned_text
        else:
            return cleaned_text

    def create_postings_lists(self):
        # Initialize corpus-level statistics
        total_tokens = 0

        # Iterate through documents and create postings lists
        for doc_id, tokens in enumerate(self.docs):
            # Update the total number of tokens
            total_tokens += len(tokens)

            # Create postings list for each term
            term_freq = Counter(tokens)
            for term, freq in term_freq.items():
                if term not in self.tok2idx:
                    token_id = len(self.tok2idx)
                    self.tok2idx[term] = token_id
                    self.idx2tok[token_id] = term
                    self.postings_lists[token_id] = []
                else:
                    token_id = self.tok2idx[term]

                self.postings_lists[token_id].append((doc_id, freq))

        # Calculate average document length (avgdl)
        mylength=len(self.docs)
        if mylength!=0:
            self.corpus_stats['avgdl'] = total_tokens / len(self.docs)

        # Save the index to a file using pickle
        with open(self.dbfile, 'wb') as f:
            pickle.dump([self.tok2idx, self.idx2tok, self.postings_lists, self.docs, self.corpus_stats], f)

class SearchAgent:
    k1 = 1.5                # BM25 parameter k1 for tf saturation
    b = 0.75                # BM25 parameter b for document length normalization

    def __init__(self, indexer):
        self.indexer = indexer

    def query(self, q_str):
        # Clean the query text
        query_tokens = self.indexer.clean_text([q_str], query=True)[0]

        results = {}
        # Calculate BM25 scores for each document
        for token in query_tokens:
            if token in self.indexer.tok2idx:
                token_id = self.indexer.tok2idx[token]
                idf = math.log((len(self.indexer.docs) - len(self.indexer.postings_lists[token_id]) + 0.5) / (len(self.indexer.postings_lists[token_id]) + 0.5) + 1.0)
                for doc_id, tf in self.indexer.postings_lists[token_id]:
                    dl = len(self.indexer.docs[doc_id])
                    tf_score = ((self.k1 + 1.0) * tf) / (self.k1 * ((1.0 - self.b) + self.b * (dl / self.indexer.corpus_stats['avgdl'])) + tf)
                    if doc_id in results:
                        results[doc_id] += idf * tf_score
                    else:
                        results[doc_id] = idf * tf_score

        # Sort the results by scores in descending order
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

        # Display the top 5 results
        self.display_results(sorted_results[:5])

    def display_results(self, results):
        if not results:
            print("No results found.")
            return

        # Decode and display the results
        for doc_id, score in results:
            print(f'\nDocID: {doc_id}')
            print(f'Score: {score}')
            print('Article:')
            print(self.indexer.raw_ds[doc_id])

if __name__ == "__main__":
    i = Indexer()           # Instantiate an indexer
    q = SearchAgent(i)      # Create a document retriever
    while True:
        query_str = input("Enter your query (or 'exit' to quit): ")
        if query_str == 'exit':
            break
        q.query(query_str)