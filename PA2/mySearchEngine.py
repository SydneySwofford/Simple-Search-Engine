"""
Simple indexer and search engine built on an inverted-index and the BM25 ranking algorithm.
"""
import os
from collections import defaultdict, Counter
import pickle
import math
import operator
import json
import re
from typing import Any
from tqdm import tqdm
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
from datasets import load_dataset


class  Indexer:

    dbfile = "./ir.idx"  # This is the index file you will create and manager for indexing

    def __init__(self):
        mystopwords=set(stopwords.words('english'))
        json_file="C:\\Users\\Sydney\\WebCrawling\\PA2\\cnn.jsonl"
        documentLengthList=[]
        lemmatizer=WordNetLemmatizer()
        nested_dict={'token_my':{'id1': 'value1'}}
        raw_data=[json.loads(line) for line in(open(json_file, encoding="utf-8"))]
       
        for i in range(len(raw_data)):
            filtered_list=[]
            start_list=raw_data[i]['article']
            #normalize to remove newline characters
            #start_list=re.sub('-\n', '', str(start_list))
            #normalize to remove numbers(not useful for this purpose)
            no_numbers=re.sub(r'\d+','',str(start_list))
            #normalize to remove all punctuation symbols
            no_punctuation=re.sub(r'[^\w\s]','',no_numbers)
            #normalize all to one case 
            lower_case=no_punctuation.lower()
            #find all words in the page response
            split_soup=re.findall(r'\S+',lower_case)
            #normalize for stopwords
            for word in split_soup:
                #if word not in stopwordsmine:
                    filtered_list.append(word)
            raw_data[i]['article']=filtered_list
    
        #print(raw_data)
        
        
        documentLengthList=dict()
       
        for i in range(len(raw_data)):
            avdl=0
            for word in raw_data[i]['article']:
                avdl+=1
                if word not in mystopwords:
                    wordStem=str(lemmatizer.lemmatize(word))
                    if wordStem in nested_dict and raw_data[i]['pageid'] in nested_dict[wordStem]:
                        nested_dict[wordStem][raw_data[i]['pageid']]+=1
                    elif wordStem in nested_dict:
                        nested_dict[wordStem][raw_data[i]['pageid']]=1
                    else:
                        nested_dict[wordStem]={raw_data[i]['pageid']:1}
                        #nested_dict[wordStem][raw_data[i]['pageid']]+=1
            documentLengthList[raw_data[i]['pageid']]=avdl

        #print(nested_dict)
        with open("./ir.idx", 'wb') as pickle_file:
            pickle.dump(nested_dict, pickle_file)

        self.output=nested_dict
        self.doclengths=documentLengthList

    def query(self,q_str):
        nested_dic=self.output
        document_lengths=self.doclengths
        #clean the query 
        new_list=[]
        token_string=q_str.split()
        #normalize to remove numbers(not useful for this purpose)
        no_numbers=re.sub(r'\d+','',str(token_string))
        #normalize to remove all punctuation symbols
        no_punctuation=re.sub(r'[^\w\s]','',no_numbers)
        #normalize all to one case 
        lower_case=no_punctuation.lower()
        #find all words in the page response
        split_soup=re.findall(r'\S+',lower_case)
        #normalize for stopwords
        for word in split_soup:
         #if word not in stopwordsmine:
            new_list.append(word)

        # Calculate IDF for each query term

        #calculate # of documents that arent empty
        count=0
        for i in self.doclengths:
            if self.doclengths[i] != 0:
                count+=1
        def count_inner_keys(outer_key, my_dict):
            if outer_key in my_dict:
                inner_dict = my_dict[outer_key]
                return len(inner_dict)
            else:
                return 0


        def access_inner_values(outer_key, my_dict):
            if outer_key in my_dict:
                inner_dict = my_dict[outer_key]
            for inner_key, value in inner_dict.items():
                return value
            else:
                return 0
            
        def access_inner_keys(outer_key, my_dict):
            docids=[]
            if outer_key in my_dict:
                inner_dict = my_dict[outer_key]
                for inner_key in inner_dict.keys():
                    docids.append(inner_key)         
            else:
                return 0
            return docids
        
        
        tfdict=dict()
        totalscoredict=dict()
        top5dict=dict()
        for i in new_list:
           
            idf= (((count-count_inner_keys(i,nested_dic)) + 0.5)/(count_inner_keys(i,nested_dic)+ 0.5)) +1

            if i in nested_dic:
                docidlist=access_inner_keys(i,nested_dic)

            for x in docidlist:
                tf = nested_dic[i][x]/document_lengths[x]
                docsum=0
                for value in document_lengths.values():
                    docsum+=value

                tf_score = ((1.5 + 1.0) * tf) /(1.5 * ((1.0 - 0.75) + 0.75 * (int(document_lengths[x]) / docsum/len(document_lengths))) + tf)
                if x in tfdict:
                    tfdict[x]+=tf_score
                else:
                    tfdict[x]=tf_score

        # calculat the total_score
        
            for key, value in tfdict.items():
                total_score=value*idf
                if key in totalscoredict:
                    totalscoredict[key]+=total_score
                else:
                    totalscoredict[key]=total_score

        #sort for the most relevant documents 
        top5dict=dict(Counter(totalscoredict).most_common(5))

        #format ouput
        this_data=[json.loads(line) for line in(open("C:\\Users\\Sydney\\WebCrawling\\PA2\\cnn.jsonl", encoding="utf-8"))]
        for key, value in top5dict.items():
            for x in range(len(this_data)):
                if this_data[x]['pageid']== key:
                    holder=this_data[x]['article']
                    print()
                    print("DocId: ", key)
                    print("Score: ", value)
                    print("Article: ", holder)

           

               
           



        #top5dict=sort_relevant_documents(totalscoredict)
        #print(top5dict)


            
           



        
            

        


if __name__ == "__main__":
    a=Indexer()
    while True:
        query_string= input("Enter your query (or 'exit' to quit): ")
        if query_string =='exit':
            break
        a.query(query_string)