"""
Created on Fri Apr 12 15:10:40 2019

@author: nihitsaxena
"""

import nltk
from pywsd.lesk import simple_lesk
import numpy as np
from nltk.corpus import wordnet as wn

class SentenceSimilarity:
    
    def __init__(self):
        self.word_order = False
        
    
    def identifyWordsForComparison(self, sentence):
        #Taking out Noun and Verb for comparison word based
        tokens = nltk.word_tokenize(sentence)        
        pos = nltk.pos_tag(tokens)
        pos = [p for p in pos if p[1].startswith('N') or p[1].startswith('V')]     
        return pos
    
    def wordSenseDisambiguation(self, sentence):
        # removing the disambiguity by getting the context
        pos = self.identifyWordsForComparison(sentence)
        sense = []
        for p in pos:
            sense.append(simple_lesk(sentence, p[0], pos=p[1][0].lower()))
        return set(sense)
    
    def getSimilarity(self, arr1, arr2, vector_len):
        #cross multilping all domains 
        vector = [0.0] * vector_len
        count = 0
        for i,a1 in enumerate(arr1):
            all_similarityIndex=[]
            for a2 in arr2:
                similarity = wn.synset(a1.name()).wup_similarity(wn.synset(a2.name()))
                if similarity != None:
                    all_similarityIndex.append(similarity)
                else:
                    all_similarityIndex.append(0.0)
            all_similarityIndex = sorted(all_similarityIndex, reverse = True)
            vector[i]=all_similarityIndex[0]
            if vector[i] >= 0.804:
                count +=1
        return vector, count        
        
        
    def shortestPathDistance(self, sense1, sense2):
        #getting the shortest path to get the similarity
        if len(sense1) >= len(sense2):
            grt_Sense = len(sense1)
            v1, c1 = self.getSimilarity(sense1, sense2, grt_Sense)
            v2, c2 = self.getSimilarity(sense2, sense1, grt_Sense)
        if len(sense2) > len(sense1):
            grt_Sense = len(sense2)
            v1, c1 = self.getSimilarity(sense2, sense1, grt_Sense)
            v2, c2 = self.getSimilarity(sense1, sense2, grt_Sense)
        return np.array(v1),np.array(v2),c1,c2
        
    def main(self, sentence1, sentence2):
        sense1 = self.wordSenseDisambiguation(sentence1)
        sense2 = self.wordSenseDisambiguation(sentence2)        
        v1,v2,c1,c2 = self.shortestPathDistance(sense1,sense2)
        dot = np.dot(v1,v2)
        print("dot", dot) # getting the dot product
        tow = (c1+c2)/1.8
        final_similarity = dot/tow
        print("similarity",final_similarity)

obj = SentenceSimilarity()
a = 'A jewel is a precious stone used to decorate valuable things that you wear, such as rings or necklaces.'
b = 'A gem is a jewel or stone that is used in jewellery.'
obj.main(a,b)