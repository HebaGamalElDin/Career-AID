# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 09:39:46 2020

@author: Heba Gamal EL-Din
"""
import heapq
import nltk

class Text_Summarization:
    def __init__(self, Text):
        self.Text = str(Text)
        self.Sentences = nltk.sent_tokenize(Text)

    def word_Frequency_(self):
        stopwords = nltk.corpus.stopwords.words('english')
        word_frequencies = {}
        for word in nltk.word_tokenize(self.Text):
            if word not in stopwords:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1
        return word_frequencies
    
    def word_weight_(self,word_Freq):
        maximum_frequncy = max(word_Freq.values())
        word_weights={}
        for word in word_Freq.keys():
            word_weights[word] = (word_Freq[word]/maximum_frequncy)
        return word_weights
    
    def sent_score(self,words_weights):
        sentence_scores = {}
        for sent in self.Sentences:
            for word in nltk.word_tokenize(sent.lower()):
                if word in words_weights.keys():
                    if len(sent.split(' ')) < 30: #Ignore too Large Sentences that more than three words
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent] = words_weights[word]
                        else:
                            sentence_scores[sent] += words_weights[word]
        return sentence_scores
    
    def max_(self,sentences_):
        summary_sentences = heapq.nlargest(10, sentences_, key=sentences_.get)
        summary = ' '.join(summary_sentences).replace(',', '\n')
        return summary
    
    def summarize_(self):
        out = self.word_Frequency_()
        out_1 = self.word_weight_(out)
        out2 = self.sent_score(out_1)
        out3 = self.max_(out2)
        file = open('Summary_Text.txt','w')
        #file.write(out3)
        return out3
