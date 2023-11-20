# _*_coding:utf-8_*_
import os, sys, getopt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

from gensim.models import KeyedVectors,word2vec,Word2Vec
import jieba
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing

def stopwordslist():
    fname = "./stopword.txt"
    stopwords = [ line.strip() for line in open(fname, mode='r',encoding='UTF-8').readlines()]
    return stopwords

def getkeysentc(trnfile,txtfile):
    ifile = open(trnfile, mode='r',encoding='UTF-8')
    sentences_list = ifile.readlines()
    ifile.close()

    stopwords = stopwordslist()
    sentences_cut = []
    i = 0
    for ele in sentences_list:
        cuts = jieba.cut(ele,cut_all=False)
        new_cuts = []
        for cut in cuts:
            if cut not in stopwords:
                new_cuts.append(cut)
        ncuts = (' '.join(new_cuts))
        sentences_cut.append(ncuts)
#    print(sentences_cut)

    fcname = 'fenci.txt'
    if not os.path.exists('fenci.txt'):
        with open(fcname,mode='w',encoding='UTF-8') as f:
            for ele in sentences_cut:
                f.write(ele)
    sentences = word2vec.Text8Corpus(u'fenci.txt')
    model = word2vec.Word2Vec(sentences)
    mdlname = "./word2vec.model"
    model.save(mdlname)

    ifile = open(txtfile, mode='r',encoding='UTF-8')
    sentences_list = ifile.readlines()
    ifile.close()
    sentences_cut = []
    for ele in sentences_list:
        cuts = jieba.cut(ele,cut_all=False)
        new_cuts = []
        for cut in cuts:
            if cut not in  stopwords:
                new_cuts.append(cut)
        ncuts = (' '.join(new_cuts))
        sentences_cut.append(ncuts)
#    print(sentences_cut)
    keylst = []
    i = 0
    for txt in sentences_cut:
        if (i > 0):
            print(txt)
            try:
                y = model.wv.similarity(txt)
            except KeyError:
            	y = 0
            keylst.append(y)
        i = i + 1
    print(keylst)
    return keylst

if __name__ == '__main__':

    filelst = ['appwords.txt ','reqwords.txt ','gapwords.txt ','valwords.txt ','prowords.txt ','chkwords.txt ','depwords.txt ']
    trnfile = "./corpus.txt"
    keylst = []
    for flst in filelst:
        kl = getkeysentc(trnfile,flst)
        keylst.append(kl)
