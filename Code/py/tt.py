# _*_coding:utf-8_*_
import sys, getopt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

from gensim.models import KeyedVectors,word2vec,Word2Vec
import jieba
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing

def getWords(fname):
#    fname = "./corpus.txt"
    ifile = open(fname, mode='r',encoding='UTF-8')
    sentences_list = ifile.readlines()
    ifile.close()

    slst = []
    for sentence in sentences_list:
        slst.append(jieba.lcut(sentence))
#    print(slst)
    words = []
    for word in slst:
        words.append(' '.join(word))
#    print(words)
    return words

def getModel(fname):
    words = getWords(fname)
    mlen  = len(words)
    vectorizer = TfidfVectorizer(stop_words = ['了','的','呢','吗','得'])
    model = vectorizer.fit(words)
#    print(model.vocabulary_)
    return model,words,mlen

def calCosin(trnfile,tstfile):
    ifile = open(trnfile, mode='r',encoding='UTF-8')
    sentences_train = ifile.readlines()
    ifile.close()

    ifile = open(tstfile, mode='r',encoding='UTF-8')
    sentences_test = ifile.readlines()
    ifile.close()

    model,words,mlen = getModel(trnfile)
    tdata = model.transform(words).toarray().reshape(mlen,-1)
    slst = []
    rawrlst = []
    findlst = []
    for sent in sentences_test:
        sentence = jieba.lcut(sent)
        words = str.join(' ',sentence)
        list = []
        list.insert(0,words)
        adata = model.transform(list).toarray().reshape(1,-1)
        rslt = cosine_similarity(tdata,adata)
        rsltlst = np.array(rslt)
        argmax  = rsltlst.argmax()
        print(rsltlst[argmax])
        rawrlst.append(sent)
        if (rsltlst[argmax] > 0.75):
            getrslt = sentences_train[argmax]
            findlst.append(getrslt)
        else:
            findlst.append(['unmatched NaN'])
    return findlst,rawrlst

def mdlw2vtrain(trnfile,tstfile):

	with open('C:\\inetpub\\aigordr\\corpus.txt',encoding='utf-8') as f:
    document = f.read()
    document_cut = jieba.cut(document)
    result = ' '.join(document_cut)
    with open('./segment.txt', 'w',encoding="utf-8") as f2:
        f2.write(result)
	sentences = word2vec.LineSentence('./segment.txt')
	model = word2vec.Word2Vec(sentences_cut, window=5, min_count=0, workers=10, sg=1, hs=1, negative=1, seed=128, compute_loss=True)

#    load or prompt text
    ifile = open(trnfile, mode='r',encoding='UTF-8')
    sentences_list = ifile.readlines()
    ifile.close()
    sentences_cut = []
    for ele in sentences_list:
        cuts = jieba.cut(ele,cut_all=False)
        words = str.join(' ',cuts)
        list = []
        list.insert(0,words)
        sentences_cut.append(list)
    print(sentences_cut)

    sentences = word2vec.LineSentence(trnfile)
    model = word2vec.Word2Vec(sentences_cut, window=5, min_count=0, workers=10, sg=1, hs=1, negative=1, seed=128, compute_loss=True)
    for txt in sentences_cut:
        print("------------ %s"%txt)
        y = model.wv.most_similar(txt, topn=3)
        print(y)
    mdlname = "./word2vec.model"
    model.save(mdlname)

def mdltrainmore(trnfile):
    mdlname = "./word2vec.model"
    model = Word2Vec.load(mdlname)

    new_sentence = list(word2vec.LineSentence(trnfile))

#    stopwords = stopwordslist()
    sentences_cut = []

    for ele in new_sentence:
        cuts = jieba.cut(ele,cut_all=False)
#        new_cuts = []
#        for cut in cuts:
#            if cut not in  stopwords:
#                new_cuts.append(cut)
        sentences_cut.append(cuts)
    model.build_vocab(sentences_cut,update=True)
    model.train(sentences_cut,total_examples=model.corpus_count,epochs=10)
    model.save(mdlname)
    return model

if __name__ == '__main__':
    trnfile = "C:\\inetpub\\aigordr\\corpus.txt"
    tstfile = "C:\\inetpub\\aigordr\\rawreq.txt"
    mdlw2vtrain(trnfile,tstfile)
#    fdlst,rawrlst = calCosin(trnfile,tstfile)
#    i = 0
#    for lst1 in rawrlst:
#        print('raw requirements: %s, find req. is: %s'%(lst1,fdlst[i]))
#        i = i + 1
