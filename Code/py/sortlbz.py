# _*_coding:UTF-8-sig_*_
import sys, getopt, re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

from gensim.models import KeyedVectors,word2vec,Word2Vec
import jieba
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing

def cutsent(wdfile,opt):
    if opt:
        ifile = open(wdfile, mode='r',encoding='UTF-8-sig')
        sentlist = ifile.readlines()
        ifile.close()
    else:
        txt = open(wdfile, mode='r',encoding='UTF-8-sig').read()
        pattern = r'\;|\?|\n|!|。|；|！'
        sentlist = re.split(pattern, txt)
    for sent in sentlist:
        print(sent)
    return sentlist

def stopwordslist():
    fname = "./stopword.txt"
    stopwords = [ line.strip() for line in open(fname, mode='r',encoding='UTF-8-sig').readlines()]
    return stopwords

def appmtch(wdfile,sentences):
    ifile = open(wdfile, mode='r',encoding='UTF-8-sig')
    wordlist = ifile.readlines()
    ifile.close()

    istrlst = []
    applist = []
    for wd in wordlist:
        wd = wd.replace('\n','')
        wd = wd.lstrip()
        if(wd[0] == '-'):
            wd = wd[1:]
        istrlst.append('*'+wd)
        for isent in sentences:
            if(wd[0] == '-'):
                if(re.search(r'(^.*?)'+wd, isent, re.I)):
                    applist.append(isent)
            else:
                mtch1 = 0
                if(re.search(r'(^.*?)'+wd, isent, re.I)):
                    mtch1 = 1
                else:
                    mtch1 = 0
                cuts = jieba.lcut(wd)
                mtch2 = 0
                ncuts = ''
                jsent = isent
                for cut in cuts:
                    ncuts = ncuts + '*' + cut
                    rslt = re.search(r'(^.*?)'+cut, isent, re.I)
                    if rslt:
                        isent = isent[rslt.span(0)[1]:]
                        mtch2 = 1
                    else:
                        isent = isent
                        mtch2 = 0
#                    print(isent+'----'+wd+'----'+str(mtch))

                if mtch1 == 1 or mtch2 == 1:
                    applist.append(jsent)
        istrlst.append(ncuts)
#    print(istrlst)
    return applist

def gapmtch(wdfile,sentences):
    ifile = open(wdfile, mode='r',encoding='UTF-8-sig')
    wordlist = ifile.readlines()
    ifile.close()

    gaplist = []
    poslist = []
    wdlist1 = []
    wdlist2 = []
    wdlist3 = []
    wdlist4 = []
    wdlist5 = []
    wdlist6 = []
    for wd in wordlist:
        wd = wd.replace('\n','')
        wd = wd.lstrip()
        if(wd[0] == '1'):
            wdlist1.append(wd[1:])
        if(wd[0] == '2'):
            wdlist2.append(wd[1:])
        if(wd[0] == '3'):
            wdlist3.append(wd[1:])
        if(wd[0] == '4'):
            wdlist4.append(wd[1:])
        if(wd[0] == '5'):
            wdlist5.append(wd[1:])
        if(wd[0] == '6'):
            wdlist6.append(wd[1:])

    i = 0
    for isent in sentences:
        rslt1 = 0
        rslt2 = 0
        rslt3 = 0
        rslt4 = 0

        for wd1 in wdlist1:
            if rslt1:
                break
            for wd4 in wdlist4:
                if rslt1:
                    break
                for wd3 in wdlist3:
                    if rslt1:
                        break
                    for wd6 in wdlist6:
                        pt = '^(?=.*'+wd1+')(?=.*'+wd4+')(?=.*'+wd3+')|(?=.*'+wd6+').*$'
                        rslt1 = re.search(pt, isent, re.I)
                        if rslt1:
                            gaplist.append(isent)
                            ps = rslt1.start()
                            poslist.append(str(i)+'-'+str(ps))
                            break
        if rslt1:
#            print(str(i)+'--------------'+pt)
            continue
        for wd2 in wdlist2:
            if rslt2:
                break
            for wd4 in wdlist4:
                if rslt2:
                    break
                for wd3 in wdlist3:
                    if rslt2:
                        break
                    for wd6 in wdlist6:
                        pt = '^(?=.*'+wd2+')(?=.*'+wd4+')(?=.*'+wd3+')|(?=.*'+wd6+').*$'
                        rslt2 = re.search(pt, isent, re.I)
                        if rslt2:
                            gaplist.append(isent)
                            ps = rslt1.start()
                            poslist.append(str(i)+'-'+str(ps))
                            break
        if rslt2:
            continue
        for wd5 in wdlist5:
            if rslt3:
                break
            for wd4 in wdlist4:
                if rslt3:
                    break
                for wd3 in wdlist3:
                    if rslt3:
                        break
                    for wd6 in wdlist6:
                        pt = '^(?=.*'+wd5+')(?=.*'+wd4+')(?=.*'+wd3+')|(?=.*'+wd6+').*$'
                        rslt3 = re.search(pt, isent, re.I)
                        if rslt3:
                            gaplist.append(isent)
                            ps = rslt1.start()
                            poslist.append(str(i)+'-'+str(ps))
                            break
        if rslt3:
            continue
        for wd4 in wdlist4:
            if rslt4:
                break
            for wd3 in wdlist3:
                if rslt4:
                    break
                for wd6 in wdlist6:
                    pt = '^(?=.*'+wd4+')(?=.*'+wd3+')|(?=.*'+wd6+').*$'
                    rslt4 = re.search(pt, isent, re.I)
                    if rslt4:
                        gaplist.append(isent)
                        ps = rslt1.start()
                        poslist.append(str(i)+'-'+str(ps))
                        break

        i = i + 1
    return gaplist,poslist

def cossim_score(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity_matrix = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    similarity_score = similarity_matrix[0][0]
    return similarity

if __name__ == '__main__':

#七大类型建模进入初步匹配标签分类
    sentlist = cutsent('corpus.txt',1)
    gaplist,gappslst = gapmtch('gapwords.txt',sentlist)
    applist,apppslst = appmtch('appwords.txt',sentlist)
    reqlist,reqpslst = reqmtch('reqwords.txt',sentlist)
    vallist,valpslst = valmtch('valwords.txt',sentlist)
    prolist,propslst = promtch('prowords.txt',sentlist)
    chklist,chkpslst = chkmtch('chkwords.txt',sentlist)
    deplist,deppslst = depmtch('depwords.txt',sentlist)

#加载HTML生成的七大类型标签prompt提问文本
    fname = "./sortprompt.txt"
    ifile = open(fname, mode='r',encoding='UTF-8-sig')
    prmts = ifile.readlines()
    ifile.close()
    instruction = """[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{} [/INST]"""
    for txt in prmts:
        prompt = instruction.format(txt)
#        print(prompt)

#加载GPT模型
    modelName = '' #sys.argv[1]
#    model_path = "./models/Chinese-Llama-2-7b"
    model_path = "./models/"+modelName
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if model_path.endswith("4bit"):
        model = AutoModelForCausalLM.from_pretrained(
                model_path,
                load_in_4bit=True,
                torch_dtype=torch.float16,
                device_map='auto')
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path).half().cuda()
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

#将raw文本输入模型进行学习和训练
#    model = mdltrain(model,'corpus.txt')

#将七大分类匹配标签作为输入，使用GPT模型进行提问判断匹配标签
    i = 0
    gapgroup = []
    appgroup = []
    reqgroup = []
    valgroup = []
    progroup = []
    chkgroup = []
    deppsgrp = []
    gappsgrp = []
    apppsgrp = []
    reqpsgrp = []
    valpsgrp = []
    propsgrp = []
    chkpsgrp = []
    deppsgrp = []
    for isent in sentlist:
        j = 0
        for txt in prmts:
            prompt = instruction.format(txt)
            gpt_ids = model.generate(tokenizer(prompt, return_tensors='pt').input_ids.cuda(), max_new_tokens=4096, streamer=streamer)
            if gpt_ids:
                if(j==0):
                    appgroup.append(isent)
                    apppsgrp.append(i)
                else if(j==5):
                    reqgroup.append(isent)
                    reqpsgrp.append(i)
                else if(j==8):
                    gapgroup.append(isent)
                    gappsgrp.append(i)
                else if(j==12):
                    progroup.append(isent)
                    propsgrp.append(i)
                else if(j==16):
                    valgroup.append(isent)
                    valpsgrp.append(i)
                else if(j==19):
                    chkgroup.append(isent)
                    chkpsgrp.append(i)
                else if(j==22):
                    depgroup.append(isent)
                    deppsgrp.append(i)
            j = j + 1
        i = i + 1

#ppo策略去冲突及联合决策输出类型标签
#    ppoadj(applist,appgroup,apppslst,apppsgrp)
#    ppoadj(reqlist,reqgroup,reqpslst,reqpsgrp)