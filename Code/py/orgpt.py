# _*_coding:UTF-8-sig_*_
import sys, getopt, re
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
    stopwords = [ line.strip() for line in open(fname, mode='r',encoding='UTF-8-sig').readlines()]
    return stopwords

def mdlw2vtrain():
#    load or prompt text
    fname = "./sortprompt.txt"
    ifile = open(fname, mode='r')
    sentences_list = ifile.readlines()
    ifile.close()

    stopwords = stopwordslist()
    sentences_cut = []

    for ele in sentences_list:
        cuts = jieba.cut(ele,cut_all=False)
        new_cuts = []
        for cut in cuts:
            if cut not in  stopwords:
                new_cuts.append(cut)
        sentences_cut.append(new_cuts)
    print(sentences_cut)

    fname = "./fcdata.txt"
    with open(fname,'w') as f:
        for ele in sentences_cut:
            ele = ele + '\n'
            f.write(ele)
    sentences = list(word2vec.LineSentence(fname))

    mdlname = "./word2vec.model"
    model = Word2Vec.load(fname)
    for txt in sentences:
        y = model.wv.most_similar(txt, topn=3)
        print(y)

def mdltrainmore():
    mdlname = "./word2vec.model"
    model = Word2Vec.load(fname)

    fname = "./corpus.txt"
    new_sentence = list(word2vec.LineSentence(fname))

    stopwords = stopwordslist()
    sentences_cut = []

    for ele in new_sentence:
        cuts = jieba.cut(ele,cut_all=False)
        new_cuts = []
        for cut in cuts:
            if cut not in  stopwords:
                new_cuts.append(cut)
        sentences_cut.append(new_cuts)
    model.build_vocab(sentences_cut,update=True)
    model.train(sentences_cut,total_examples=model.corpus_count,epochs=10)
    mdlname = "./word2vec.model"
    model.save(fname)
    return model


def init_params(label='TfidfVectorizer'):
    params_count={
        'analyzer': 'word',  # 取值'word'-分词结果为词级、'char'-字符级(结果会出现he is，空格在中间的情况)、'char_wb'-字符级(以单词为边界)，默认值为'word'
        'binary': False,  # boolean类型，设置为True，则所有非零计数都设置为1.（即，tf的值只有0和1，表示出现和不出现）
        'decode_error': 'strict',
        'dtype': np.float64, # 输出矩阵的数值类型
        'encoding': 'UTF-8-sig',
        'input': 'content', # 取值filename，文本内容所在的文件名；file，序列项必须有一个'read'方法，被调用来获取内存中的字节；content，直接输入文本字符串
        'lowercase': True, # boolean类型，计算之前是否将所有字符转换为小写。
        'max_df': 1.0, # 词汇表中忽略文档频率高于该值的词；取值在[0,1]之间的小数时表示文档频率的阈值，取值为整数时(>1)表示文档频数的阈值；如果设置了vocabulary，则忽略此参数。
        'min_df': 1, # 词汇表中忽略文档频率低于该值的词；取值在[0,1]之间的小数时表示文档频率的阈值，取值为整数时(>1)表示文档频数的阈值；如果设置了vocabulary，则忽略此参数。
        'max_features': None, # int或 None(默认值).设置int值时建立一个词汇表，仅用词频排序的前max_features个词创建语料库；如果设置了vocabulary，则忽略此参数。
        'ngram_range': (1, 2),  # 要提取的n-grams中n值范围的下限和上限，min_n <= n <= max_n。
        'preprocessor': None, # 覆盖预处理（字符串转换）阶段，同时保留标记化和 n-gram 生成步骤。仅适用于analyzer不可调用的情况。
        'stop_words': 'english', # 仅适用于analyzer='word'。取值english，使用内置的英语停用词表；list，自行设置停停用词列表；默认值None，不会处理停用词
        'strip_accents': None,
        'token_pattern': '(?u)\\b\\w\\w+\\b', # 分词方式、正则表达式，默认筛选长度>=2的字母和数字混合字符（标点符号被当作分隔符）。仅在analyzer='word'时使用。
        'tokenizer': None, # 覆盖字符串标记化步骤，同时保留预处理和 n-gram 生成步骤。仅适用于analyzer='word'
        'vocabulary': None, # 自行设置词汇表（可设置字典），如果没有给出，则从输入文件/文本中确定词汇表
    }
    params_tfidf={
        'norm': None, # 输出结果是否标准化/归一化。l2：向量元素的平方和为1，当应用l2范数时，两个向量之间的余弦相似度是它们的点积；l1：向量元素的绝对值之和为1
        'smooth_idf': True, # 在文档频率上加1来平滑 idf ，避免分母为0
        'sublinear_tf': False, # 应用次线性 tf 缩放，即将 tf 替换为 1 + log(tf)
        'use_idf': True, # 是否计算idf，布尔值，False时idf=1。
    }
    if label=='CountVectorizer':
        return params_count
    elif label=='TfidfTransformer':
        return params_tfidf
    elif label=='TfidfVectorizer':
        params_count.update(params_tfidf)
        return params_count

def getWords(fname):
#    fname = "./corpus.txt"
    ifile = open(fname, mode='r',encoding='UTF-8-sig')
    sentences_list = ifile.readlines()
    ifile.close()

    stopwords = stopwordslist()
#    words = []
#
#    for ele in sentences_list:
#        cuts = jieba.lcut(ele,cut_all=False)
#        new_cuts = []
#        for cut in cuts:
#            if cut not in  stopwords:
#                new_cuts.append(cut)
#        words.append(''.join(new_cuts))
#    print(words)

    slst = []
    for sentence in sentences_list:
        cuts = jieba.lcut(sentence)
        new_cuts = []
        for cut in cuts:
            if cut not in stopwords:
                new_cuts.append(cut)
        slst.append(new_cuts)
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
    ifile = open(trnfile, mode='r',encoding='UTF-8-sig')
    sentences_train = ifile.readlines()
    ifile.close()

    ifile = open(tstfile, mode='r',encoding='UTF-8-sig')
    sentences_test = ifile.readlines()
    ifile.close()
    stopwords = stopwordslist()
    test_words = []

    slst = []
    for sentence in sentences_test:
        cuts = jieba.lcut(sentence)
        new_cuts = []
        for cut in cuts:
            if cut not in stopwords:
                new_cuts.append(cut)
        slst.append(new_cuts)
    for word in slst:
        test_words.append(' '.join(word))

    model,words,mlen = getModel(trnfile)
    tdata = model.transform(words).toarray().reshape(mlen,-1)
    slst = []
    findlst = []
    unfindlst = []
    findseq = []
    i = 0
    for sent in test_words:
#        sentence = jieba.lcut(sent)
#        words = str.join(' ',sentence)
        list = []
#        list.insert(0,words)
        list.insert(0,sent)
        adata = model.transform(list).toarray().reshape(1,-1)
        rslt = cosine_similarity(tdata,adata)
        rsltlst = np.array(rslt)
        argmax  = rsltlst.argmax()
        print(rsltlst[argmax])
        if (rsltlst[argmax] > 0.75):
            getrslt = sentences_train[argmax]
            findlst.append(getrslt)
            findseq.append(1)
        else:
            unfindlst.append(sent)
            findseq.append(0)
        i = i + 1
    return findlst,unfindlst,findseq

def getkeysentc():
    fname = "./sortprompt.txt"
    ifile = open(fname, mode='r')
    sentences_list = ifile.readlines()
    ifile.close()

    for ele in sentences_list:
        comm = ele
        result = np.array(consine(comm))
        argmax = result.argmax()
        data = command[argmax][1]
        print('命令：{0}\n回复：{1}'.format(comm,data))

if __name__ == '__main__':

#进行标签分类后的原始需求文本的需求相似度分析，以生成OR需求
    trnfile = "./corpus.txt"
    tstfile = "./rawreq.txt"
    ifile = open(tstfile, mode='r',encoding='UTF-8-sig')
    sentences_test = ifile.readlines()
    ifile.close()
    fdlst,rawrlst,fdseq = calCosin(trnfile,tstfile)
    i = 0
    j = 0
    for lst1 in sentences_test:
        if(fdseq[i] == 1):
            print('NO%d: raw requirements: %s, find req. is: %s'%(i+1,lst1,fdlst[j]))
            j = j + 1
        else:
            print('NO%d: raw requirements: %s, unfind req.'%(i+1,lst1))
        i = i + 1


#加载生成的原始需求提问prompt文本
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
    modelName = sys.argv[1]
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

#将相似度匹配的OR需求文本作为输入，使用GPT模型进行提问生成最终准确的OR需求
    for txt in prmts:
        prompt = instruction.format(txt)
        gpt_ids = model.generate(tokenizer(prompt, return_tensors='pt').input_ids.cuda(), max_new_tokens=4096, streamer=streamer)