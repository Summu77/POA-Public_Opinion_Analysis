# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle
import argparse


from spacy.tokens import Doc

class ChineseTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        # spaCy 的中文模型会自动进行分词
        return Doc(self.vocab, words=list(text), spaces=[False] * len(text))

# 创建中文字符分词器
nlp = spacy.load("zh_core_web_sm")
nlp.tokenizer = ChineseTokenizer(nlp.vocab)

def dependency_adj_matrix(text):
    print("text:", text)
    tokens = nlp(text)
    # print("tokens:", tokens)
    words = list(text)
    # print("words:", words)
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))
    
    # 输出分词和词性标注
    # for token in tokens:
    #     print(token.text, token.pos_)

    for token in tokens:
        matrix[token.i][token.i] = 1
        for child in token.children:
            matrix[token.i][child.i] = 1
            matrix[child.i][token.i] = 1
    # print("matrix:", matrix)
    return matrix

def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph', 'wb')
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].strip()
        adj_matrix = dependency_adj_matrix(text_left+' '+aspect+' '+text_right)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)        
    fout.close() 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, type=str, help='path to dataset')
    opt = parser.parse_args()
    process('./my_datasets/output.raw')

    # process('./datasets/acl-14-short-data/train.raw')
    # process('./datasets/acl-14-short-data/test.raw')
    # process('./datasets/semeval14/Restaurants_Train.xml.seg')
    # process('./datasets/semeval14/Restaurants_Test_Gold.xml.seg')
    # process('./datasets/semeval14/Laptops_Train.xml.seg')
    # process('./datasets/semeval14/Laptops_Test_Gold.xml.seg')

