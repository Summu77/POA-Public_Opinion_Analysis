import argparse
import os
import json
import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import jieba
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import transformers
from transformers import AutoTokenizer
from gensim.models import KeyedVectors
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader
import json
import config
import data_loader
import utils
from model import Model

# change the format of the given text
def init_dict_element(txt_content):
    result = {"sentence": [], "ner": [], "words": []}
    for word in txt_content:
        if word:
            result["sentence"].append(word)
    return result

def sen2words(txt_content):
    words = list(jieba.cut(txt_content))
    words_idx = []
    idx = 0
    for word in words:
        letter_idx = []
        for letter in word:
            letter_idx.append(idx)
            idx = idx + 1
        words_idx.append(letter_idx)
    return words_idx

def get_json_element(text):
    json_list = []
    json_element = init_dict_element(text)
    words = sen2words(text)
        
    json_element['words'] = words
    json_list.append(json_element)
    return json_list

def data_process(demo_data, config):
    with open('./data/{}/train.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('./data/{}/dev.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    with open('./data/{}/test.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained("./cache/" + config.bert_name, cache_dir="./cache/")

    vocab = data_loader.Vocabulary()
    train_ent_num = data_loader.fill_vocab(vocab, train_data)
    dev_ent_num = data_loader.fill_vocab(vocab, dev_data)
    test_ent_num = data_loader.fill_vocab(vocab, test_data)
    
    config.label_num = len(vocab.label2id)
    config.vocab = vocab

    demo_dataset = data_loader.RelationDataset(*data_loader.process_bert(demo_data, tokenizer, vocab))
    
    return demo_dataset, demo_data
    
def predict_entities(text, config, epoch=1):
    model_path = 'model.pt'
    demo_data = get_json_element(text)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/conll03.json')
    parser.add_argument('--save_path', type=str, default='./model.pt')
    parser.add_argument('--predict_path', type=str, default='./output.json')
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--dist_emb_size', type=int)
    parser.add_argument('--type_emb_size', type=int)
    parser.add_argument('--lstm_hid_size', type=int)
    parser.add_argument('--conv_hid_size', type=int)
    parser.add_argument('--bert_hid_size', type=int)
    parser.add_argument('--ffnn_hid_size', type=int)
    parser.add_argument('--biaffine_size', type=int)

    parser.add_argument('--dilation', type=str, help="e.g. 1,2,3")

    parser.add_argument('--emb_dropout', type=float)
    parser.add_argument('--conv_dropout', type=float)
    parser.add_argument('--out_dropout', type=float)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--clip_grad_norm', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)

    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)
    parser.add_argument('--warm_factor', type=float)

    parser.add_argument('--use_bert_last_4_layers', type=int, help="1: true, 0: false")

    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    config = config.Config(args)

    logger = utils.get_logger(config.dataset)
    logger.info(config)
    config.logger = logger
    
    logger.info("Loading Data")
    
    datasets, ori_data = data_process(demo_data, config)
    
    test_loader = DataLoader(dataset=datasets,
                   batch_size=1,
                   collate_fn=data_loader.collate_fn,
                   shuffle=False,
                   num_workers=4,
                   drop_last=False)

    updates_total = len(datasets[0]) // config.batch_size * config.epochs
        
    
    model = Model(config)
    
    model = model.cuda()
        
    model.load_state_dict(torch.load(model_path))
    
    model.eval()

    pred_result = []
    label_result = []

    result = []
    i = 0
    with torch.no_grad():
        for data_batch in test_loader:
            sentence_batch = ori_data[i:i+config.batch_size]
            entity_text = data_batch[-1]
            data_batch = [data.cuda() for data in data_batch[:-1]]
            bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

            outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
            length = sent_length

            grid_mask2d = grid_mask2d.clone()

            outputs = torch.argmax(outputs, -1)
            ent_c, ent_p, ent_r, decode_entities = utils.decode(outputs.cpu().numpy(), entity_text, length.cpu().numpy())

            for ent_list, sentence in zip(decode_entities, sentence_batch):
                sentence = sentence["sentence"]
                instance = {"sentence": sentence, "entity": []}
                for ent in ent_list:
                    instance["entity"].append({"text": [sentence[x] for x in ent[0]],
                                               "idx": ent[0],
                                                "type": config.vocab.id_to_label(ent[1])})
                result.append(instance)

            grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
            outputs = outputs[grid_mask2d].contiguous().view(-1)

            label_result.append(grid_labels.cpu())
            pred_result.append(outputs.cpu())
            i += config.batch_size

    return result

if __name__ == '__main__':
    
    with open('/mnt/data/niesen/Conect/log/input_text.txt', 'r', encoding='utf-8') as file:
        text = file.read()
   
    print(text)
    result = predict_entities(text, config)

    with open('/mnt/data/niesen/Conect/log/model_save.json', 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=4)

    print(result)
    