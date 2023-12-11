import os
import re
import json
import jieba
import random
import shutil
from tqdm import tqdm

LABEL_DICT = set(['Positive', 'Negative', 'Neutral'])

# initialize a dict element in the .json file and fill the key of "sentence"
def init_dict_element(txt_content):
    result = {"sentence": [], "ner": [], "words": []}
    for word in txt_content:
        if word:
            result["sentence"].append(word)
    return result

# return a list with all the NER elements in .ann file to be filled in the key of "ner" 
def process_ann_file(ann_content):
    annotations = re.findall(r'T\d+\s(\w+)\s(\d+)\s(\d+)\s(\S+)', ann_content)
    result = []
    for annotation in annotations:
        entity_type, start, end, text = annotation
        if entity_type not in LABEL_DICT:
            result.append({"index": list(range(int(start), int(end))), "type": entity_type})
    return result

# use jieba to segment the sentence and fill the key of "words"
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

def get_json_element(txt_file_path, ann_file_path):
    with open(txt_file_path, 'r', encoding='utf-8') as txt_file, open(ann_file_path, 'r', encoding='utf-8') as ann_file:
        txt_content = txt_file.read()
        ann_content = ann_file.read()
        
        json_element = init_dict_element(txt_content)
        ann_data = process_ann_file(ann_content)
        words = sen2words(txt_content)
        
        json_element['ner'] = ann_data 
        json_element['words'] = words
    return json_element

def get_all_files_in_folder(folder_path):
    try:
        # Get a list of all files and directories in the specified folder
        files_and_directories = os.listdir(folder_path)
        
        # Filter out only the files from the list
        files = [f for f in files_and_directories if os.path.isfile(os.path.join(folder_path, f))]
        
        return files

    except OSError as e:
        print(f"Error reading files in folder {folder_path}: {e}")
        return []
    
def split_dataset(input_folder, output_folder, file_list, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1, seed_value=42):
    random.seed(seed_value)
    random.shuffle(file_list)
    
    num_files = len(file_list)
    train_split = int(num_files * train_ratio)
    test_split = int(num_files * (train_ratio + test_ratio))
    
    train_files = file_list[:train_split]
    test_files = file_list[train_split:test_split]
    val_files = file_list[test_split:]
    
    os.makedirs(os.path.join(output_folder, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'val'), exist_ok=True)
    
    for file in train_files:
        shutil.copy(os.path.join(input_folder, file), os.path.join(output_folder, 'train', file))

    for file in test_files:
        shutil.copy(os.path.join(input_folder, file), os.path.join(output_folder, 'test', file))

    for file in val_files:
        shutil.copy(os.path.join(input_folder, file), os.path.join(output_folder, 'val', file))
    
txt_folder_path = 'data/txt_files/val/'
ann_folder_path = 'data/ann_files/val/'
json_file_path = 'data/processed/val.json'
txt_file_names = get_all_files_in_folder(txt_folder_path)
ann_file_names = get_all_files_in_folder(ann_folder_path)


json_list = []
for txt_file, ann_file in tqdm(zip(txt_file_names, ann_file_names), desc="Reading Files", unit="file"):
    name1, _ = os.path.splitext(txt_file)
    name2, _ = os.path.splitext(ann_file)
    
    if name1 == name2:
        element = get_json_element(txt_folder_path + txt_file, ann_folder_path + ann_file)
        json_list.append(element)

with open(json_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(json_list, json_file, ensure_ascii=False, indent=None)