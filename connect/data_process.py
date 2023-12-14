import json


with open('/mnt/data/niesen/Conect/log/model_save.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

sentences = data[0]['sentence']
entities = data[0]['entity']


############### 这里是过滤掉重复的实体 #####################
# filtered_entities = []
# seen_texts = set()

# for entity in entities:
#     text = tuple(entity['text'])  
#     if text not in seen_texts:
#         filtered_entities.append(entity)
#         seen_texts.add(text)

# entities = filtered_entities

# print(entities)

############### 下面是确定偏移 #####################
entity_offsets = []
for entity in entities:
    offset = entity['idx']
    entity_offsets.append(offset)

result = [''.join(sentences)] + entity_offsets
print(f"数据处理之前：{data}")
print(f"数据处理之后：{result}")

with open('/mnt/data/niesen/Conect/log/model_save_processed.json', 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=4)
