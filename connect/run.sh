#!/bin/bash
# 说明：请忽略中间关于conda init的输出信息


source activate W2NER
cd /mnt/data/niesen/W2NER/W2NER
python demo.py --config ./config/news.json


python /mnt/data/niesen/Conect/data_process.py


source activate bert
cd /mnt/data/niesen/ABSA-PyTorch
python infer_example.py
