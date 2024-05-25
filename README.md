# POA-Public_Opinion_Analysis

This project is the final project of the public opinion analysis course.

SCHOOL OF CYBER SCIENCE AND ENGINEERING-WHU

The following are the original requirements for the experiment:

Task 2: Entity-Position Analysis

l **Description**

Analyze the position of each entity for all the entities that appear in the data.

l **Data**

- Entities: people, institutions, policies
- Position: +/-/N
- Fields: Taiwan situation, cross-strait relations, Sino-US relations, vaccines (vaccine name, country, person)
- Data: 3000 articles
- Genre: News/Twitter

l **Requirement**

1. Complete data collection and perform visual display of data information, including data content, quantity, and collection platform
2. After completing the model construction, the analysis results can be displayed in the form of a web page.
3. Complete the writing of the experimental report and submit the data and complete code. The code part needs to describe the installation and deployment steps to ensure that it can be correctly

Based on the above requirements, we have chosen the substantive position analysis on the theme of Sino-US relations.

We provide annotated datasets on Sino-US relations, sourced from major news sites.

## Overview

The training results of model 1 and model 2:

![image](https://github.com/Summu77/POA-Public_Opinion_Analysis/assets/115442864/67bed3a7-6502-401c-8351-717c626ae522)

![image](https://github.com/Summu77/POA-Public_Opinion_Analysis/assets/115442864/ddfadbaa-1a46-409a-8df8-9cce569cd8c5)

The data visualization is shown below:

![image](https://github.com/Summu77/POA-Public_Opinion_Analysis/assets/115442864/12bfa5ad-ccf5-4ce4-a4eb-f43825dda311)


Some interesting examples:

![image](https://github.com/Summu77/POA-Public_Opinion_Analysis/assets/115442864/e8bb5264-6bde-4176-8613-148ee583d545)

If you are not satisfied with the above results, please contact us and we will deal with it promptly.

## Method

We refer to the paper A New Direction in Stance Detection: Target-Stance Extraction in the Wild and divide the overall task of the model into two sub-tasks. Sub-task one: realize named entity recognition. Sub-task two: perform entity stance judgment, and The difference between the paper is that the paper extracts the topic and then determines the polarity, while our project extracts the entity and then determines the position. The two-stage task diagram is as follows:

![image](https://github.com/Summu77/POA-Public_Opinion_Analysis/assets/115442864/9317620b-61dd-4b7b-9a5c-7410b5c9ae97)


If you want to learn more about the principles and details of the method, please refer to our experimental report.

## Quick start

We offer two ways to get started quickly !

- Method 1: Direct access to the online platform we provide.  http://172.28.6.61:8501/ 

  (Currently, it is only available to students of Wuhan University, please contact us if the website is invalid!)

- Method 2: Configure the environments of model 1 and model 2 and run the script file provided.

l **Requirement-Model1**

- numpy (1.21.4)
- torch (1.10.0)
- gensim (4.1.2)
- transformers (4.13.0)
- pandas (1.3.4)
- scikit-learn (1.0.1)
- prettytable (2.4.0)

For the NER task model W2NER, you need to manually download the [bert-base-chinese](https://huggingface.co/bert-base-chinese) model and place it in the ./cache/bert-base-chinese folder.

l **Requirement-Model2**

- pytorch >= 0.4.0
- transformers
- sklearn

Use `pip install -r requirements.txt` to download dependent libraries. If it is rtx30x and above series graphics cards, use `pip install -r requirements_rtx30.txt` to download the environment.

For model two, the code uses transformer to automatically download the bert and spacy pre-trained models. If an http or ssl error occurs, download it manually from here [bert-base-chinese](https://huggingface.co/bert-base-chinese) Pretrain the model and in infer_example.py

Modify the path for reading the pre-trained model to the local path. To train on a new data set, you also need to download the model here [zh_core_web_sm](https://spacy.io/models/zh)

The above two models should be configured in different Conda environments, and our script files will be automatically switched at runtime. If you need to customize the script file, you can directly modify it to suit your needs.

## Dataset

The theme chosen for this experiment is Sino-US relations. In order to obtain a more extensive and real data set, we crawled data from the following websites through technical means, and cleaned and preprocessed the data:

- BBC News is the news arm of the British Broadcasting Corporation, providing global news coverage and analysis, including coverage of China-U.S. relations. 
- CNN is a well-known American cable news network that provides global news reports, including news on Sino-US relations.
- Reuters is one of the world's leading news organizations, providing independent journalism and international news services. Its reporting is generally known for being neutral and objective. Reuters provides global news and market information, with in-depth and extensive coverage of financial markets and economic events.
- The New York Times is a well-known newspaper in the United States, providing a variety of reports including international news. Its coverage often features in-depth analysis and commentary. 
- The content of Asia Weekly covers a wide range of news and events in various Asian countries and regions, including politics, economy, society, culture and other fields. 

There are a lot of open source resources for crawlers on the Internet. It is inefficient to build a crawler tool from scratch. Therefore, our team uses the popular crawler tool on github - EasySpider, a visual browser automated testing/data collection, it  can graphically design and execute crawler tasks.

![image](https://github.com/Summu77/POA-Public_Opinion_Analysis/assets/115442864/8a9ad9fd-d215-4eb7-af2e-d4e55ee6dbe4)

Above we have shown the amount of data we get from various websites.

## Training

l **Model1**

Place the formatted data set in the ./data/news folder

Run `python main.py --config ./config/news.json` in the model 1 project folder to start training

l **Model2**

If you need to train on a new data set, please refer to data.zip to organize the data format. We also provide code that directly converts the ann file generated by the Brat annotation platform into a txt file and uses it. For details, please refer to data.zip.

Then execute the following command `python train.py --model_name bert_spc --dataset restaurant` to start training

## Reference

- Li J, Fei H, Liu J, et al. Unified named entity recognition as word-word relation classification[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2022, 36(10): 10965-10973.
- Li Y, Garg K, Caragea C. A New Direction in Stance Detection: Target-Stance Extraction in the Wild[C]//Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2023: 10071-10085.
- Zeng B, Yang H, Xu R, et al. Lcf: A local context focus mechanism for aspect-based sentiment classification[J]. Applied Sciences, 2019, 9(16): 3389.
- Li Y, Zhao C, Caragea C. Improving stance detection with multi-dataset learning and knowledge distillation[C]//Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. 2021: 6332-6345.
- Sun C, Huang L, Qiu X. Utilizing BERT for aspect-based sentiment analysis via constructing auxiliary sentence[J]. arXiv preprint arXiv:1903.09588, 2019.
- https://github.com/google-research/bert
