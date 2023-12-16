# public-opinion-analysis
> Wuhan University Public Opinion Analysis Group Assignment (NS YXR FJL WCX)

> 中美关系主题下的实体立场检测
  
### 快速开始
我们提供两种快速开始的方式：
- 方式一：直接访问我们提供的在线平台 http://172.28.6.61:8501/ （目前仅支持武汉大学学生使用，若网站失效请与我们联系）
- 方式二：分别配置模型一和模型二的环境后，运行我们提供的脚本文件 webui.sh（位于connect文件夹下）

### Requirement-Model2
- python 3.6/3.7
- pytorch >= 0.4.0
- transformers
- sklearn

使用`pip install -r requirements.txt`下载依赖库，如果是rtx30x及以上系列显卡使用`pip install -r requirements_rtx30.txt`下载环境

### 手动下载预训练模型
对于NER任务的模型W2NER，需手动下载[bert-base-chinese](https://huggingface.co/bert-base-chinese)模型放置于./cache/bert-base-chinese文件夹下。

代码使用transformer自动下载bert和spacy预训练模型，如果产生http或ssl错误，则从此处[bert-base-chinese](https://huggingface.co/bert-base-chinese)手动下载预训练模型并在infer_example.py中
修改读取预训练模型的路径到本地路径。在新的数据集上训练也需在此处下载模型[zh_core_web_sm](https://spacy.io/models/zh)

### 重新训练
如需在新的数据集上训练则参考data.zip组织数据格式，我们也提供了直接将Brat标注平台生成的ann文件转换为txt文件并使用的代码，详情参考[data.zip](https://github.com/Summu77/public-opinion-analysis/blob/main/data%20.zip)

随后执行如下命令`python train.py --model_name bert_spc --dataset restaurant`开始训练

### 数据集
  我们提供已经标注好的数据集4000+供学习交流使用，请见data.zip
  该数据集收集自纽约时报中文、BBC中文、路透社等国外知名平台

### 模型原理与效果
  请见我们的实验报告

### 参考资料
- Li J, Fei H, Liu J, et al. Unified named entity recognition as word-word relation classification[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2022, 36(10): 10965-10973.
- Li Y, Garg K, Caragea C. A New Direction in Stance Detection: Target-Stance Extraction in the Wild[C]//Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2023: 10071-10085.
- Zeng B, Yang H, Xu R, et al. Lcf: A local context focus mechanism for aspect-based sentiment classification[J]. Applied Sciences, 2019, 9(16): 3389.
- Li Y, Zhao C, Caragea C. Improving stance detection with multi-dataset learning and knowledge distillation[C]//Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. 2021: 6332-6345.
- Sun C, Huang L, Qiu X. Utilizing BERT for aspect-based sentiment analysis via constructing auxiliary sentence[J]. arXiv preprint arXiv:1903.09588, 2019.
- https://github.com/google-research/bert

