# 机器阅读理解模型改进

本项目是在毕业设计的机器阅读理解模型上进行改进，结合其他的模型思路以及自己的想法，争取做出改进模型。

## 数据

数据依旧使用百度的数据集Dureader。

### 数据内容

百度提供的数据是百度知道与百度搜索中的真实数据，都是由人提问回答，然后选取问题的正确答案进行标注。

### 数据特点

正是因为是真实数据，由用户产生的，所以QA用语比较随意，通俗。

- 所有的问题、原文都来源于实际数据（百度搜索引擎数据和百度知道问答社区），答案是由人类回答的。
- 数据集中包含大量的之前很少研究的**是非和观点类**的样本。
- **每个问题都对应多个答案**，数据集包含200k问题、1000k原文和420k答案，是目前最大的中文MRC数据集。

### question的分析

根据答案类型，DuReader将问题分为：**Entity（实体）、Description（描述）和YesNo（是非）**。

- 对于**实体类问题**，其答案一般是单一确定的回答，比如：iPhone是哪天发布？
- 对于**描述类问题**，其答案一般较长，是多个句子的总结，典型的how/why类型的问题，比如：消防车为什么是红的？
- 对于**是非类问题**，其答案往往较简单，是或者否，比如：39.5度算高烧吗？