#!/usr/bin/env python
# @time: 2019/9/24 15:07
# @author: wb
# @file: vocab.py

import numpy as np
from config import Config

class Vocab(object):
    '''
    实现生成词表，读取预训练词向量
    用于将字符转换成id，将id转换成字符
    输入initial_tokens，返回embeddings
    '''
    def __init__(self, filename=None, initial_tokens=None, lower=False):
        '''
        初始化，生成词表
        :param filename:
        :param initial_tokens:
        :param lower:
        '''
        # 读取配置文件
        self.config = Config()

        # 分为word和char
        # word
        # id->token 存放的是id:token
        self.id2word = {}
        # token->id 存放的是token:id
        self.word2id = {}
        # token的统计词数
        self.word_cnt = {}

        # char
        self.id2char = {}
        self.char2id = {}
        self.char_cnt = {}

        # 英文小写
        self.lower = lower

        # 填充标签，将question和passage填充到一定的长度
        self.pad_token = '<pad>'
        # 如果在词表中不存在，返回UNK
        self.unk_token = '<unk>'

        # 词向量维度、词表长度
        self.word_embed_size = self.config.get_default_params().word_embed_size
        self.char_embed_size = self.config.get_default_params().char_embed_size

        # 在本程序中，这个word_embeddings和char_embeddings并没有用到，没删掉，先留着吧
        # 感觉这个词表只需要词语和词语相应的id就行，这个embedding是在模型中使用的
        self.word_embeddings = None
        self.char_embeddings = None

        # 这个initial_tokens应该是初始传入的tokens
        # 如果为空就是空，不为空就是这个
        self.initial_tokens = initial_tokens if initial_tokens is not None else []
        # 添加补全和空
        self.initial_tokens.extend([self.pad_token, self.unk_token])
        # 所以只要输入initial_tokens，就能自己把tokens和ids存入词典中
        for token in self.initial_tokens:
            self.add_word(token)
            self.add_char(token)

        if filename is not None:
            self.load_from_file(filename)

    def load_from_file(self, file_path):
        '''
        读入文件
        :param file_path:
        :return:
        '''
        for line in open(file_path, 'r'):
            token = line.rstrip('\n')
            self.add_word(token)
            [self.add_char(ctoken) for ctoken in token]

    def get_vocab_size(self):
        '''
        获取词典的大小
        :return:
        '''
        return len(self.id2word)

    def get_char_size(self):
        '''
        获取char的大小
        :return:
        '''
        return len(self.id2char)

    def get_id_byword(self, token):
        '''
        获取token对应的id
        :param token:
        :return:
        '''
        token = token.lower() if self.lower else token
        try:
            return self.word2id[token]
        except KeyError:
            return self.word2id[self.unk_token]

    def get_word_byid(self, idx):
        '''
        获取id对应的token
        :param idx:
        :return:
        '''
        try:
            return self.id2word[idx]
        except KeyError:
            return self.unk_token

    def get_id_bychar(self, token):
        '''
        获取char对应的id
        :param token:
        :return:
        '''
        token = token.lower() if self.lower else token
        try:
            return self.char2id[token]
        except KeyError:
            return self.char2id[self.unk_token]

    def add_word(self, token, cnt=1):
        '''
        添加word到词典
        :param token: 单词
        :param cnt: 词频
        :return:
        '''
        # 需要小写就小写
        token = token.lower() if self.lower else token
        # 如果有就查询出id
        if token in self.word2id:
            idx = self.word2id[token]
        # 没有就在词表中添加
        else:
            idx = len(self.id2word)
            self.id2word[idx] = token
            self.word2id[token] = idx
        # 如果词表中有这个token，那么就把这个次数加上去
        # 如果没有，赋值为1
        if cnt > 0:
            if token in self.word_cnt:
                self.word_cnt[token] += cnt
            else:
                self.word_cnt[token] = cnt
        return idx

    def add_char(self, token, cnt=1):
        '''
        添加char到词典
        :param token: 单词
        :param cnt: 词频
        :return:
        '''
        token = token.lower() if self.lower else token
        if token in self.char2id:
            idx = self.char2id[token]
        else:
            idx = len(self.id2char)
            self.id2char[idx] = token
            self.char2id[token] = idx
        if cnt > 0:
            if token in self.char_cnt:
                self.char_cnt[token] += cnt
            else:
                self.char_cnt[token] = cnt
        return idx

    def filter_words_by_cnt(self, min_cnt):
        '''
        按计数过滤词汇中的标记
        :param min_cnt:
        :return:
        '''
        filtered_tokens = [token for token in self.word2id if self.word_cnt[token] >= min_cnt]
        # rebuild the token x id map
        self.word2id = {}
        self.id2word = {}
        for token in self.initial_tokens:
            self.add_word(token, cnt=0)
        for token in filtered_tokens:
            self.add_word(token, cnt=0)

    def filter_chars_by_cnt(self, min_cnt):
        '''
        按计数过滤char中的标记
        :param min_cnt: 最小的词频
        :return:
        '''
        filtered_tokens = [token for token in self.char2id if self.char_cnt[token] >= min_cnt]
        # rebuild the token x id map
        self.char2id = {}
        self.id2char = {}
        for token in self.initial_tokens:
            self.add_char(token, cnt=0)
        for token in filtered_tokens:
            self.add_char(token, cnt=0)

    def randomly_init_word_embeddings(self, word_embed_size):
        '''
        随机初始化token的词向量
        :param word_embed_size: word embedding size
        :return:
        '''
        self.word_embed_size = word_embed_size
        self.word_embeddings = np.random.rand(self.get_vocab_size(), self.word_embed_size)
        # 当单词不在词典中
        for token in [self.pad_token, self.unk_token]:
            self.word_embeddings[self.get_id_byword(token)] = np.zeros([self.word_embed_size])

    def randomly_init_char_embeddings(self, char_embed_size):
        '''
        随机初始化char的词向量
        :param char_embed_size: char embedding size
        :return:
        '''
        self.char_embed_size = char_embed_size
        self.char_embeddings = np.random.rand(self.get_char_size(), char_embed_size)
        for token in [self.pad_token, self.unk_token]:
            self.char_embeddings[self.get_id_bychar(token)] = np.zeros([self.char_embed_size])

    def load_pretrained_word_embeddings(self, word_vector_filepath):
        '''
        读取预训练的中文词向量
        :param word_vector_filepath: 预训练词向量文件
        :return:
        '''
        trained_embeddings = {}
        with open(word_vector_filepath, 'r', encoding='utf-8') as f:
            # 跳过第一行，因为第一行是词表的元信息
            next(f)
            while True:
                line = f.readline()  # 逐行读取
                if not line:  # 如果读取到文件末尾
                    break
                contents = line.strip().split()
                token = contents[0]

                if token not in self.word2id:
                    continue
                # 存入字典，token为key，词向量为value
                trained_embeddings[token] = list(map(float, contents[1:]))
                if self.word_embed_size is None:
                    self.word_embed_size = len(contents) - 1

        filtered_tokens = trained_embeddings.keys()
        # 重新构建token到id的映射
        self.word2id = {}
        self.id2word = {}
        # 把initial_tokens里面的token放进词典中
        for token in self.initial_tokens:
            self.add_word(token, cnt=0)
        # 把预训练词向量中的token也放入词表
        for token in filtered_tokens:
            self.add_word(token, cnt=0)
        # embeddings
        self.word_embeddings = np.zeros([self.get_vocab_size(), self.word_embed_size])
        # 在词表中的token
        for token in self.word2id.keys():
            # 如果token在预训练的词向量的token中
            if token in trained_embeddings.keys():
                # 获取词表中该token的id
                # 把这个id和预训练词向量中的token的词向量存入embeddings中
                self.word_embeddings[self.get_id_byword(token)] = trained_embeddings[token]

    def load_pretrained_char_embeddings(self, char_vector_filepath):
        '''
        读取预训练的中文词向量
        :param char_vector_filepath: 预训练词向量文件
        :return:
        '''
        trained_embeddings = {}
        with open(char_vector_filepath, 'r', encoding='utf-8') as f:
            # 跳过第一行，因为第一行是词表的元信息
            next(f)
            while True:
                line = f.readline()  # 逐行读取
                if not line:  # 如果读取到文件末尾
                    break

                contents = line.strip().split()
                token = contents[0]

                if token not in self.char2id:
                    continue
                # 存入字典，token为key，词向量为value
                trained_embeddings[token] = list(map(float, contents[1:]))
                if self.char_embed_size is None:
                    self.char_embed_size = len(contents) - 1

        filtered_chars = trained_embeddings.keys()
        # 重新构建char到id的映射
        self.char2id = {}
        self.id2char = {}
        # 把initial_tokens里面的token放进词典中
        for token in self.initial_tokens:
            self.add_char(token, cnt=0)
        # 把预训练词向量中的token也放入词表
        for token in filtered_chars:
            self.add_char(token, cnt=0)
        # embeddings
        self.char_embeddings = np.zeros([self.get_char_size(), self.char_embed_size])
        # 在词表中的token
        for token in self.char2id.keys():
            # 如果token在预训练的词向量的token中
            if token in trained_embeddings.keys():
                # 获取词表中该token的id
                # 把这个id和预训练词向量中的token的词向量存入embeddings中
                self.char_embeddings[self.get_id_bychar(token)] = trained_embeddings[token]

    def convert_word_to_ids(self, tokens):
        '''
        将tokens转换成id，如果token不在词典，使用unk
        :param tokens:
        :return:
        '''
        vec = [self.get_id_byword(label) for label in tokens]
        return vec

    def convert_char_to_ids(self, tokens):
        '''
        将char转换成id
        :param tokens:
        :return:
        '''
        vec = []
        for token in tokens:
            char_vec = []
            for char in token:
                char_vec.append(self.get_id_bychar(char))
            vec.append(char_vec)
        return vec

    def recover_word_from_ids(self, ids, stop_id=None):
        '''
        将id转换成token，如果遇到停止符就停止转换
        :param ids:
        :param stop_id:
        :return:
        '''
        tokens = []
        for i in ids:
            tokens += [self.get_word_byid(i)]
            if stop_id is not None and i == stop_id:
                break
        return tokens