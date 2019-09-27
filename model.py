#!/usr/bin/env python
# @time: 2019/9/25 9:11
# @author: wb
# @file: model.py

import numpy as np
import tqdm
import tensorflow as tf
import logging
from config import Config


class Model(object):
    '''
    模型文件，保存RC模型
    '''

    def __init__(self, vocab):
        self.config = Config()

        # logger
        self.logger = logging.getLogger("brc")
        # vocabulary
        self.vocab = vocab
        self.trainable = trainable
        # 使用的优化函数
        self.optim_type = 'adam'

        # batch的size
        self.batch_size = self.config.get_default_params().batch_size
        # 隐藏单元
        self.char_hidden = self.config.get_default_params().char_hidden
        self.hidden_size = self.config.get_default_params().hidden_size
        self.attn_size = self.config.get_default_params().attn_size

        # size limit
        self.max_p_num = self.config.get_default_params().max_p_num
        self.max_p_len = self.config.get_default_params().max_p_len
        self.max_q_len = self.config.get_default_params().max_q_len
        self.max_a_len = self.config.get_default_params().max_a_len
        self.max_ch_len = self.config.get_default_params().max_ch_len

        # gru单元，是否使用cundnn
        self.gru = cudnn_gru if self.config.get_default_params().use_cudnn else native_gru
        # keep_prob
        self.keep_prob = self.config.get_default_params().keep_prob
        # ptr_keep_prob
        self.ptr_keep_prob = self.config.get_default_params().ptr_keep_prob

        self.weight_decay = self.config.get_default_params().weight_decay

        # session info
        sess_config = tf.ConfigProto()
        # 程序按需申请内存
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        # 构建计算图
        self._build_graph()

        # 保存模型
        self.saver = tf.train.Saver()

        # initialize the model
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):