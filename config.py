#!/usr/bin/env python
# @time: 2019/9/24 15:24
# @author: wb
# @file: config.py
import os
import tensorflow as tf

class Config(object):
    '''
    配置文件，包括file_path，超参数
    '''
    # 文件参数
    def get_filepath(self):
        '''
            所有的文件都在data文件夹中
            如zhidao的数据集就在zhidao文件夹中
        '''
        # 训练文件文件名
        data = os.path.expanduser("data")
        zhidao_train_file = os.path.join(data, "zhidao", "zhidao.train.json")
        zhidao_dev_file = os.path.join(data, "zhidao", "zhidao.dev.json")
        zhidao_test_file = os.path.join(data, "zhidao", "zhidao.test.json")

        search_train_file = os.path.join(data, "search", "search.train.json")
        search_dev_file = os.path.join(data, "search", "search.dev.json")
        search_test_file = os.path.join(data, "search", "search.test.json")

        # 输出文件夹
        output_dir = os.path.join(data, "output")

        # 模型保存文件夹
        model_dir = os.path.join(data, "models")

        # 统计，tensorboard页面
        summary_dir = os.path.join(data, "summary")

        # vocab dir
        vocab_dir = os.path.join(data, "vocab")

        # 日志文件
        log_file = os.path.join(data, "log", "dev_log.txt")

        # 预训练的中文词向量
        vector_file = os.path.join(data, "vector", "sgns.merge.bigram")
        # 字向量的文件
        char_vector_file = os.path.join(data, "vector", "sgns.char.dim300.iter5")

        # tf.app.flags 是用于接受命令行传来的参数
        return tf.contrib.training.HParams(
            zhidao_train_file=zhidao_train_file,
            zhidao_dev_file=zhidao_dev_file,
            zhidao_test_file=zhidao_test_file,
            search_train_file=search_train_file,
            search_dev_file=search_dev_file,
            search_test_file=search_test_file,
            vector_file=vector_file,
            char_vector_file=char_vector_file,
            log_file=log_file,
            output_dir=output_dir,
            model_dir=model_dir,
            summary_dir=summary_dir,
            vocab_dir=vocab_dir
        )

    def get_default_params(self):
        '''
        超参数
        :return:
        '''
        # 最大的passage数量
        max_p_num = 5
        # 最大的passage长度
        max_p_len = 400
        # question长度
        max_q_len = 60
        # answer长度
        max_a_len = 200
        # 单词的最大字符长度
        max_ch_len = 20
        # 词向量维度
        word_embed_size = 300
        # 字向量维度
        char_embed_size = 300
        # 词典长度
        vocab_size = 1285531
        # dropout
        keep_prob = 0.5
        # ptr_dropout
        ptr_keep_prob = 0.5
        # hidden size
        hidden_size = 48
        # char hidden
        char_hidden = 48
        # attention size
        attn_size = 48
        # batch size
        batch_size = 8
        # epoch
        epoch = 20
        # 优化函数
        opt_arg = {'adadelta': {'learning_rate': 1, 'rho': 0.95, 'epsilon': 1e-6},
                   'adam': {'learning_rate': 1e-3, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8},
                   'gradientdescent': {'learning_rate': 1},
                   'adagrad': {'learning_rate': 1}}
        # adadelta init lr
        init_lr = 0.5
        # 是否使用 cudnn
        use_cudnn = True
        # 全局梯度削减速率
        grad_clip = 5.0
        # 批量保存loss的大小
        period = 1
        # 梯度衰减
        weight_decay = 0.005

        return tf.contrib.training.HParams(
            max_p_num=max_p_num,
            max_p_len=max_p_len,
            max_q_len=max_q_len,
            max_a_len=max_a_len,
            max_ch_len=max_ch_len,
            word_embed_size=word_embed_size,
            char_embed_size=char_embed_size,
            vocab_size=vocab_size,
            keep_prob=keep_prob,
            ptr_keep_prob=ptr_keep_prob,
            hidden_size=hidden_size,
            char_hidden=char_hidden,
            attn_size=attn_size,
            batch_size=batch_size,
            epoch=epoch,
            opt_arg=opt_arg,
            init_lr=init_lr,
            use_cudnn=use_cudnn,
            grad_clip=grad_clip,
            period=period,
            weight_decay=weight_decay
        )

if __name__ == '__main__':
    con = Config()
    print(con.get_default_params().opt_arg['adam']['learning_rate'])