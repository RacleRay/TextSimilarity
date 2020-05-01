# -*- encoding:utf-8 -*-

import tensorflow as tf


class QACNNConfig(object):
    def __init__(self, vocab_size, embeddings=None):
        self.max_q_length = 200  # 输入问题(句子)长度
        self.max_a_length = 200  # 输入答案长度

        self.num_epochs = 100
        self.batch_size = 128

        self.vocab_size = vocab_size  # 词表大小
        self.embeddings = embeddings
        self.embedding_size = 100  # 词向量大小
        if self.embeddings is not None:
            self.embedding_size = embeddings.shape[1]

        # 不同类型的filter,相当于1-gram,2-gram,3-gram和5-gram
        self.filter_sizes = [1, 2, 3, 5, 7, 9]
        self.hidden_size = 128  # 隐层大小
        self.num_filters = 128  # 每种filter的数量
        self.l2_reg_lambda = 0.
        self.keep_prob = 0.5

        self.lr = 0.001  # 学习率
        self.m = 0.5  # margin: for loss

        self.cf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.cf.gpu_options.per_process_gpu_memory_fraction = 0.2