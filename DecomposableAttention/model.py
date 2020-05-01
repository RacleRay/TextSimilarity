# -*- encoding:utf-8 -*-

import tensorflow as tf
import numpy as np


class DecompAtt:
    def __init__(self, config):
        self.config = config
        # 输入
        self.add_placeholders()
        # [batch_size, sequence_size, embed_size]
        q_embed, a_embed = self.add_embeddings()
        q_encode, a_encode = self.context_encoding(q_embed, a_embed)  # 上下文编码
        q_attend, a_attend = self.attend(q_encode, a_encode)  # attention层
        q_comp, a_comp = self.compare(
            q_encode, a_encode, q_attend, a_attend)  # compare
        pred = self.aggregate(q_comp, a_comp)  # aggregate层

        # 预测概率分布与损失
        self.y_hat, self.total_loss = self.add_loss_op(pred)
        # 训练节点
        self.train_op = self.add_train_op(self.total_loss)

    def add_placeholders(self):
        # Q
        self.q = tf.placeholder(tf.int32,
                                shape=[None, self.config.max_q_length],
                                name='Question')
        # A
        self.a = tf.placeholder(tf.int32,
                                shape=[None, self.config.max_a_length],
                                name='Ans')
        self.y = tf.placeholder(tf.int32, shape=[None, ], name='label')
        # drop_out
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.batch_size = tf.shape(self.q)[0]

    def add_embeddings(self):
        with tf.variable_scope('embedding'):
            if self.config.embeddings is not None:
                embeddings = tf.Variable(self.config.embeddings,
                    name="embeddings", trainable=False)
            else:
                embeddings = tf.get_variable('embeddings',
                    shape=[self.config.vocab_size, self.config.embedding_size],
                    initializer=tf.uniform_unit_scaling_initializer())
            q_embed = tf.nn.embedding_lookup(embeddings, self.q)
            a_embed = tf.nn.embedding_lookup(embeddings, self.a)
            return q_embed, a_embed

     def context_encoding(self, q, a):
        """
        q: [batch_size, q_length, embedding_dim]
        a: [batch_size, a_length, embedding_dim]
        """
        with tf.variable_scope('context_encoding') as scope:
            q = tf.nn.dropout(q, keep_prob=self.keep_prob)
            a = tf.nn.dropout(a, keep_prob=self.keep_prob)
            q_encode = self.rnn_layer(q)
            tf.get_variable_scope().reuse_variables()
            a_encode = self.rnn_layer(a)
        return q_encode, a_encode

    def rnn_layer(self, h):
        sequence_length = h.get_shape()[1]
        # (batch_size, time_step, embed_size) -> (time_step, batch_size, embed_size)
        inputs = tf.transpose(h, [1, 0, 2])

        if self.config.cell_type == 'lstm':
            birnn_fw, birnn_bw = self.bi_lstm(self.config.rnn_size, self.config.layer_size, self.config.keep_prob)
        else:
            birnn_fw, birnn_bw = self.bi_gru(self.config.rnn_size, self.config.layer_size, self.config.keep_prob)
        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(birnn_fw, birnn_bw, inputs, dtype=tf.float32)
        # (time_step, batch_size, 2*rnn_size) -> (batch_size, time_step, 2*rnn_size)
        output = tf.transpose(outputs, (1, 0, 2))
        return output

    def bi_lstm(self, rnn_size, layer_size, keep_prob):
        # forward rnn
        with tf.name_scope('fw_rnn'), tf.variable_scope('fw_rnn'):
            lstm_fw_cell_list = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(layer_size)]
            lstm_fw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_fw_cell_list), output_keep_prob=keep_prob)

        # backward rnn
        with tf.name_scope('bw_rnn'), tf.variable_scope('bw_rnn'):
            lstm_bw_cell_list = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(layer_size)]
            lstm_bw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_fw_cell_list), output_keep_prob=keep_prob)

        return lstm_fw_cell_m, lstm_bw_cell_m

    def bi_gru(self, rnn_size, layer_size, keep_prob):
        # forward rnn
        with tf.name_scope('fw_rnn'), tf.variable_scope('fw_rnn'):
            gru_fw_cell_list = [tf.contrib.rnn.GRUCell(rnn_size) for _ in range(layer_size)]
            gru_fw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(gru_fw_cell_list), output_keep_prob=keep_prob)

        # backward rnn
        with tf.name_scope('bw_rnn'), tf.variable_scope('bw_rnn'):
            gru_bw_cell_list = [tf.contrib.rnn.GRUCell(rnn_size) for _ in range(layer_size)]
            gru_bw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(gru_bw_cell_list), output_keep_prob=keep_prob)

        return gru_fw_cell_m, gru_bw_cell_m

    def attend(self, q, a):
        """
        decomposable attention
        q: [batch_size, q_length, represent_dim]
        a: [batch_size, a_length, represent_dim]
        """
        q = tf.nn.dropout(q, keep_prob=self.keep_prob)
        a = tf.nn.dropout(a, keep_prob=self.keep_prob)
        q_map = tf.layers.dense(q, 128, activation=tf.nn.relu, name='embed_map')
        a_map = tf.layers.dense(a, 128, activation=tf.nn.relu, name='embed_map', reuse=True)
        # [batch_size, q_length, a_length]
        att_inner_product = tf.matmul(q_map, tf.transpose(a_map, [0, 2, 1]))
        # [batch_size, a_length, q_length]
        q_weights = tf.nn.softmax(
                        tf.transpose(att_inner_product, (0, 2, 1)), dim=-1)
        # [batch_size, q_length, a_length]
        a_weights = tf.nn.softmax(att_inner_product, dim=-1)

        output_a = tf.matmul(q_weights, q)  # [batch_size, a_length, represent_dim]
        output_q = tf.matmul(a_weights, a)  # [batch_size, q_length, represent_dim]

        return output_q, output_a

    def compare(self, q, a, q_att, a_att):
        """
        q: [batch_size, q_length, represent_dim]
        a: [batch_size, a_length, represent_dim]
        q_att: [batch_size, q_length, represent_dim]
        a_att: [batch_size, a_length, represent_dim]
        """
        q_combine = tf.concat([q, q_att], axis=-1)
        a_combine = tf.concat([a, a_att], axis=-1)
        q_combine = tf.nn.dropout(q_combine, keep_prob=self.keep_prob)
        a_combine = tf.nn.dropout(a_combine, keep_prob=self.keep_prob)
        q_map = self.mlp(q_combine, self.config.hidden_size, 2, 'embed_compare')
        a_map = self.mlp(a_combine, self.config.hidden_size, 2, 'embed_compare', reuse=True)
        # [batch_size, a_length, hidden_size]
        return q_map, a_map

    def mlp(self, inputs, size, layer_num, name, reuse=None):
        """
        inputs: 上层输入
        size: 神经元大小
        layer_num: 神经网络层数
        name: mlp的名称
        reuse: 是否复用层
        """
        tmp = inputs
        for i in range(layer_num):
            tmp = tf.layers.dense(tmp, size,
                                  activation=tf.nn.relu,
                                  name=name + '_{}'.format(i),
                                  reuse=reuse)
        return tmp

    def aggregate(self, q, a):
        """
        q: [batch_size, q_length, hidden_size]
        a: [batch_size, a_length, hidden_size]
        """
        # [batch_size, represent_dim]
        q_sum = tf.reduce_sum(q, 1)
        a_sum = tf.reduce_sum(a, 1)
        q_sum = tf.nn.dropout(q_sum, keep_prob=self.keep_prob)
        a_sum = tf.nn.dropout(a_sum, keep_prob=self.keep_prob)

        q_a_rep = tf.concat([q_sum, a_sum], axis=-1)
        pred = self.mlp(q_a_rep, self.config.output_size, 2, 'embed_aggregate')
        pred = tf.layers.dense(pred, 2, activation=None, name='prediction')
        return pred

    def add_loss_op(self, pred):
        # [batch_size, 2]
        y_hat = tf.nn.softmax(pred, dim=-1)
        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(self.y, pred))
        tf.add_to_collection('total_loss', loss)
        total_loss = tf.add_n(tf.get_collection('total_loss'))
        return y_hat, total_loss

    def add_train_op(self, loss):
        with tf.name_scope('train_op'):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            opt = tf.train.AdamOptimizer(self.config.lr)
            train_variables = tf.trainable_variables()
            gradients, variables = zip(*opt.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.grad_clip)
            train_op = opt.apply_gradients(zip(gradients, variables), global_step=self.global_step)
        return train_op


class DecompAttConfig(object):
    def __init__(self, vocab_size, embeddings=None):
        # 输入问题(句子)长度
        self.max_q_length = 200
        # 输入答案长度
        self.max_a_length = 200
        # 循环数
        self.num_epochs = 100
        # batch大小
        self.batch_size = 128
        # 词表大小
        self.vocab_size = vocab_size
        self.embeddings = embeddings
        self.embedding_size = 100
        if self.embeddings is not None:
            self.embedding_size = embeddings.shape[1]
        # RNN单元
        self.cell_type = 'GRU'
        self.rnn_size = 128
        self.layer_size = 1
        # 隐层大小
        self.hidden_size = 128
        self.output_size = 128
        # keep_prob=1-dropout
        self.keep_prob = 0.6
        # 学习率
        self.lr = 0.0003
        self.grad_clip = 5.

        self.cf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.cf.gpu_options.per_process_gpu_memory_fraction = 0.2