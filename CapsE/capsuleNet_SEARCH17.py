import tensorflow as tf

from capsuleLayer import CapsLayer
import math

epsilon = 1e-9

class CapsE(object):
    def __init__(self, sequence_length, embedding_size, num_filters, iter_routing, batch_size=256,
                 num_outputs_secondCaps=1, vec_len_secondCaps=10, initialization=[], filter_size=1, useConstantInit=False):
        # Placeholders for input, output
        self.input_x = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [batch_size, 1], name="input_y")
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size
        self.iter_routing = iter_routing
        self.num_outputs_secondCaps = num_outputs_secondCaps
        self.vec_len_secondCaps = vec_len_secondCaps
        self.batch_size = batch_size
        self.useConstantInit = useConstantInit
        # Embedding layer
        with tf.name_scope("embedding"):
            self.W_query = tf.get_variable(name="W_query", initializer=initialization[0], trainable=False)
            self.W_user = tf.get_variable(name="W_user", initializer=initialization[1])
            self.W_doc = tf.get_variable(name="W_doc", initializer=initialization[2], trainable=False)

        self.embedded_query = tf.nn.embedding_lookup(self.W_query, self.input_x[:, 0])
        self.embedded_user = tf.nn.embedding_lookup(self.W_user, self.input_x[:, 1])
        self.embedded_doc = tf.nn.embedding_lookup(self.W_doc, self.input_x[:, 2])

        self.embedded_query = tf.reshape(self.embedded_query, [batch_size, 1, self.embedding_size])
        self.embedded_user = tf.reshape(self.embedded_user, [batch_size, 1, self.embedding_size])
        self.embedded_doc = tf.reshape(self.embedded_doc, [batch_size, 1, self.embedding_size])

        self.embedded_chars = tf.concat([self.embedded_query, self.embedded_user, self.embedded_doc], axis=1)
        self.X = tf.expand_dims(self.embedded_chars, -1)

        self.build_arch()
        self.loss()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=500)

        tf.logging.info('Seting up the main structure')

    def build_arch(self):
        #The first capsule layer
        with tf.variable_scope('FirstCaps_layer'):
            self.firstCaps = CapsLayer(num_outputs_secondCaps=self.num_outputs_secondCaps, vec_len_secondCaps=self.vec_len_secondCaps,
                                    with_routing=False, layer_type='CONV', embedding_size=self.embedding_size,
                                    batch_size=self.batch_size, iter_routing=self.iter_routing,
                                    useConstantInit=self.useConstantInit, filter_size=self.filter_size,
                                    num_filters=self.num_filters, sequence_length=self.sequence_length)

            self.caps1 = self.firstCaps(self.X, kernel_size=1, stride=1)
        #The second capsule layer
        with tf.variable_scope('SecondCaps_layer'):
            self.secondCaps = CapsLayer(num_outputs_secondCaps=self.num_outputs_secondCaps, vec_len_secondCaps=self.vec_len_secondCaps,
                                    with_routing=True, layer_type='FC',
                                    batch_size=self.batch_size, iter_routing=self.iter_routing,
                                    embedding_size=self.embedding_size, useConstantInit=self.useConstantInit, filter_size=self.filter_size,
                                    num_filters=self.num_filters, sequence_length=self.sequence_length)
            self.caps2 = self.secondCaps(self.caps1)

        self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keepdims=True) + epsilon)

    def loss(self):
        self.scores = tf.reshape(self.v_length, [self.batch_size, 1])
        self.predictions = tf.nn.sigmoid(self.scores)
        print("Using square softplus loss")
        losses = tf.square(tf.nn.softplus(self.scores * self.input_y))
        self.total_loss = tf.reduce_mean(losses)
