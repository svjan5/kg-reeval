import tensorflow as tf

from capsuleLayer import CapsLayer
import math

epsilon = 1e-9

class CapsE(object):
	def __init__(self, sequence_length, embedding_size, num_filters, vocab_size, iter_routing, batch_size=256,
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
			if initialization == []:
				self.W = tf.Variable(
					tf.random_uniform([vocab_size, embedding_size], -math.sqrt(1.0 / embedding_size),
									  math.sqrt(1.0 / embedding_size), seed=1234), name="W")
			else:
				self.W = tf.get_variable(name="W2", initializer=initialization)

		self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
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
