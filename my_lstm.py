#from __future__ import print_function


import numpy as np
import theano
import theano.tensor as T
import theano.sparse
import lasagne

from theano import sparse

import array 
import csv 
import os
import numpy as np 
from scipy.sparse import csr_matrix 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

import pickle
from collections import defaultdict

from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

# All gradients above this will be clipped
GRAD_CLIP = 100

# Number of units in the two hidden (LSTM) layers
N_HIDDEN = 512

# Optimization learning rate
LEARNING_RATE = .01

dataset_path = '.\\test_building_train_data_small_15'
#dataset_path = 'C:\Diploma\with_usefull_features\\test'
SENTENCE_ID = 0
LEX = 3
ID = 4
PARENT = 5
FEATURES = 6

SHIFT = 1
REDUCE = 2
LEFTARC = 3
RIGHTARC = 4

IS_ROOT_FEATURE = 'IsRoot'

TRAIN_PART = 1.0 / 5 * 4 

# read data from dataset
class DataKeeper:
	def __init__(self):
		# self.parent_ids = [] seems to be not used
		self.features = []

		self.features_dictionary = {} # dictionary for features
		self.features_names = []
		self.row = [] # aux container for sparse matrix
		self.col = [] # aux container for sparse matrix
		self.data = [] # aux container for sparse matrix
		self.answers = [] # 0 - no connection, 1 - first is a parent of second, -1 - second is a parent of first

		self.FEAT_NUM = -1 # number of features in dictionary
		self.sentences_number = 0 # number of sentences in input data, == len(input)
		self.input = [] # list of sentences

	'''
		Input is filling here as a seq of feature vectors.
	'''
	def load_data(self):
		# read one by one file from every folder
		# every file has structure: sentence_id, begin_index, lexeme, word, id, parent_id, feature1, feature2, ...
		for foldername in os.listdir(dataset_path):
			print foldername
			path = os.path.join(dataset_path, foldername)
			for filename in os.listdir(path):
				fullname = os.path.join(dataset_path, foldername, filename)
				self.load_file(fullname)
		print 'Input is ready!'

	def get_file_size(self, fullname):
		dataset_len = 0
		with open(fullname, "rU") as data_initial:
			dataset = csv.reader(data_initial, delimiter="\t")
			dataset_len = sum(1 for row in dataset)
		return dataset_len

	def get_root(self, sent_number):
		return [sent_number, '0', 'ROOT', 'ROOT', '-1', '-2', IS_ROOT_FEATURE]

	def load_file(self, fullname):
		# need to know the size of file
		dataset_len = self.get_file_size(fullname)
		with open(fullname, "rU") as data_initial:
			dataset = csv.reader(data_initial, delimiter="\t")
			# initialization
			current_sentence = []
			current_sentence_features = []
			prev_sentence_num = 0

			# for each row (=word with features) in file
			for idx, row in enumerate(dataset):
				# if row is not empty
				if len(row) != 0 and row[LEX] != '':
					# do transitions inside one sentence
					# if token is not last in input
					if idx == dataset_len - 1:
						# add fictive root vertex in the end of sentence.
						# Root's parent is ROOT!!!
						current_sentence.append(row)

						root = self.get_root(prev_sentence_num)
						current_sentence.append(list(root))
						current_sentence_features.append(root[6:])
						
						self.input.append(list(current_sentence))
						self.features.extend(list(current_sentence_features))
						break

					# check if previous token is from the same sentence
					if int(row[SENTENCE_ID]) != prev_sentence_num:
						prev_sentence_num = int(row[SENTENCE_ID])
						stack = [] # reset stack

						# check if the length of sentence is large enough
						if len(current_sentence) >= 5:
							# add fictive root vertex in the end of sentence.
							# Root's parent is ROOT!!!
							root = self.get_root(prev_sentence_num)
							current_sentence.append(list(root))
							current_sentence_features.append(root[6:])

							self.input.append(list(current_sentence))
							self.features.extend(list(current_sentence_features))
							self.sentences_number += 1
							# TODO: add aux root vertex
						current_sentence = []
						current_sentence_features = []

					current_sentence_features.append(row[6:])
					current_sentence.append(row)

	# fill the dictionary of features & set the value of FEAT_NUM
	def fill_dictionary(self):
		counter = 0
		for i, word_features in enumerate(self.features):
			for j, feature in enumerate(word_features):
				feat_num = self.features_dictionary.setdefault(feature, counter)
				if feat_num == counter:
					counter += 1
					self.features_names.append(feature)
		self.FEAT_NUM = len(self.features_names)
		print "number of features in dictionary is ", self.FEAT_NUM

	def get_feat_num_by_name(self, feature):
		return self.features_dictionary[feature]

classes = ['shift', 'reduce', 'left', 'right'] # 1, 2, 3, 4

# Number of units in the two hidden (LSTM) layers
N_HIDDEN = 512

class MaltParser:
	def __init__(self):
		# to initialize, read text with words' features from files
		self.data_keeper = DataKeeper()
		self.data_keeper.load_data()
		self.data_keeper.fill_dictionary()

		self.svm_clf = OneVsRestClassifier(SVC(kernel='linear', verbose=True), n_jobs=-1)
		self.tokens = []
		self.stack = [] # stack for deterministic algorithm
		self.input = self.data_keeper.input
		self.arcs = [] # arc labels
		self.in_verticies = defaultdict(list) # a collection of output edges "vertex_in": vertex from
		self.out_verticies = defaultdict(list) # "vertex from": vertex_in
		self.current_sentence_position = 0 # current position in input
		self.current_word_position = 0 # current position in input
		self.feat_count = len(self.data_keeper.features_dictionary)

		# for training, filled in build_train_samples
		#self.train_samples = sparse.csr_matrix(name='train_samples', dtype='int16')
		self.train_samples = []
		self.train_answers = []

		self.huge_matrix = []

		self.row = []
		self.col = []
		self.data = []

	'''
		Apply predicted actions to the sentence and return the accuracy
	'''
	def left_arc(self):
		word = self.input[self.current_sentence_position][self.current_word_position]
		self.out_verticies[word[ID]].append(self.stack[-1])
		self.in_verticies[self.stack[-1][ID]].append(word)
		self.stack.pop()

	def right_arc(self):
		word = self.input[self.current_sentence_position][self.current_word_position]
		self.out_verticies[self.stack[-1][ID]].append(word)
		self.in_verticies[word[ID]].append(self.stack[-1])
		self.stack.append(word)
		self.current_word_position += 1

	def reduce(self):
		self.stack.pop()

	def shift(self):
		self.stack.append(self.input[self.current_sentence_position][self.current_word_position])
		self.current_word_position += 1

	# add history feature map in sparse matrix
	# TODO: divide features depend on its kind (morph, lex, pos, etc),
	# TODO: add dynamic features
	def add_history_feature_map_for_training(self):
		current_row_num = len(self.train_answers)

		# add features of top
		if (len(self.stack) != 0):
			for feature in self.stack[-1][FEATURES:]:
				feat_num = self.data_keeper.get_feat_num_by_name(feature)
				self.row.append(current_row_num)
				self.col.append(feat_num)
				self.data.append(1)

		# add features of top-1
		if (len(self.stack) > 1):
			for feature in self.stack[-2][FEATURES:]:
				feat_num = self.data_keeper.features_dictionary[feature]
				self.row.append(current_row_num)
				self.col.append(feat_num + self.feat_count)
				self.data.append(1)

		# add features of next
		if len(self.input) != 0:
			sent = self.input[self.current_sentence_position]
			w = sent[self.current_word_position]

			for feature in w[FEATURES:]:
				feat_num = self.data_keeper.features_dictionary[feature]
				self.row.append(current_row_num)
				self.col.append(feat_num + self.feat_count * 2)
				self.data.append(1)

			# add features of next+1
			if (len(self.input[self.current_sentence_position]) > self.current_word_position + 1):
				for feature in self.input[self.current_sentence_position][self.current_word_position + 1][FEATURES:]:
					feat_num = self.data_keeper.features_dictionary[feature]
					self.row.append(current_row_num)
					self.col.append(feat_num + self.feat_count * 3)
					self.data.append(1)

			# add features of next+2
			if (len(self.input[self.current_sentence_position]) > self.current_word_position + 2):
				for feature in self.input[self.current_sentence_position][self.current_word_position + 2][FEATURES:]:
					feat_num = self.data_keeper.features_dictionary[feature]
					self.row.append(current_row_num)
					self.col.append(feat_num + self.feat_count * 4)
					self.data.append(1)

			# add features of next+3
			if (len(self.input[self.current_sentence_position]) > self.current_word_position + 3):
				for feature in self.input[self.current_sentence_position][self.current_word_position + 3][FEATURES:]:
					feat_num = self.data_keeper.features_dictionary[feature]
					self.row.append(current_row_num)
					self.col.append(feat_num + self.feat_count * 5)
					self.data.append(1)

		# add features of head(top)
		if (len(self.stack) != 0):
			top = self.stack[-1]
			if self.in_verticies.has_key(top[ID]):
				for feature in self.in_verticies[top[ID]][0][FEATURES:]:
					feat_num = self.data_keeper.features_dictionary[feature]
					self.row.append(current_row_num)
					self.col.append(feat_num + self.feat_count * 6)
					self.data.append(1)

			# add features of ldep(top)
			if self.out_verticies.has_key(top[ID]):
				for feature in self.out_verticies[top[ID]][0][FEATURES:]:
					feat_num = self.data_keeper.features_dictionary[feature]
					self.row.append(current_row_num)
					self.col.append(feat_num + self.feat_count * 7)
					self.data.append(1)

			# add features of rdep(top)
			if self.out_verticies.has_key(top[ID]):
				for feature in self.out_verticies[top[ID]][-1][FEATURES:]:
					feat_num = self.data_keeper.features_dictionary[feature]
					self.row.append(current_row_num)
					self.col.append(feat_num + self.feat_count * 8)
					self.data.append(1)

		# add features of ldep(next)
		if len(self.input) != 0:
			next = self.input[self.current_sentence_position][-1]
			if next[ID] in self.out_verticies:
				for feature in self.out_verticies[next[ID]][0][FEATURES:]:
					feat_num = self.data_keeper.features_dictionary[feature]
					self.row.append(current_row_num)
					self.col.append(feat_num + self.feat_count * 9)
					self.data.append(1)

		# fake feature, in order to keep size constant
		self.row.append(current_row_num)
		self.col.append(self.feat_count * 10)
		self.data.append(1)

	# erase self.data, self.row, self.col, self.train_answers
	def erase_containers(self):
		self.data = []
		self.row = []
		self.col = []
		self.train_answers = []

	# some auxillary function
	def build_train_sparse_matrix(self):
		rdata = np.asarray(self.data)
		rrow = np.asarray(self.row)
		rcol = np.asarray(self.col)
		self.train_samples = csr_matrix( ( rdata,(rrow,rcol) ) )
		#self.train_samples = sparse.CSR(rdata, rrow, rcol, (len(self.train_answers), self.feat_count * 10))

		#rdata, rrow, rcol = sparse.csm_properties(self.train_samples)

	def build_train_samples(self, begin, end):
		print 'building train samples...'
		self.erase_containers() # init containers

		# will use first half of data in range from begin index to end index
		print 'size of training data is ', end - begin, 'sentences'

		self.current_sentence_position = begin
		for sentence in self.input[begin:end]:
			self.out_verticies.clear()
			self.in_verticies.clear()
			self.current_word_position = 0
			#print self.stack #DEBUG
			if self.current_sentence_position % 5000 == 0:
				print self.current_sentence_position

			self.stack = [] # reset stack
			# do transitions inside one sentence
			# remember the last vertex is ROOT
			for idx, word_with_features in enumerate(sentence):
				self.current_word_position = idx
				# until right or shift
				# priority is left > right > reduce > shift
				while 1:
					# create history-based feature model instance
					# and add it to train data
					self.add_history_feature_map_for_training()
					if len(self.stack) == 0:
						self.train_answers.append(SHIFT)
						self.shift()						
						break
					if self.stack[-1][PARENT] == word_with_features[ID]:
						self.train_answers.append(LEFTARC)
						self.left_arc()
						continue
					if self.stack[-1][ID] == word_with_features[PARENT]:
						self.train_answers.append(RIGHTARC)
						self.right_arc()
						break
					for stack_word in self.stack:
						has_stack_dependencies = False
						if word_with_features[ID] == stack_word[PARENT]:
							has_stack_dependencies = True
							break
					if has_stack_dependencies:
						self.train_answers.append(REDUCE)
						self.reduce()
						continue
					for right_word in sentence[idx+1:]:
						has_right_dependencies = False
						if right_word[ID] == word_with_features[PARENT] or right_word[PARENT] == word_with_features[ID]:
							has_right_dependencies = True
							break
					if has_right_dependencies:
						self.train_answers.append(SHIFT)
						self.shift()
						break
					else:
						self.train_answers.append(REDUCE)
						self.reduce()
			self.current_sentence_position += 1

		self.build_train_sparse_matrix()
		print 'train samples were built!'

	def build_test_samples(self, begin, end):
		print 'building test samples...'
		self.erase_containers() # init containers

		print 'size of test data is', end - begin
		self.current_sentence_position = begin
		for sentence in self.input[begin:end]:
			self.stack = [] # reset stack
			for idx, word_with_features in enumerate(sentence):
				self.current_word_position = idx
				self.add_history_feature_map_for_training()
			self.current_sentence_position += 1

		self.build_train_sparse_matrix()
		print 'test samples were built!'

	# do experiment and count the score
	def execute_experiment(self):
		print 'experiment starting...'
		print "number of train samples: ", self.train_samples.shape

		train_data_size = int(len(self.input) * TRAIN_PART) # number of _sentences_

		# work with network
		print("Building network ...")

   		help(lasagne.layers.LSTMLayer)

   		#input_var = theano.sparse.csr_matrix(name='inputs', dtype='int16')
	 	input_var = T.tensor3('inputs')
	 	target_var = T.matrix('targets')
	 	train_matrix = self.train_samples.todense()
	 	print train_matrix.shape[0], train_matrix.shape[1], 'SHAPEEE'

	 	#self.train_answers = T.ivector('target_output')

	 	# (batch size, SEQ_LENGTH, num_features)
	 	l_in = lasagne.layers.InputLayer(shape=(1, 1, train_matrix.shape[1]), input_var=input_var)
	 	l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.3)


		#W = np.arange(3*train_matrix.shape[1]).reshape((3, 1, train_matrix.shape[1])).astype('float32')
		#l_resized = lasagne.layers.ReshapeLayer(l_in_drop, shape=(-1, 1))

	 	# We clip the gradients at GRAD_CLIP to prevent the problem of exploding gradients. 
	 	l_forward_1 = lasagne.layers.LSTMLayer(l_in_drop, 4, grad_clipping=GRAD_CLIP,nonlinearity=lasagne.nonlinearities.tanh)

		#l_resized = lasagne.layers.ReshapeLayer(l_forward_1, shape=(-1, 1))
	 	'''l_forward_2 = lasagne.layers.LSTMLayer(
	 	 	l_forward_1, N_HIDDEN, grad_clipping=GRAD_CLIP,
	 	 	nonlinearity=lasagne.nonlinearities.tanh)'''

	 	# The l_forward layer creates an output of dimension (batch_size, SEQ_LENGTH, N_HIDDEN)
	 	# Since we are only interested in the final prediction, we isolate that quantity and feed it to the next layer.
	 	# The output of the sliced layer will then be of size (batch_size, N_HIDDEN)
	 	#l_forward_slice = lasagne.layers.SliceLayer(l_forward_1, -1, 1)

	 	# The sliced output is then passed through the softmax nonlinearity to create probability distribution of the prediction
	 	# The output of this stage is (batch_size, vocab_size)
	 	l_out = lasagne.layers.DenseLayer(l_forward_1, num_units=4, W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)

	 	# lasagne.layers.get_output produces a variable for the output of the net
	 	network_output = lasagne.layers.get_output(l_out)

	 	# The loss function is calculated as the mean of the (categorical) cross-entropy between the prediction and target.
	 	cost = T.nnet.categorical_crossentropy(network_output,target_var).mean()

	 	# Retrieve all parameters from the network
	 	all_params = lasagne.layers.get_all_params(l_out,trainable=True)

	 	# Compute AdaGrad updates for training
	 	print("Computing updates ...")
	 	updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)

	 	# Theano functions for training and computing cost
	 	print("Compiling functions ...")
	 	train = theano.function([input_var, target_var], cost, updates=updates, allow_input_downcast=True)
	 	compute_cost = theano.function([input_var, target_var], cost, allow_input_downcast=True)

	 	# In order to generate text from the network, we need the probability distribution of the next character given
	 	# the state of the network and the input (a seed).
	 	# In order to produce the probability distribution of the prediction, we compile a function called probs. 
	 	
	 	probs = theano.function([input_var],network_output,allow_input_downcast=True)
	 	# end working with network

		# train model
		print("Training ...")

	 	for idx, row in enumerate(train_matrix):
			if idx % 100 == 0:
				print "training", idx
			inputs = np.array([row])
			#targets = np.array(self.train_answers)
			target = self.train_answers[idx]
			if target == 1:
				targets = [[1, 0, 0, 0]]
			elif target == 2:
				targets = [[0, 1, 0, 0]]
			elif target == 3:
				targets = [[0, 0, 1, 0]]
			elif target == 4:
				targets = [[0, 0, 0, 1]]

			#print targets, targets.shape
	 		avg_cost = train(inputs, targets)
		
		# test model
		print 'start testing...'
		# For each sentence predict consequence of actions,
		# parse it using this actions and evaluate result
		general_score = 0

		self.current_sentence_position = train_data_size #!!!!!!!!!
		# for each sentence in test data
		for sentence in self.input[train_data_size:-2]:
		#for sentence in self.input[:train_data_size]:
			for word in sentence:
				print word[LEX], word[ID], '!', word[PARENT]

			self.current_word_position = 0
			self.stack = [] # reset stack
			self.in_verticies.clear()
			self.out_verticies.clear()
			score = 0 # count matches in the sentence
			# for each word in the sentence
			for idx, word_with_features in enumerate(sentence):
				while 1:
					self.erase_containers()
					self.add_history_feature_map_for_training()
					self.build_train_sparse_matrix()
					# +1 for starting with 1, not 0
					next_action = np.argmax(probs(np.array([self.train_samples.todense()]))) + 1
					print next_action
					#-------------------------------------------------------------------

					if len(self.stack) == 0:
						self.shift()
						break

					word = self.input[self.current_sentence_position][self.current_word_position]
					if next_action == LEFTARC:
						if word[ID] == self.stack[-1][PARENT]:
							#print '----------------------yes'
							score += 1							
						self.left_arc()
						continue

					elif next_action == RIGHTARC:
						#print 'exp right'
						if word[PARENT] == self.stack[-1][ID]:
							#print '----------------------yes'
							score += 1
						self.right_arc()
						break

					elif next_action == REDUCE:
						self.reduce()
						#print 'exp reduce'
						continue

					elif next_action == SHIFT:
						self.shift()
						#print 'exp shift'
						break

			self.current_sentence_position += 1

			# test and count the score
			#print 'sent ', self.current_sentence_position, 'score', score
			general_score += ( score * 1.0 / (len(sentence) - 1) )
		print 'end testing...'
		print 'general score ', general_score
		print '!!! accuracy is ', general_score * 1.0 / len(self.input[train_data_size:-2])

	def dump_clf_parameters(self):
		filename = "dyn_feat6629.pkl"
		with open(filename, "wb") as f:
			s = pickle.dump(self.svm_clf, f, protocol=2)

if __name__ == "__main__":
	#input = [[0, 'Masha', 1], [1, 'washes', -1], [2, 'dishes', 1], [3, 'fast', 1]]
	#input = [[0, 'Beautiful', 1], [1, 'day', -1]]
	mparser = MaltParser()
	train_data_size = int(len(mparser.input) * TRAIN_PART) # number of _sentences_
	mparser.build_train_samples(0, train_data_size)
	mparser.execute_experiment() # some parameters for learning&testing?
	#mparser.train()
	#mparser.cross_validation()


