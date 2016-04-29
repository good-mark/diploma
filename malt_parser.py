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

# ===================================================
# Constant declaration
# ===================================================
#dataset_path = '..\\test_building_train_data_small_15'
dataset_path = '..\\fix_test'
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

# ===================================================
# Class for reading data from dataset, preparing it for training.
# ===================================================
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
			is_cur_sent_content_root = False
			current_sentence_features = []
			prev_sentence_num = 0

			# for each row (=word with features) in file
			for idx, row in enumerate(dataset):
				# if row is not empty
				if len(row) != 0 and len(row[LEX]) != 0:
					# do transitions inside one sentence
					
					# if token is last one in input, finish data reading
					if idx == (dataset_len - 1):
						# filtering the sentences with no root vertex (is it possible? seems to be)
						if row[PARENT] == '-1':
							is_cur_sent_content_root = True
						if is_cur_sent_content_root:
							# add fictive root vertex in the end of sentence.
							# Root's parent is ROOT!!!
							current_sentence.append(row)

							root = self.get_root(prev_sentence_num)
							current_sentence.append(list(root))
							current_sentence_features.append(root[6:])

							self.input.append(list(current_sentence))
							self.features.extend(list(current_sentence_features))
							break

					# if token is not last one in input,
					# check if previous token is from the same sentence
					if int(row[SENTENCE_ID]) != prev_sentence_num:
						prev_sentence_num = int(row[SENTENCE_ID])
						stack = [] # reset stack

						# check if the length of sentence is large enough
						if len(current_sentence) >= 5 and is_cur_sent_content_root:
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
						is_cur_sent_content_root = False

					current_sentence_features.append(row[6:])
					current_sentence.append(row)					
					if row[PARENT] == '-1':
						is_cur_sent_content_root = True

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

# ===================================================
# Class implementing MaltParser - deterministic parser.
# Reconstucts the sequence of parser's actions for the particular parse tree
# Trains the One-Vs-Rest linear SVM classifier using history feature-based approach.
# Predicts the next parser action for the particular stack-input-partially built tree state.
# Evaluates the attachment score (it is just an accuracy for unlabeled case) 
# counting the percentage of vertecies for which parent vertex is assigned correctly. 
# ===================================================
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
		self.train_samples = []
		self.train_answers = []

		self.row = []
		self.col = []
		self.data = []

	# ===================================================
	# Four main parser actions.
	# Each of them changes the state of input, stack and building tree.
	# ===================================================
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


	# ===================================================
	# A range of auxillary functions adding different kinds of features
	# filling the history feature-based model.
	# ===================================================
	def add_one_feature(self, feature, current_row, order):
		feat_num = self.data_keeper.get_feat_num_by_name[feature]
		self.row.append(current_row)
		self.col.append(feat_num + self.feat_count * order)
		self.data.append(1)

	def add_top_features(self, order, current_row):
		if (len(self.stack) != 0):
			for feature in self.stack[-1][FEATURES:]:
				self.add_one_feature(feature, current_row, order)

	def add_top_1_features(self, order, current_row):
		if (len(self.stack) > 1):
			for feature in self.stack[-2][FEATURES:]:
				self.add_one_feature(feature, current_row, order)

	def add_next_features(self, order, current_row):
		if len(self.input) != 0:
			sent = self.input[self.current_sentence_position]
			w = sent[self.current_word_position]
			for feature in w[FEATURES:]:
				self.add_one_feature(feature, current_row, order)

	def add_next_1_features(self, order, current_row):
		if len(self.input) != 0:
			sent = self.input[self.current_sentence_position]
			if (len(sent) > self.current_word_position + 1):
				for feature in sent[self.current_word_position + 1][FEATURES:]:
					self.add_one_feature(feature, current_row, order)

	def add_next_2_features(self, order, current_row):
		if len(self.input) != 0:
			sent = self.input[self.current_sentence_position]
			if (len(sent) > self.current_word_position + 2):
				for feature in sent[self.current_word_position + 2][FEATURES:]:
					self.add_one_feature(feature, current_row, order)

	def add_next_3_features(self, order, current_row):
		if len(self.input) != 0:
			sent = self.input[self.current_sentence_position]
			if (len(sent) > self.current_word_position + 3):
				for feature in sent[self.current_word_position + 3][FEATURES:]:
					self.add_one_feature(feature, current_row, order)

	def add_head_top_features(self, order, current_row):
		if (len(self.stack) != 0):
			top = self.stack[-1]
			if self.in_verticies.has_key(top[ID]):
				for feature in self.in_verticies[top[ID]][0][FEATURES:]:
					self.add_one_feature(feature, current_row, order)

	def add_ldep_top_features(self, order, current_row):
		if (len(self.stack) != 0):
			top = self.stack[-1]
			if self.out_verticies.has_key(top[ID]):
				for feature in self.out_verticies[top[ID]][0][FEATURES:]:
					self.add_one_feature(feature, current_row, order)

	def add_rdep_top_features(self, order, current_row):
		if (len(self.stack) != 0):
			top = self.stack[-1]
			if self.out_verticies.has_key(top[ID]):
				for feature in self.out_verticies[top[ID]][-1][FEATURES:]:
					self.add_one_feature(feature, current_row, order)


	def add_fake_feature(self, order, current_row):
		self.row.append(current_row)
		self.col.append(self.feat_count * 10)
		self.data.append(1)

	# add history feature map in sparse matrix
	# TODO: divide features depend on its kind (morph, lex, pos, etc),
	def add_history_feature_map_for_training(self):
		current_row = len(self.train_answers)
		self.add_top_features(0, current_row) # add features of top		
		self.add_top_1_features(1, current_row) # add features of top-1

		self.add_next_features(2, current_row) # add features of next
		self.add_next_1_features(3, current_row) # add features of next+1
		self.add_next_2_features(4, current_row) # add features of next+2
		self.add_next_3_features(5, current_row) # add features of next+3
		
		self.add_head_top_features(6, current_row) # add features of head(top)
		self.add_ldep_top_features(7, current_row) # add features of ldep(top)
		self.add_rdep_top_features(8, current_row) # add features of rdep(top)

		# add features of ldep(next)
		if len(self.input) != 0:
			next = self.input[self.current_sentence_position][-1]
			if next[ID] in self.out_verticies:
				for feature in self.out_verticies[next[ID]][0][FEATURES:]:
					feat_num = self.data_keeper.features_dictionary[feature]
					self.row.append(current_row)
					self.col.append(feat_num + self.feat_count * 9)
					self.data.append(1)

		# fake feature, in order to keep size constant
		self.add_fake_feature(10, current_row)

	# erase self.data, self.row, self.col, self.train_answers
	def erase_containers(self):
		self.data = []
		self.row = []
		self.col = []
		self.train_answers = []

	def erase_state(self):
		self.out_verticies.clear()
		self.in_verticies.clear()
		self.current_word_position = 0
		self.stack = [] # reset stack

	# some auxillary function
	def build_train_sparse_matrix(self):
		rdata = np.asarray(self.data)
		rrow = np.asarray(self.row)
		rcol = np.asarray(self.col)
		self.train_samples = csr_matrix( ( rdata,(rrow,rcol) ) )

	def build_train_samples(self, begin, end):
		print 'building train samples...'
		self.erase_containers() # init containers

		# will use first half of data in range from begin index to end index
		print 'size of training data is ', end - begin, 'sentences'

		self.current_sentence_position = begin
		for sentence in self.input[begin:end]:
			self.erase_state()
			#print self.stack #DEBUG
			if self.current_sentence_position % 5000 == 0:
				print self.current_sentence_position

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

	# For each sentence predict consequence of actions,
	# parse it using this actions and evaluate result
	def test(self, begin, end):
		general_score = 0
		self.current_sentence_position = begin #!!!!!!!!!

		# for each sentence in test data
		for sentence in self.input[begin:end]:
			for word in sentence:
				print word[LEX], word[ID], '!', word[PARENT]

			self.erase_state()
			score = 0 # count matches in the sentence
			# for each word in the sentence
			for idx, word_with_features in enumerate(sentence):
				while 1:
					self.erase_containers()
					self.add_history_feature_map_for_training()
					self.build_train_sparse_matrix()
					next_action = self.svm_clf.predict(self.train_samples)

					# If stack is empty, we can only do shift to fill it.
					if len(self.stack) == 0:
						self.shift()
						break

					word = self.input[self.current_sentence_position][self.current_word_position]
					if next_action == LEFTARC:
						if word[ID] == self.stack[-1][PARENT]:
							score += 1	
						else:
							print '--------------------------------NO'
						self.left_arc()
						continue

					elif next_action == RIGHTARC:
						#print 'exp right' #DEBUG
						if word[PARENT] == self.stack[-1][ID]:
							score += 1
						else:
							print '--------------------------------NO'
						self.right_arc()
						break

					elif next_action == REDUCE:
						self.reduce()
						#print 'exp reduce' #DEBUG
						continue

					elif next_action == SHIFT:
						self.shift()
						#print 'exp shift' #DEBUG
						break

			self.current_sentence_position += 1

			# test and count the score
			#print 'sent ', self.current_sentence_position, 'score', score
			general_score += ( score * 1.0 / (len(sentence) - 1) )
		print 'end testing...'
		print 'general score ', general_score
		print '!!! accuracy is ', general_score * 1.0 / len(self.input[begin:end])

	# do experiment and count the score
	def execute_experiment(self, train_part):
		print 'Experiment starting...'
		train_data_size = int(len(self.input) * train_part) # number of _sentences_
		self.build_train_samples(0, train_data_size)
		print "Number of train samples: ", self.train_samples.shape

		# train model
		print 'start training...'
		self.svm_clf.fit(self.train_samples, self.train_answers)
		print 'end training...'
		self.dump_clf_parameters()

		# test model
		print 'start testing...'
		self.test(train_data_size, -2)
		

	def dump_clf_parameters(self):
		filename = "malt_parser_model.pkl"
		with open(filename, "wb") as f:
			s = pickle.dump(self.svm_clf, f, protocol=2)

if __name__ == "__main__":
	mparser = MaltParser()
	''' You can change the part to be training data	'''
	train_part = 1.0 / 5 * 4 
	mparser.execute_experiment(train_part) # some parameters for learning&testing?