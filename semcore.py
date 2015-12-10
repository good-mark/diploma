# coding: utf-8

# Ниже приведен один из вариантов организации модуля загрузки датасета
# Основными функциями предположительно станут loadDataset и convert
# Можно писать прямо поверх него, нужно реализовать:
#  - Core.save
#  - Core.load
#  - настоящую загрузку из большого xml (убрать заглушку в loadSentence)
#  - saveDataset
#
#  + Core, возможно, нужно будет расширять новыми методами

import xml.etree.ElementTree as et
import os
import csv
import itertools
import operator # for sorting

#import sys, os
#sys.stdout = open("output.txt", 'w')
###############################################################
# Вспомогательная мелочь

CORE = 7

class IDGEN:
	_lastId = 0 # счетчик уникальных id
	
def getNewId():
	"""Генерирует новый уникальный числовой id"""
	IDGEN._lastId += 1
	return IDGEN._lastId
	
###############################################################
# Core - тот самый 'компактный' класс с данными
class Core:
	"""Ядро составляющей в дереве разбора"""
	
	# В конструкторе описана примерная структура класса
	def __init__(self):
		"""Инициализировать объект с новым уникальным id"""
		# уникальный для датасета id ядра
		self.id = getNewId()
		
		# индекс начала ядра во всей составляющей
		self.begin_index = -1

		# номер предложения, в которое попадает ядро
		self.sentence_id = -1

		# текст ядра
		self.text = ''
		# текст всего поддерева
		self.constituent_text = ''

		# id, указанный в xml
		self.original_id = 0

		# родитель в дереве
		self.parent = None

		# набор признаков в самом простом для реализации виде
		self.features = dict()

		self.lexical_variants = dict()
		self.semantic_variants = dict()
		self.lexemes = dict()
		self.morphological = dict()
		self.class_for_lexeme = dict()
		self.idiomatic = dict()
		self.universal_syntactic = dict()
		self.predefined = dict()

	# Collocation?
	def get_norm_core(self):
		return self.text.replace(' ', '_');

	# Сохранение/загрузка в выбранный компактный формат
	# Возможно, нужно поменять сигнатуру этих методов - в зависимости от формата
	
	def save(self):
		"""Сохраняет ядро в компактном формате и возвращает объект"""
		'''with open('my.csv', 'a') as csvfile:
			#w = csv.writer(csvfile)
			row = [self.constituent_text.encode('utf-8'), self.text.encode('utf-8'), str(self.id), str(self.parent)]
			for list_values in self.features.values():
				for value in list_values:
					row.append(value.encode('utf-8'))
			csvfile.writerow(row)
		return self'''
		prefix_row = [str(self.sentence_id), str(self.begin_index), self.constituent_text.encode('utf-8'), self.text.encode('utf-8'), str(self.id), str(self.parent), ""]
		prefix_list = "\t".join(prefix_row)
		lexical_variants_row = []
		semantic_variants_row = []
		lexemes_row = []
		morphological_row = []
		class_for_lexeme_row = []
		idiomatic_row = []
		universal_syntactic_row = []
		predefined_row = []

		# Записываем в формате: ИмяФичи1 ЗначениеФичи1 ИмяФичи2 ЗначениеФичи2 ИмяФичи3 ЗначениеФичи3
		# Отдельная строка - отдельная категория фич: лексические варианты, семантические варианты, лексемы, etc
		for feature_name in self.lexical_variants:
			for feature_value in self.lexical_variants[feature_name]:
				lexical_variants_row.append(feature_name.encode('utf-8'))
				lexical_variants_row.append(feature_value.encode('utf-8'))

		for feature_name in self.semantic_variants:
			for feature_value in self.semantic_variants[feature_name]:
				semantic_variants_row.append(feature_name.encode('utf-8'))
				semantic_variants_row.append(feature_value.encode('utf-8'))

		for feature_name in self.lexemes:
			for feature_value in self.lexemes[feature_name]:
				lexemes_row.append(feature_name.encode('utf-8'))
				lexemes_row.append(feature_value.encode('utf-8'))				
		
		for feature_name in self.morphological:
			for feature_value in self.morphological[feature_name]:
				morphological_row.append(feature_name.encode('utf-8'))
				morphological_row.append(feature_value.encode('utf-8'))	

		for feature_name in self.class_for_lexeme:
			for feature_value in self.class_for_lexeme[feature_name]:
				class_for_lexeme_row.append(feature_name.encode('utf-8'))
				class_for_lexeme_row.append(feature_value.encode('utf-8'))	

		for feature_name in self.idiomatic:
			for feature_value in self.idiomatic[feature_name]:
				idiomatic_row.append(feature_name.encode('utf-8'))
				idiomatic_row.append(feature_value.encode('utf-8'))	

		for feature_name in self.universal_syntactic:
			for feature_value in self.universal_syntactic[feature_name]:
				universal_syntactic_row.append(feature_name.encode('utf-8'))
				universal_syntactic_row.append(feature_value.encode('utf-8'))	

		for feature_name in self.predefined:
			for feature_value in self.predefined[feature_name]:
				predefined_row.append(feature_name.encode('utf-8'))
				predefined_row.append(feature_value.encode('utf-8'))	

		rows_list = []
		rows_list.append(prefix_list + "\t".join(lexical_variants_row))
		rows_list.append(prefix_list + "\t".join(semantic_variants_row))
		rows_list.append(prefix_list + "\t".join(lexemes_row))
		rows_list.append(prefix_list + "\t".join(morphological_row))
		rows_list.append(prefix_list + "\t".join(class_for_lexeme_row))
		rows_list.append(prefix_list + "\t".join(idiomatic_row))
		rows_list.append(prefix_list + "\t".join(universal_syntactic_row))
		rows_list.append(prefix_list + "\t".join(predefined_row))
		text = '\n'.join(rows_list)

		return text
	
	
	def load(self, src):
		"""Загружает ядро из компактного """
		with open(src, 'rU') as csvfile:
			r = csv.reader(csvfile)
			for row in r:
				self.constituent_text = row[0]
				self.text = row[1]
				self.id = row[2]
				self.parent = row[3]
				for i in range(3, -1):
					self.features[row[i]] = 1


	# Если будет использован экспорт в csv, то может понадобиться 'нормализовать'
	# набор признаков так, чтобы у каждого ядра был одинаковый набор ключей
	# Для этого можно будет использовать что-то вроде normalizeFeatures:
	
	# Иван сказал, что это не понадобится, т.к. у всех ядер будет одинаковый
	# набор признаков - но мне лень выпиливать
	#    Стас
		
	def fillMissingFeatures(self, all_features, default_value=''):
		"""Дозаполняет словарь признаков отсутствующими ключами из all_features.
		Каждому добавленному ключу ставится в соответствие default_value"""
		
		for f in all_features:
			if not f in self.features:
				self.features[f] = ''

	# Последний сугубо вспомогательный метод
	def __repr__(self):
		"""Для вменяемого вывода"""
		return '{:7} : {}'.format( self.id, (self.constituent_text).encode('utf-8') )
	
###############################################################
# Основные функции
def printToCSV(sentence_cores, file_id):
	filename = "./with_all_features/file_"+file_id+".csv"
	with open(filename, "a") as out_file:
		#fl = ["Sentence_id","Begin_index","Constituent_text","Core_text","Core_id","Parent_id"]
		#out_file.write("\t".join(fl)+"\n")
		for sentence in sentence_cores:
			for core in sentence:
				out_file.write(core+"\n")
	sort_csv(filename, [int,int], [1,2])

# сортировка сначала по номеру предложения, потом по позиции в нем
# получим файл, в котором слова записаны в том же порядке, в котором идут в предложении
def sort_csv(csv_filename, types, sort_key_columns):
	"""sort (and rewrite) a csv file.
	types:  data types (conversion functions) for each column in the file
	sort_key_columns: column numbers of columns to sort by"""
	data = []
	with open(csv_filename, 'rb') as f:
		for row in csv.reader(f, delimiter='\t'):
			row[0] = int(row[0])
			row[1] = int(row[1])
			data.append(row)
	data.sort(key=operator.itemgetter(0,1))
	with open(csv_filename, 'wb') as f:
		csv.writer(f, delimiter='\t').writerows(data)

def convert(types, values):
	return [t(v) for t, v in zip(types, values)]

def loadDataset(path):
	"""Проходит по одному все файлы из папки, загружая их содержимое в
	список компактных данных о ядрах. Возвращает плоский список всех ядер"""
	
	cores = []
	
	for filename in os.listdir(path):
		print filename
		fullname = os.path.join(path, filename)
		file_id = filename.split('.')[0]
		# если нужен не плоский список, поменять на .append:
		#cores.extend(loadFromXml(fullname))
		printToCSV(loadFromXml(fullname), file_id)

	return cores

def loadFromXml(filename):
	"""Загружает полный список ядер из файла. Возвращает плоский список"""
	sentence_cores = []
	
	# собственно, загрузка файла
	tree = et.parse(filename)
	for sentence_id, sentence in enumerate(tree.getroot()):
		# если нужен двухуровневый список, поменять на .append:
		if (sentence[1].find("Structure0") != None):
			sentence_cores.append(loadSentence(sentence, sentence_id))

	return sentence_cores

def loadSentence(element, sentence_id):
	"""Загружает ядра из xml-элемента, соответствующего предложению.
	Возвращает плоский список"""
	structure0 = element[1].find('Structure0')
	normal_sentence = structure0[2]
	children = normal_sentence.find('Children')
	##original_id = int( normal_sentence.find('ConstituentId').text )
	# в конце названия координаты начала и конца составляющей в исходном тексте
	sentence_text = element[1].get('name')
	
	cores = []
	for child in children:
		# передаем элемент с тегом <ChildConstituent>
		# из элемента <TreeLink_Predicate>
		cores.extend(loadChild(child[-2][-1], -1, sentence_id))
	return cores

def loadChild(element, parent_id, sentence_id):
	cores = []

	# добавляем саму составляющую
	core = Core()
	##core.original_id = int( element.find( 'ConstituentId' ).text )

	core.constituent_text = element.find( 'SourceTextInfo' ).text.split('(')[0]
	# оставляем пустые составляющие-восстановленные эллипсисы
	#if core.constituent_text == "":
	#   return cores

	first_symbol = int( element[CORE].find( 'BeginInSourceText' ).text )
	last_symbol = int( element[CORE].find( 'EndInSourceText' ).text )
	core.text = extractCoreText(core.constituent_text, first_symbol, last_symbol)
	core.parent = parent_id
	core.sentence_id = sentence_id
	core.begin_index = int( element[CORE].find( 'BeginPosition' ).text )

	#добавляем лексические варианты
	lexical_variants = element[CORE].find('LexicalVariants')
	for lexical_variant in lexical_variants:
		if lexical_variant.text != None:
			core.lexical_variants[lexical_variant.tag] = lexical_variant.text.split(' | ')
	
	#добавляем семантические варианты
	semantic_variants = element[CORE].find('SemanticVariants')
	for semantic_variant in semantic_variants:
		if semantic_variant.text != None:
			core.semantic_variants[semantic_variant.tag] = semantic_variant.text.split(' | ')

	#добавляем лексемы
	#кажется, что нельзя их использовать, так как они даны с СК, в которые попадают
	lexemes = element[CORE].find('Lexemes')
	if lexemes != None:
		for lexeme in lexemes:
			core.features[lexeme.tag] = lexeme.text.split(' | ')
	
	# добавляем морфологические признаки
	morpho_features = element[CORE].find('GrammarValue').find('Morphological')
	for morpho_feature in morpho_features:
		if morpho_feature.text != None:
			core.morphological[morpho_feature.tag] = morpho_feature.text.split(' | ')

	# add classifying for lexeme
	class_for_lexemes = element[CORE].find('GrammarValue')[2]
	for class_for_lexeme in class_for_lexemes:
		if class_for_lexeme.text != None:
			core.class_for_lexeme[class_for_lexeme.tag] = class_for_lexeme.text.split(' | ')

	# add idiomatic
	idiomatics = element[CORE].find('GrammarValue').find('Idiomatic')
	for idiomatic in idiomatics:
		if idiomatic.text != None:
			core.idiomatic[idiomatic.tag] = idiomatic.text.split(' | ')

	# add universal syntactic
	universals = element[CORE].find('GrammarValue')[4]
	for universal in universals:
		if universal.text != None:
			core.universal_syntactic[universal.tag] = universal.text.split(' | ')

	# add predefined
	predefineds = element[CORE].find('GrammarValue').find('Predefined')
	for predefined in predefineds:
		if predefined.text != None:
			core.predefined[predefined.tag] = predefined.text.split(' | ')

	cores.append(core.save())

	# добавляем детей
	children = element.find('Children')

	if children != None:
		for child in children:
			cores.extend( loadChild(child[-2][-1], core.id, sentence_id) )

	return cores

def extractCoreText(constituent_text, first, last):
	if first == last == -1:
		return constituent_text
	return constituent_text[first:last]

def saveDataset(cores, target_filename):
	"""Сохраняет датасет в указанный файл
	
	cores - плоский список ядер
	target_filename - путь к файлу, в который нужно сохранить датасет

	# тут должно быть что-то вроде
	# 
	# some_format = IDontEvenKnowWhat(some, parameters)
	# for core in cores:
	#     some_format.AddSomehow(core.save())
	# some_format.write()
	"""
	with open(target_filename.csv, 'wt') as csvfile:
		w = csv.writer(csvfile)
		#fl = ["Sentence id","Begin index","Constituent text","Core text","Core id","Parent id"]
		#w.writerow(fl)
		for core in cores:
			row = [core.constituent_text, core.text, core.id, core.parent]
			for feature in core.features:
				row.append(feature)
			w.writerow(row)
	
def convert_data(xml_path, compact_filename):
	"""Конвертирует данные из папки с xml в компактный датасет и сохраняет его.
	Возвращает загруженный датасет
	
	xml_path - путь к папке с xml
	compact_filename - файл с компактным датасетом"""
	
	cores = loadDataset(xml_path)
	saveDataset(cores, compact_filename)
	
	return cores    


###############################################################
# Небольшой проверяющий код

def test():
	"""Небольшой проверяющий метод"""
	#some_xml = '_sentence.xml'
	#cores = loadFromXml(some_xml)
	loadDataset("./new_wiki/6627")
	#print len(cores)
	#print('\n'.join([repr(core) for core in cores]))
	
# не нужно писать такие вызовы на верхнем уровне модуля
# т.к. потом 'import semanticcore' будет работать 'с сюрпризом'
if( __name__ == '__main__'):
	test()

