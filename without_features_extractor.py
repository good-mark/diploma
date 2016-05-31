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
import re # for reg exp 

#import sys, os
#sys.stdout = open("output.txt", 'w')

CORE = 7
SOURCE = "\\Smurov\Lenta_lexical\\best\xml\\7621"
SOURCE_WITHOUT = "\\Smurov\Lenta_lexical\lexical\xml\\7621\\" #"C:\Diploma\Original_data\Ivan\\6629\\" #
DESTINATION = "./!WithoutOmonimia\csv_data\\7621\\"

class IDGEN:
	_lastId = 0 # счетчик уникальных id
	
def getNewId():
	"""Генерирует новый уникальный числовой id"""
	IDGEN._lastId += 1
	return IDGEN._lastId
	
###############################################################
# Core - 'компактный' класс с данными
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
		row = [str(self.sentence_id), str(self.begin_index), self.constituent_text.encode('utf-8'), self.text.encode('utf-8'), str(self.id), str(self.parent)]
		for list_values in self.features.values():
			for value in list_values:
				row.append(value.encode('utf-8'))
		return "\t".join(row)
	
	
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
	filename = DESTINATION + "file_" + file_id + ".csv"
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

def loadDataset(path, path_without):
	"""Проходит по одному все файлы из папки, загружая их содержимое в
	список компактных данных о ядрах. Возвращает плоский список всех ядер"""
	
	cores = []
	
	for filename in os.listdir(path):
		print filename
		fullname = os.path.join(path, filename)
		fullname_without = os.path.join(path_without, filename)
		file_id = filename.split('.')[0]
		# если нужен не плоский список, поменять на .append:
		#cores.extend(loadFromXml(fullname))
		printToCSV(loadFromXml(fullname, fullname_without), file_id)

	return cores

def loadFromXml(filename, fullname_without):
	"""Загружает полный список ядер из файла. Возвращает плоский список"""
	sentence_cores = []
	
	# собственно, загрузка файла
	tree = et.parse(filename)
	tree_without = et.parse(filename_without)
	if (not tree or not tree_without): # Проверить на наличие файла в обоих корпусах!!!
		print "Error! No such file: ", filename
		return []
	for sentence_id, sentence in enumerate(tree.getroot()):
		# если нужен двухуровневый список, поменять на .append:
		if (sentence.find("Structure0") != None):
			sentence_without = tree_without.getroot()[sentence_id]
			sentence_cores.append(loadSentence(sentence, sentence_without, sentence_id))

	return sentence_cores

def loadSentence(element, element_without, sentence_id):
	"""Загружает ядра из xml-элемента, соответствующего предложению.
	Возвращает плоский список"""
	normal_sentence = element.find('Structure0')[2]
	all_cores_without = element_without.find('ConstituentsWithoutParent')
	children = normal_sentence.find('Children')
	##original_id = int( normal_sentence.find('ConstituentId').text )
	# в конце названия координаты начала и конца составляющей в исходном тексте
	sentence_text = element.get('name')
	if sentence_text != '!Action Pact!': 	
		cores = []
		for child in children:
			# передаем элемент с тегом <ChildConstituent>
			# из элемента <TreeLink_Predicate>
			loaded_child = loadChild(all_cores_without, child[-2][-1], -1, sentence_id)
			if loaded_child != None:
				cores.extend(loaded_child)
			else:
				return []
		if len(cores) < 5:
			return []
		return cores

def loadChild(all_elements_without, element, parent_id, sentence_id):
	cores = []

	# добавляем саму составляющую
	core = Core()
	##core.original_id = int( element.find( 'ConstituentId' ).text )

	core.constituent_text = element.find( 'SourceTextInfo' ).text.split('(')[0]
	# if constituent is a proform, delete it.
	# Exclude all sentences contents proforms.

	if element[CORE].find('IsSubstitutedProform') == 'yes':
		return None
	if u'вики' in core.constituent_text or 'Number' in core.constituent_text:
		return None
	#if re.match(r'.|]|[|\\|[0-9]', core.constituent_text) != None:
		#return None
	if '.' in core.constituent_text or ']' in core.constituent_text:
		return None
	if '\\' in core.constituent_text or '[' in core.constituent_text:
		return None
	if '1' in core.constituent_text:
		return None
	if 'XV' in core.constituent_text or 'VI' in core.constituent_text:
		return None
	if len(core.constituent_text.split(' ')) > 1:
		return None


	lex_variants = element[CORE].find('LexicalVariants')
	if lex_variants != None:
		lex_class = lex_variants.find('LexicalClass')
		if lex_class != None:
			if '#ForeignWord:FOREIGN_WORD' in lex_class or '#Roman:ROMAN_ORDINAL' in lex_class:
				return None 
			if 'DIGITAL_NUMBER' in lex_class or 'REFERENCE_IN_TEXT' in lex_class:
				return None
			if 'Number' in lex_class:
				return None
			if 'WIKI_AS_WEB_RESOURCE' in lex_class:
				return None
	# оставляем пустые составляющие-восстановленные эллипсисы
	#if core.constituent_text == "":
	#   return cores

	first_symbol = int( element[CORE].find( 'BeginInSourceText' ).text )
	last_symbol = int( element[CORE].find( 'EndInSourceText' ).text )
	core.text = extractCoreText(core.constituent_text, first_symbol, last_symbol)
	core.parent = parent_id
	core.sentence_id = sentence_id
	core.begin_index = int( element[CORE].find( 'BeginPosition' ).text )

	# добавляем морфологические признаки из элемента БЕЗ снятой омонимии
	start = False
	for element_without in all_elements_without:
		if int( element_without.find("*[@name='BeginPosition']").text ) == core.begin_index:
			start = True
			grammar_value = element_without[-3].find('GrammarValue')
			morpho_features = grammar_value.find('Morphological')
			for morpho_feature in morpho_features:
				if morpho_feature.text != None:
					core.features[morpho_feature.get('name')] = morpho_feature.text.split(' | ')

			predefined = grammar_value.find('Predefined')
			for predefined_feature in predefined:
				if predefined_feature.text != None:
					core.features[predefined_feature.get('name')] = predefined_feature.text.split(' | ')

		else:
			if start:
				break # Несколько вариантов идут подряд друг за другом. Делаем брейк чтобы не перебирать лишнее
		

	#добавляем лексические варианты
	'''
	lexical_variants = element[CORE].find('LexicalVariants')
	for lexical_variant in lexical_variants:
		if lexical_variant.text != None:
			core.features[lexical_variant.tag] = lexical_variant.text.split(' | ')
	'''

	#добавляем лексемы
	#кажется, что нельзя их использовать, так как они даны с СК, в которые попадают
	'''
	lexemes = element[CORE].find('Lexemes')
	if lexemes != None:
		for lexeme in lexemes:
			core.features[lexeme.tag] = lexeme.text
	'''
	cores.append(core.save())

	# добавляем детей
	children = element.find('Children')

	if children != None:
		#print type(children), sentence_id #DEBUG
		for child in children:
			loaded_child = loadChild(all_elements_without, child[-2][-1], core.id, sentence_id)
			if loaded_child != None:
				cores.extend(loaded_child)
			else:
				return None

	return cores

def extractCoreText(constituent_text, first, last):
	if first == last == -1:
		return constituent_text
	return constituent_text[first:last]

def saveDataset(cores, target_filename):
	"""Сохраняет датасет в указанный файл
	
	cores - плоский список ядер
	target_filename - путь к файлу, в который нужно сохранить датасет"""

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
	loadDataset(SOURCE, SOURCE_WITHOUT)
	#print len(cores)
	#print('\n'.join([repr(core) for core in cores]))
	
# не нужно писать такие вызовы на верхнем уровне модуля
# т.к. потом 'import semanticcore' будет работать 'с сюрпризом'
if( __name__ == '__main__'):
	test()

