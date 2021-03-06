# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file define a set of functions which converting data type
import os
import numpy as np
import math
from itertools import islice
import struct
import functools

# from io import isdict
from ..io.check import *
# from check import *

######################################################### string related #########################################################
def character2onehot(character):
	'''
	In this function you need to output a one hot encoding of the ASCII character.
	'''
	assert isinteger(character) or isstring(character), 'input data type is not correct'
	
	# convert string to integer number
	if isstring(character):
		assert len(character) == 1, 'character should be a string with length 1'
		character = ord(character)

	one_hot_vec = np.zeros([self.nFeats, ], dtype='float32')
	one_hot_vec[character] = 1
	return one_hot_vec

def string2onehot(string):
	'''
	convert a string to a set of 2d one hot tensor
	'''
	assert isstring(string) and len(string) > 0, 'input should be a string with length larger than 0'
	
	one_hot_vec = [character2onehot(ord(string[0]))]
	for character in string[1:]:
		one_hot_vec = np.vstack((one_hot_vec, [character2onehot(ord(character))]))
	return one_hot_vec

def onehot2ord(onehot):
	'''
	convert one hot vector to a ord integer number
	'''
	assert isinstance(onehot, np.ndarray) and onehot.ndim == 1, 'input should be 1-d numpy array'
	assert sum(onehot) == 1 and np.count_nonzero(onehot) == 1, 'input numpy array is not one hot vector'
	return np.argmax(onehot)

def onehot2character(onehot):
	'''
	convert one hot vector to a character
	'''
	return chr(onehot2ord(onehot))

def onehot2string(onehot):
	'''
	convert a set of one hot vector to a string
	'''
	if isinstance(onehot, np.ndarray):
		onehot.ndim == 2, 'input should be 2-d numpy array'
		onehot = list(onehot)
	elif isinstance(onehot, list):
		assert CHECK_EQ_LIST([tmp.ndim for tmp in onehot]), 'input list of one hot vector should have same length'
	else:
		assert False, 'unknown error'

	assert all(sum(onehot_tmp) == 1 and np.count_nonzero(onehot_tmp) == 1 for onehot_tmp in onehot), 'input numpy array is not a set of one hot vector'
	ord_list = [onehot2ord(onehot_tmp) for onehot_tmp in onehot]
	return ord2string(ord_list)


def string2ord(string):
	'''
	convert a string to a list of ASCII character
	'''
	assert isstring(string) and len(string) > 0, 'input should be a string with length larger than 0'
	
	ord_list = []
	for character in string:
		ord_list.append(ord(character))
	return ord_list

def ord2string(ord_list):
	'''
	convert a list of ASCII character to a string
	'''
	assert isinstance(ord_list, list) and len(ord_list) > 0, 'input should be a list of ord with length larger than 0'
	assert all(isinteger(tmp) for tmp in ord_list), 'all elements in the list of ord should be integer'
	
	L = ''
	for o in ord_list:
		L += chr(o)
	
	return L

def string2ext_filter(string, debug=True):
	'''
	convert a string to an extension filter
	'''
	if debug:
		assert isstring(string), 'input should be a string'

	if isext(string):
		return string
	else:
		return '.' + string

######################################################### time related #########################################################
def convert_secs2time(seconds):
    '''
    format second to human readable way
    '''
    assert isfloat(seconds) or isinteger(seconds), 'input should be an integer or floating number to represent number of seconds'
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return '[%d:%02d:%02d]' % (h, m, s)

######################################################### dict related #########################################################
def get_subdict(dictionary, num, debug=True):	
	if debug:
		assert isdict(dictionary), 'dictionary is not correct'
		assert num > 0 and isinteger(num) and num <= len(dictionary), 'number of sub-dictionary is not correct'

	def take(num, iterable):
		return dict(islice(iterable, num))

	return take(num, dictionary.iteritems())

def sort_dict(dictionary, sort_base='value', order='descending', debug=True):
	'''
	sort a dictionary to a list
	'''
	if debug:
		assert isdict(dictionary), 'the input is not a dictionary'
		assert sort_base == 'value' or sort_base == 'key', 'the sorting is based on key or value'
		assert order == 'descending' or order == 'ascending', 'the sorting order is not descending or ascending'

	reverse = True if order == 'descending' else False
	if sort_base == 'value':
		return sorted(dictionary.iteritems(), key= lambda (k,v): (v,k), reverse=reverse)
	else:
		return sorted(dictionary.iteritems(), reverse=reverse)

def construct_dict_from_lists(list_key, list_value, debug=True):
	'''
	construct a distionary from two lists
	'''
	if debug:
		assert islist(list_key) and islist(list_value), 'the input key list and value list are not correct'
		assert len(list_key) == len(list_value), 'the length of two input lists are not equal'

	return dict(zip(list_key, list_value))

######################################################### list related #########################################################
def str2float_from_list(str_list, debug=True):
	'''
	convert a list of string to a list of floating number
	'''
	if debug:
		assert islist(str_list), 'input is not a list'
		assert all(isstring(str_tmp) for str_tmp in str_list), 'input is not a list of string'
	if any(len(str_tmp) == 0 for str_tmp in str_list):
		if debug:
			print('warning: the list of string contains empty element which will be removed before converting to floating number')
		str_list = filter(None, str_list)
	return [float(str_tmp) for str_tmp in str_list]

def merge_listoflist(listoflist, debug=True):
	'''
	merge a list of list in original order
	'''
	if debug:
		assert islistoflist(listoflist), 'the input is not a list of list'

	merged = list()
	for individual in listoflist:
		merged = merged + individual

	return merged

def remove_item_from_list(list_to_remove, item, debug=True):
	'''
	remove a single item from a list
	'''
	if debug:
		assert islist(list_to_remove), 'input list is not a list'
		
	try:
		list_to_remove.remove(item)
	except ValueError:
		print('Warning!!!!!! Item to remove is not in the list. Remove operation is not done.')

	return list_to_remove

def remove_list_from_list(list_all, list_to_remove, debug=True):
	'''
	remove a list "list_to_remove" from a list "list_all"
	'''
	if debug:
		assert islist(list_all), 'input list is not a list'
		assert islist(list_to_remove), 'remove list is not a list'
		
	for item in list_to_remove:
		try:
			list_all.remove(item)
		except ValueError:
			print('Warning!!!!!! Item to remove is not in the list. Remove operation is not done.')

	return list_all

def remove_empty_item_from_list(list_to_remove, debug=True):
	'''
	remove an empty string from a list
	'''
	if debug:
		assert islist(list_to_remove), 'input list is not a list'
	
	return remove_item_from_list(list_to_remove, '', debug=debug)

def scalar_list2str_list(scalar_list, debug=True):
	'''
	convert a list of scalar to a list of string
	'''	
	if debug:
		assert islist(scalar_list) and all(isscalar(scalar_tmp) for scalar_tmp in scalar_list), 'input list is not a scalar list'
	
	str_list = list()
	for item in scalar_list:
		str_list.append(str(item))

	return str_list

def scalar_list2float_list(scalar_list, debug=True):
	'''
	convert a list of scalar to a list of floating number
	'''
	if debug:
		assert islist(scalar_list) and all(isscalar(scalar_tmp) for scalar_tmp in scalar_list), 'input list is not a scalar list'
	
	float_list = list()
	for item in scalar_list:
		float_list.append(float(item))

	return float_list

def float_list2bytes(float_list, debug=True):
	'''
	convert a list of floating number to bytes
	'''
	if debug:
		assert isfloat(float_list) or (islist(float_list) and all(isfloat(float_tmp) for float_tmp in float_list)), 'input is not a floating number or a list of floating number'

	# convert a single floating number to a list with one item
	if isfloat(float_list):
		float_list = [float_list]

	try:
		binary = struct.pack('%sf' % len(float_list), *float_list)
	except ValueError:
		print('Warnings!!!! Failed to convert to bytes!!!!!')

	return binary

def list2tuple(input_list, debug=True):
	'''
	convert a list to a tuple
	'''
	if debug:
		assert islist(input_list), 'input is not a list'

	return tuple(input_list)

def find_common_from_lists(list1, list2, debug=True):
	'''
	find common items from 2 lists
	'''
	if debug:
		assert islist(list1), 'input is not a list'
		assert islist(list2), 'input is not a list'

	return list(set(list1).intersection(list2))

def list_reorder(input_list, order_index, debug=True):
	'''
	reorder a list based on a list of index
	'''
	if debug:
		assert islist(input_list) and islist(order_index), 'inputs are not two lists'
		assert len(input_list) == len(order_index), 'length of input lists is not equal'
		assert all(isscalar(index_tmp) for index_tmp in order_index), 'the list of order is not correct'

	return [ordered for whatever, ordered in sorted(zip(order_index, input_list))]

def reverse_list(input_list, debug=True):
	'''
	reverse a list
	'''
	if debug:
		assert islist(input_list), 'input is not a list'

	return input_list[::-1]

######################################################### math related #########################################################
def degree2radian(degree, debug=True):
	'''
	this function return degree given radians, difference from default math.degrees is that this function normalize the output in range [0, 2*pi)
	'''
	if debug:
		assert isfloat(degree) or isinteger(degree) or (isnparray(degree) and degree.size == 1), 'input degree number is not correct'

	radian = math.radians(degree)
	while radian < 0:
		radian += 2*math.pi
	while radian >= 2*math.pi:
		radian -= 2*math.pi

	return radian

def radian2degree(radian, debug=True):
	'''
	this function return radian given degree, difference from default math.degrees is that this function normalize the output in range [0, 360)
	'''
	if debug:
		assert isfloat(radian) or isinteger(radian) or (isnparray(radian) and radian.size == 1), 'input radian number is not correct'

	degree = math.degrees(radian)
	while degree < 0:
		degree += 360.0
	while degree >= 360.0:
		degree -= 360.0

	return degree

def float2percent(number, debug=True):
	'''
	convert a floating number to a string representing percentage
	'''
	try:
		number = float(number)
	except ValueError:
		print('could not convert to a floating number')
	return '{:.1%}'.format(number)

def number2onehot(number, ranges, debug=True):
	'''
	this function convert an integer number to a one hot vector
	inputs:
			number:			an integer
			ranges:			[min, max], inclusive, both are integers
	'''
	if debug:
		assert isinteger(number), 'input number is not an integer'
		assert len(ranges) == 2, 'input range is not correct'
		assert isinteger(ranges[0]) and isinteger(ranges[1]), 'the input range should be integer'
		assert ranges[0] <= ranges[1], 'the input range is not correct'

	num_integers = ranges[1] - ranges[0] + 1
	index = number - ranges[0]
	onehot = np.zeros([num_integers, ], dtype='float32')
	onehot[index] = 1