# Author: Xinshuo Weng
# Email: xinshuow@andrew.cmu.edu

import numpy as np
import os, argparse, json


from libs.io import load_txt_file, mkdir_if_missing, save_txt_file, is_path_exists
from libs.miscellaneous import sort_dict, scalar_list2str_list
from libs.visualization import visualize_bar

def get_index_from_list(input_list, value):
	if value in input_list:
		return input_list.index(value)
	else:
		return input_list.index('UNK')

def preprocessing(data_path, cache_path, volcabulary_size, debug=True):
	'''
	preprocessing the text data
	'''
	assert is_path_exists(data_path), 'the input path is not correct'
	mkdir_if_missing(cache_path)

	train_data_file = os.path.join(data_path, 'train.txt')
	valid_data_file = os.path.join(data_path, 'val.txt')
	train_data, num_lines_train = load_txt_file(train_data_file, debug)
	valid_data, num_lines_valid = load_txt_file(valid_data_file, debug);

	# select the unique word from the text
	num_words_accu = 0
	volcabulary = dict()
	for line_tmp in train_data:
		word_in_line = line_tmp.lower().split(' ')
		num_words_accu += len(word_in_line)

		# go through all words in the line
		for word_tmp in word_in_line:
			if word_tmp in volcabulary:
				volcabulary[word_tmp] += 1
			else:
				volcabulary[word_tmp] = 1
	
	assert num_words_accu == sum(volcabulary.values()), 'volcabulary is wrong'
	print('the size of untruncated dictionary is %d' % len(volcabulary))

	# truncate the dictionary
	sorted_volcabulary_list = sort_dict(volcabulary, debug=debug)
	truncated_volcabulary = dict(sorted_volcabulary_list[0:volcabulary_size-3])
	truncated_volcabulary['START'], truncated_volcabulary['END'], truncated_volcabulary['UNK'] = num_lines_train, num_lines_train, 0
	print('the size of truncated dictionary is %d' % len(truncated_volcabulary))
	final_volcabulary = truncated_volcabulary.keys()
	final_volcabulary_savepath = os.path.join(cache_path, 'volcabulary.txt')
	save_txt_file(final_volcabulary, final_volcabulary_savepath, debug=debug)

	# re-write the text
	save_train_file = os.path.join(cache_path, 'parsed_train_text.txt')
	if not is_path_exists(save_train_file):
		parsed_txt = []
		line_index = 1
		for line_tmp in train_data:
			word_in_line = line_tmp.lower().split(' ')
			parsed_line = ['START']
			# print('processing line %d/%d' % (line_index, num_lines_train))

			# go through all words in the line
			for word_tmp in word_in_line:
				if word_tmp in truncated_volcabulary:
					parsed_line.append(word_tmp)
				else:
					parsed_line.append('UNK')
					truncated_volcabulary['UNK'] += 1

			parsed_line.append('END')
			new_line = ' '.join(parsed_line)
			parsed_txt.append(new_line)
			line_index += 1
		save_txt_file(parsed_txt, save_train_file, debug=debug)

	# parse the text again and count the n-gram
	parsed_data, num_lines_train = load_txt_file(save_train_file, debug)
	n_gram_dict = dict()
	parsed_train_data = []
	for line_tmp in parsed_data:
		word_in_line = line_tmp.split(' ')
		# print('processing line %d/%d' % (line_index, num_lines_train))
		
		for gram_index in range(len(word_in_line) - 3):
			gram_tmp = ' '.join([word_in_line[gram_index], word_in_line[gram_index+1], word_in_line[gram_index+2], word_in_line[gram_index+3]])
			if gram_tmp in n_gram_dict:
				n_gram_dict[gram_tmp] += 1
			else:
				n_gram_dict[gram_tmp] = 1

			index_tmp1 = get_index_from_list(final_volcabulary, word_in_line[gram_index]) + 1			# convert to 1-indexed
			index_tmp2 = get_index_from_list(final_volcabulary, word_in_line[gram_index+1]) + 1
			index_tmp3 = get_index_from_list(final_volcabulary, word_in_line[gram_index+2]) + 1
			index_tmp4 = get_index_from_list(final_volcabulary, word_in_line[gram_index+3]) + 1
			index_list_tmp = [index_tmp1, index_tmp2, index_tmp3, index_tmp4]
			str_list_tmp = scalar_list2str_list(index_list_tmp, debug=debug)

			index_gram_list = ' '.join(str_list_tmp)
			parsed_train_data.append(index_gram_list)
	
	parsed_train_data_filepath = os.path.join(cache_path, 'parsed_train.txt')
	save_txt_file(parsed_train_data, parsed_train_data_filepath, debug=debug)
	print('the size of 4_gram dictionary is %d' % len(n_gram_dict))

	# save the top 50 n_gram
	num_top = 50
	top_gram_list = sort_dict(n_gram_dict, order='descending', debug=debug)[0:num_top]
	if not is_path_exists(top_gram_list):
		# print(top_gram_list)
		save_top_gram = './cache/top_gram.txt'
		top_gram_txt = []
		for top_index in range(num_top):
			top_gram_txt.append(top_gram_list[top_index][0])
		save_txt_file(top_gram_txt, save_top_gram, debug=debug)


	# parse the validation data
	parsed_valid_data_filepath = os.path.join(cache_path, 'parsed_valid.txt')
	parsed_valid_text_filepath = os.path.join(cache_path, 'parsed_valid_text.txt')
	if not (is_path_exists(parsed_valid_data_filepath) and is_path_exists(parsed_valid_text_filepath)):
		parsed_valid_data = []
		parsed_valid_text = []
		for line_tmp in valid_data:
			word_in_line = line_tmp.lower().split(' ')
			word_in_line = ['START'] + word_in_line + ['END']

			# print(word_in_line)
			word_in_line_parsed = [word_tmp if word_tmp in final_volcabulary else 'UNK' for word_tmp in word_in_line]
			# print(word_in_line_parsed)
			parsed_valid_text.append(' '.join(word_in_line_parsed))
		
			for gram_index in range(len(word_in_line) - 3):
				index_tmp1 = get_index_from_list(final_volcabulary, word_in_line[gram_index]) + 1			# convert to 1-indexed
				index_tmp2 = get_index_from_list(final_volcabulary, word_in_line[gram_index+1]) + 1
				index_tmp3 = get_index_from_list(final_volcabulary, word_in_line[gram_index+2]) + 1
				index_tmp4 = get_index_from_list(final_volcabulary, word_in_line[gram_index+3]) + 1
				index_list_tmp = [index_tmp1, index_tmp2, index_tmp3, index_tmp4]
				str_list_tmp = scalar_list2str_list(index_list_tmp, debug=debug)

				index_gram_list = ' '.join(str_list_tmp)
				parsed_valid_data.append(index_gram_list)

		save_txt_file(parsed_valid_data, parsed_valid_data_filepath, debug=debug)
		save_txt_file(parsed_valid_text, parsed_valid_text_filepath, debug=debug)

	# visualize the distribution
	save_distribution = './gram_distribution.eps'
	print('visualizing the bar graph')
	visualize_bar(sorted(n_gram_dict.values(), reverse=True), xlabel='index of 4-grams', title='Distribution of 4-grams and Counts', bin_size=0.5, save_path=save_distribution, vis=False, debug=debug)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', type=str, default='./datasets', help='dataset dir')
	parser.add_argument('--cache_path', type=str, default='./cache', help='root dir')
	parser.add_argument('--volca_size', type=int, default=8000, help='start frame')
	args = parser.parse_args()

	debug = True
	preprocessing(args.data_path, args.cache_path, args.volca_size, debug=debug)

if __name__ == '__main__':
	main()