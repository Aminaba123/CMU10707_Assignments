# Author: Xinshuo Weng
# Email: xinshuow@andrew.cmu.edu

import os, sys, random
import numpy as np

from libs.visualization import visualize_pts_array
from libs.io import *


debug = True
num_sample = 500

embedding_filepath = os.path.join('./outputs', '20171122_141431_hidden_00128_batch_size_00512_activation_tanh_lr_0.01000000_embedding_2', 'visualization', 'dictionary', 'volcabulary_initial_matrix_epoch0099.txt')
embedding = load_2dmatrix_from_file(embedding_filepath, delimiter=',', dtype='float32', debug=debug)
embedding = np.transpose(embedding)

dictionary_filepath = os.path.join('./cache', 'volcabulary.txt')
dictionary, _ = load_txt_file(dictionary_filepath, debug=debug)

# print(embedding.shape)
num_embedding = embedding.shape[1]

sampled_index = random.sample(range(num_embedding), num_sample)			# 0-indexed, 0 - 7999

# print(dictionary)
label_list = [dictionary[sampled_index_tmp] for sampled_index_tmp in sampled_index]			# note that the dictionary is 1-indexed

# print(sampled_index)
# print(label_list)

save_path = 'embedding_visualization.eps'
visualize_pts_array(embedding[:, sampled_index], color_index=2, pts_size=1, label=True, label_list=label_list, xlim=[-1, 2], ylim=[-1, 2], label_size=3, save_path=save_path, debug=debug)