% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

function inference()
	startup();
	%% perplexity

	%% configuration
	fprintf('\n\n############################################### configuration ###############################################\n\n');
	config = struct();
	% config.resume = false;
	config.output_dir = '20171122_110501_hidden_00128_batch_size_00512_activation_tanh_lr_0.01000000';
	config.resume_file = fullfile('./outputs', config.output_dir, 'data_epoch_100_lr_0.01_weightdecay_0_optim_momentum_seed_9904.mat');				% if one wants to test a trained model, please specify the file path
	assert(ischar(config.resume_file), 'the resume file does not exist');
	load(config.resume_file);

	fprintf('configuration is:\n'); disp(config);
	% rng(config.seed);

	% config.start_word = {'government', 'of', 'united'};
	% config.start_word = {'city', 'of', 'new'};
	% config.start_word = {'he', 'is', 'the'};
	config.start_word = {'you', 'do', 'n''t'};
	% config.start_word = {'although', 'she', 'was'};
	config.request_word = 'is';
	config.num_close_word = 5;


	config.cache_path = fullfile('cache');
	config.debug_mode = false;
	config.vis = true;
	config.batch_size = 1;
	config.activation = config.train.activation;
	config.num_output = 10;

	fprintf('\n\n############################################### loading data ###############################################\n\n');
	fprintf('loading data.....\n\n');

	assert(exist(config.cache_path, 'dir') == 7, 'the cached data is not found. Please run preprocessing.py first.');
	vocabulary_filepath = fullfile(config.cache_path, 'volcabulary.txt');
	[vocabulary_index, ~, ~] = parse_text_file(vocabulary_filepath, config.debug_mode);
	vocabulary_cell = cellfun(@(x) x{1}, vocabulary_index, 'UniformOutput', false);


	fprintf('\n\n############################################### inference ###############################################\n\n');
	fprintf('start inference.....\n\n');

	assert(length(config.start_word) == 3, 'the input length of words is not correct');
	
	%% encode
	data_sample = zeros(48, config.batch_size);
	word_index1 = find(strcmp(vocabulary_cell, config.start_word{1}) == 1);
	word_index2 = find(strcmp(vocabulary_cell, config.start_word{2}) == 1);
	word_index3 = find(strcmp(vocabulary_cell, config.start_word{3}) == 1);
	assert(length(word_index1) == 1 && length(word_index2) == 1 && length(word_index3) == 1, 'the length of word index is not correct');
	dictionary = fc_weight.input;
	data_sample = [dictionary(word_index1, :), dictionary(word_index2, :), dictionary(word_index3, :)]';
	% size(data_sample)

	%% predict and decode
	results_sentence = config.start_word;
	for output_index = 1:config.num_output
		output = forward_fc(fc_weight, data_sample, config, config.debug_mode);

		% size(output)
		[~, max_index] = max(output);
		if max_index == 6658
			break;
		end
		% max_index

		results_sentence{end + 1} = vocabulary_cell{max_index};

		data_sample = [data_sample(17:end); dictionary(max_index, :)'];
	end

	%% output the string
	fprintf('The predicted sentence starting with ''%s %s %s'' is: ''%s''\n', config.start_word{1}, config.start_word{2}, config.start_word{3}, strjoin(results_sentence));
	% strjoin(results_sentence)


	%% compute distance from the requested word
	word_index = find(strcmp(vocabulary_cell, config.request_word) == 1);
	assert(length(word_index) == 1, 'the requested word is not correct');
	word_embedding = dictionary(word_index, :);		
	word_embedding_matrix = repmat(word_embedding, 8000, 1);
	word_distance = distance_matrix(word_embedding_matrix, dictionary);
	[~, sorted_index] = sort(word_distance);
	
	closest_string = {};
	for close_word_index = 1:config.num_close_word
		word_index_tmp = sorted_index(close_word_index+1);
		word_tmp = vocabulary_cell{word_index_tmp};
		closest_string{end + 1} = word_tmp;
	end

	fprintf('The top 5 closest words to ''%s'' in dictionary are: %s\n', config.request_word, strjoin(closest_string));
	% sorted_index(1:10)
	% value(1:10)

	fprintf('\n\n############################################### done ###############################################\n\n');
end



