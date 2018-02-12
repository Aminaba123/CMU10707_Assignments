% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

function main()
	startup();
	%% perplexity

	%% configuration
	fprintf('\n\n############################################### configuration ###############################################\n\n');
	config = struct();
	config.resume = false;
	config.resume_file = 'data_epoch_300_lr_0.01_weightdecay_0_optim_sgd_seed_3873.mat';				% if one wants to test a trained model, please specify the file path

	config.num_epoch = 100;
	config.num_class = 8000;
	config.num_units = 128;
	config.length_embedding = 16;
	config.num_gram = 4;

	config.data_path = fullfile('dataset');
	config.cache_path = fullfile('cache');
	config.debug_mode = false;
	config.vis = true;
	config.seed = floor(rand(1) * 10000);
	% config.seed = 5602;								% if one wants to reproduce the results, please use the same seed logged in the files
	config.initialization_method = 'gaussian';
	config.check_grad = false;
	config.num_check = 10;

	config.train.batch_size = 512;
	config.train.shuffle = true;
	config.train.optim = 'momentum';
	config.train.momentum = 0.9;
	config.train.lr = 0.01;
	config.train.weight_decay = 0;
	config.train.activation = 'tanh';
	config.train.length_embedding = config.length_embedding;

	config.exp_title = sprintf('%s_hidden_%05d_batch_size_%05d_activation_%s_lr_%.8f_embedding_%d', get_timestamp(), config.num_units, config.train.batch_size, config.train.activation, config.train.lr, config.length_embedding);
	config.save_dir = fullfile('outputs', config.exp_title);

	%% save
	mkdir_if_missing(config.save_dir);
	if config.vis
		vis_dir = fullfile(config.save_dir, 'visualization');
		dictionary_dir = fullfile(vis_dir, 'dictionary');
		mkdir_if_missing(vis_dir);
		mkdir_if_missing(dictionary_dir);
	end
	config_savepath = fullfile(config.save_dir, 'configuration.txt');
	save_struct(config, config_savepath, config.debug_mode);
	train_savepath = fullfile(config.save_dir, 'training_parameters.txt');
	save_struct(config.train, train_savepath, config.debug_mode);
	log_name = sprintf('log_epoch_%s_lr_%s_weightdecay_%s_optim_%s_seed_%s.txt', num2str(config.num_epoch), num2str(config.train.lr), num2str(config.train.weight_decay), config.train.optim, num2str(config.seed));
	logfile_savepath = fullfile(config.save_dir, log_name);
	diary(logfile_savepath);

	fprintf('configuration is:\n'); disp(config);
	rng(config.seed);

	fprintf('\n\n############################################### loading data ###############################################\n\n');
	fprintf('loading data.....\n\n');

	assert(exist(config.cache_path, 'dir') == 7, 'the cached data is not found. Please run preprocessing.py first.');
	cached_train_data = fullfile(config.cache_path, 'train_data.mat');
	cached_valid_data = fullfile(config.cache_path, 'valid_data.mat');
	if exist(cached_train_data, 'file') && exist(cached_valid_data, 'file')
		fprintf('cache data found in %s.....\n', config.cache_path);		
		load(cached_valid_data, 'valid_text_data', 'valid_labels_matrix', 'length_lines_valid');
		load(cached_train_data, 'train_text_data', 'train_labels_matrix', 'length_lines_train');
	else
		volcabulary_filepath = fullfile(config.cache_path, 'volcabulary.txt');
		train_datapath = fullfile(config.cache_path, 'parsed_train.txt');
		valid_datapath = fullfile(config.cache_path, 'parsed_valid.txt');

		assert(exist(volcabulary_filepath, 'file') == 2, 'the volcabulary is not found in the cached data. Please run preprocessing.py first');
		assert(exist(train_datapath, 'file') == 2, 'the parsed training data is not found in the cached data. Please run preprocessing.py first');
		assert(exist(valid_datapath, 'file') == 2, 'the parsed validation data is not found in the cached data. Please run preprocessing.py first');

		% fprintf('reading matrix file...\n');
		% train_data = parse_matrix_file(train_datapath);
		train_data = dlmread(train_datapath);
		train_text_data = train_data(:, 1:3);					% N x 3		number
		train_labels = train_data(:, end);						% N x 1 	number
		% valid_data = parse_matrix_file(valid_datapath);
		valid_data = dlmread(valid_datapath);
		valid_text_data = valid_data(:, 1:3);
		valid_labels = valid_data(:, end);

		% fprintf('converting to one-hot... validation\n');
		num_train = length(train_text_data);
		num_valid = length(valid_text_data);
		label_range = [1, config.num_class];
		valid_labels_matrix = vector2onehot(valid_labels, label_range, config.debug_mode);
		train_labels_matrix = vector2onehot(train_labels, label_range, config.debug_mode);

		% parse the line count
		train_text_datapath = fullfile(config.cache_path, 'parsed_train_text.txt');
		valid_text_datapath = fullfile(config.cache_path, 'parsed_valid_text.txt');
		[~, ~, length_lines_train] = parse_text_file(train_text_datapath, config.debug_mode);
		[~, ~, length_lines_valid] = parse_text_file(valid_text_datapath, config.debug_mode);

		save(cached_train_data, 'train_text_data', 'train_labels_matrix', 'length_lines_train');
		save(cached_valid_data, 'valid_text_data', 'valid_labels_matrix', 'length_lines_valid');
	end

	% normrnd(0, 2/(net(i) + net(i+1)), [net(i+1), net(i)]);

	%% convert the index of word to 16-dimensional vector
	% volcabulary = normrnd(0, 0.01, [config.num_class, config.length_embedding]);						% 8000 x 16
	volcabulary = rand(config.num_class, config.length_embedding);						% 8000 x 16
	volcabulary_savepath = fullfile(dictionary_dir, 'volcabulary_initial_matrix.txt');
	dlmwrite(volcabulary_savepath, volcabulary);

	num_train = length(train_text_data);
	train_data = zeros(num_train, config.length_embedding * (config.num_gram - 1));		% 86402 x 48
	for data_index = 1:num_train
		train_data(data_index, 1:config.length_embedding) = volcabulary(train_text_data(data_index, 1), :);
		train_data(data_index, config.length_embedding+1:config.length_embedding*2) = volcabulary(train_text_data(data_index, 2), :);
		train_data(data_index, config.length_embedding*2+1:config.length_embedding*3) = volcabulary(train_text_data(data_index, 3), :);
	end
	num_valid = length(valid_text_data);
	valid_data = zeros(num_valid, config.length_embedding * (config.num_gram - 1));		% 86402 x 48
	for data_index = 1:num_valid
		valid_data(data_index, 1:config.length_embedding) = volcabulary(valid_text_data(data_index, 1), :);
		valid_data(data_index, config.length_embedding+1:config.length_embedding*2) = volcabulary(valid_text_data(data_index, 2), :);
		valid_data(data_index, config.length_embedding*2+1:config.length_embedding*3) = volcabulary(valid_text_data(data_index, 3), :);
	end

	%% start training
	if config.resume
		assert(ischar(config.resume_file), 'the resume file does not exist');
		load(config.resume_file);
		
		config_savepath = fullfile(config.save_dir, 'configuration.txt');
		save_struct(config, config_savepath, config.debug_mode);
		train_savepath = fullfile(config.save_dir, 'training_parameters.txt');
		save_struct(config.train, train_savepath, config.debug_mode);
	else
		fprintf('\n\n############################################ create the network ############################################\n\n');
		config.net = [(config.num_gram - 1) * config.length_embedding, config.num_units, config.num_class];
		num_layer = length(config.net) - 1;
		fc_weight = weights_initialization_fc(config.net, config.initialization_method, config.debug_mode);	
		fc_weight.input = volcabulary;

		fprintf('network is:\n'); disp(config.net);
		fprintf('network has %d layers, including the input layer\n', num_layer);

		fprintf('training parameters are:\n'); disp(config.train);

		if config.check_grad
			fprintf('\n\n############################################### gradient checking ###############################################\n\n');
			epsilon = 0.0001;
			config_check = config;
			config_check.train.batch_size = 1;

			% size(train_data, 1)
			check_sample_id = randperm(size(train_data, 1), config_check.train.batch_size);

			% check_sample_id
			data_temp = train_data(check_sample_id, :)';  						% 48 x batch_size
			label_temp = train_labels_matrix(check_sample_id, :)';      		% 8000 x batch_size

			[~, post_activation, ~] = forward_fc(fc_weight, data_temp, config_check.train, config_check.debug_mode);
			gradients = backward_fc(fc_weight, data_temp, label_temp, post_activation, config_check.train, config_check.debug_mode);

			% check the gradient of weight W for each layer
			fprintf('start checking %d randomly chosen weight W at each layer\n', config_check.num_check);
			fc_weight_check = struct();
			fc_weight_check.b = fc_weight.b;
			for layer_index = 1:num_layer
				fprintf('checking layer %d\n', layer_index);
				dim1 = randperm(size(fc_weight.W{layer_index}, 1), config_check.num_check);   % randomly choose some weight to check
				dim2 = randperm(size(fc_weight.W{layer_index}, 2), config_check.num_check);
				for check_index = 1:config_check.num_check           % check each weight individually
					new_W_plus = fc_weight.W;
					new_W_minus = fc_weight.W;
					new_W_plus{layer_index}(dim1(check_index), dim2(check_index)) = new_W_plus{layer_index}(dim1(check_index), dim2(check_index)) + epsilon;
					new_W_minus{layer_index}(dim1(check_index), dim2(check_index)) = new_W_minus{layer_index}(dim1(check_index), dim2(check_index)) - epsilon;

					fc_weight_check.W = new_W_plus;
					[output_W_plus, ~, ~] = forward_fc(fc_weight_check, data_temp, config_check.train, config_check.debug_mode);				% 8000 x batch_size
					fc_weight_check.W = new_W_minus;
					[output_W_minus, ~, ~] = forward_fc(fc_weight_check, data_temp, config_check.train, config_check.debug_mode);

					% size(output_W_plus)

					loss_plus_W = -log(output_W_plus' * label_temp);        % new computed loss
					loss_minus_W = -log(output_W_minus' * label_temp);

					% loss_plus_W
					% loss_minus_W

					grad_check = (loss_plus_W - loss_minus_W) / (2 * epsilon);    % gradient with respect to W, 				% 8000 x batch_size
					grad_W_temp = gradients.W{layer_index}(dim1(check_index), dim2(check_index));

					% size(grad_check)
					% size(grad_W_temp)

					% compute the relative error 
					if grad_check == 0 && grad_W_temp == 0
						error_tmp = 0;
					else
						error_tmp = abs(grad_check - grad_W_temp) / max(abs(grad_check), abs(grad_W_temp));	
					end
					if error_tmp > epsilon
						disp('gradient error for weights!!!');
						disp(error_tmp)
						keyboard;
					end
				end
			end
			disp('no gradient error found in weight W');

			% check the gradient of bias b for each layer
			fprintf('start checking %d randomly chosen bias b at each layer\n', config_check.num_check);
			fc_weight_check = struct();
			fc_weight_check.W = fc_weight.W;
			for layer_index = 1:num_layer
				fprintf('checking layer %d\n', layer_index);
				dim1 = randperm(size(fc_weight.b{layer_index}, 1), config_check.num_check);   % randomly choose some weight to check
				for check_index = 1:config_check.num_check           % check each weight individually
					new_b_plus = fc_weight.b;
					new_b_minus = fc_weight.b;
					new_b_plus{layer_index}(dim1(check_index)) = new_b_plus{layer_index}(dim1(check_index)) + epsilon;
					new_b_minus{layer_index}(dim1(check_index)) = new_b_minus{layer_index}(dim1(check_index)) - epsilon;

					fc_weight_check.b = new_b_plus;
					[output_b_plus, ~, ~] = forward_fc(fc_weight_check, data_temp, config_check.train, config_check.debug_mode);
					fc_weight_check.b = new_b_minus;
					[output_b_minus, ~, ~] = forward_fc(fc_weight_check, data_temp, config_check.train, config_check.debug_mode);

					loss_plus_b = -log(output_b_plus' * label_temp);        % new computed loss
					loss_minus_b = -log(output_b_minus' * label_temp);

					grad_check = (loss_plus_b - loss_minus_b) / (2 * epsilon);    % gradient with respect to b
					grad_b_temp = gradients.b{layer_index}(dim1(check_index));

					% compute the relative error
					if grad_check == 0 && grad_W_temp == 0
						error_tmp = 0;
					else
						error_tmp = abs(grad_check - grad_b_temp) / max(abs(grad_check), abs(grad_b_temp));	
					end
					if error_tmp > epsilon
						disp('gradient error for bias!!!');
						keyboard;
					end
				end
			end
			disp('no gradient error found in bias b');
		end

		fprintf('\n\n############################################### start training ###############################################\n\n');

		train_loss = zeros(config.num_epoch, 1);
		valid_loss = zeros(config.num_epoch, 1);

		%% train the network
		time = tic;
		for j = 1:config.num_epoch
			diary off;

			fc_weight = train_fc(fc_weight, train_text_data, train_labels_matrix, config.train, config.debug_mode);

			volcabulary = fc_weight.input;
			volcabulary_savepath = fullfile(dictionary_dir, sprintf('volcabulary_initial_matrix_epoch%04d.txt', j));
			dlmwrite(volcabulary_savepath, volcabulary);

			% num_train = length(train_text_data);
			% train_data = zeros(num_train, config.length_embedding * (config.num_gram - 1));		% 8000 x 48
			for data_index = 1:num_train
				train_data(data_index, 1:config.length_embedding) = volcabulary(train_text_data(data_index, 1), :);
				train_data(data_index, config.length_embedding+1:config.length_embedding*2) = volcabulary(train_text_data(data_index, 2), :);
				train_data(data_index, config.length_embedding*2+1:config.length_embedding*3) = volcabulary(train_text_data(data_index, 3), :);
			end
			% num_valid = length(valid_text_data);
			% valid_data = zeros(num_valid, config.length_embedding * (config.num_gram - 1));		% 8000 x 48
			for data_index = 1:num_valid
				valid_data(data_index, 1:config.length_embedding) = volcabulary(valid_text_data(data_index, 1), :);
				valid_data(data_index, config.length_embedding+1:config.length_embedding*2) = volcabulary(valid_text_data(data_index, 2), :);
				valid_data(data_index, config.length_embedding*2+1:config.length_embedding*3) = volcabulary(valid_text_data(data_index, 3), :);
			end

			[train_perplexity(j), train_loss_avg(j)] = eval_perplexity(fc_weight, train_data, train_labels_matrix, length_lines_train, config, config.debug_mode);
			[valid_perplexity(j), valid_loss_avg(j)] = eval_perplexity(fc_weight, valid_data, valid_labels_matrix, length_lines_valid, config, config.debug_mode);

			diary on;
			% count the time
			elapsed = toc(time);
			remaining_str = string(convert_secs2time(elapsed / j * (config.num_epoch - j)));
			elapsed_str = string(convert_secs2time(toc(time)));

			fprintf('Epoch %d - perplexity (train, validation): (%.5f, %.5f) \t loss (train, validation): (%.5f, %.5f), EP: %s, ETA: %s\n', j, train_perplexity(j), valid_perplexity(j), train_loss_avg(j), valid_loss_avg(j), elapsed_str, remaining_str);

			% fprintf('Running part: %d/%d, %s, %s, %s, model: %s, path: %s, EP: %s, ETA: %s\n', i, num_images, config.datasets, config.subset, size_set, config.model_name, filename, elapsed_str, remaining_str);
		end

		%% save the model and parameters
		save_name = sprintf('data_epoch_%s_lr_%s_weightdecay_%s_optim_%s_seed_%s.mat', num2str(config.num_epoch), num2str(config.train.lr), num2str(config.train.weight_decay), config.train.optim, num2str(config.seed));
		save_path = fullfile(config.save_dir, save_name);
		save(save_path, 'fc_weight', 'config', 'train_loss_avg', 'valid_loss_avg', 'train_perplexity', 'valid_perplexity');
	end

	fprintf('\n\n############################################### plot curve and save ###############################################\n\n');
	vis_curve_dir = fullfile(vis_dir, 'training_curve');
	mkdir_if_missing(vis_curve_dir);

	param_tmp = sprintf('(units: %d)', config.num_units);
	% param_tmp = sprintf('(step: %d)', config.train.sampling_step);
	% param_tmp = '';
	title1 = sprintf('Perplexity %s', param_tmp);
	title2 = sprintf('Cross-Entropy Loss %s', param_tmp);

	figure(1);
	plot(1:config.num_epoch, train_perplexity);
	hold on;
	plot(1:config.num_epoch, valid_perplexity);
	hold off;
	lg = legend('training dataset', 'validation dataset', 'Location', 'northeast');
	lg.FontSize = 16;
	title(title1, 'FontSize', 26);
	xlabel('Epoch', 'FontSize', 26);
	ylabel('Error', 'FontSize', 26);
	set(gca, 'fontsize', 16);
	save_path = fullfile(vis_curve_dir, 'perplexity_graph.eps');
	print(save_path, '-depsc');
	fprintf('save classification error vs training curve to %s\n', save_path);
	close(1);

	figure(2);
	plot(1:config.num_epoch, train_loss_avg);
	hold on;
	plot(1:config.num_epoch, valid_loss_avg);
	hold off;
	lg = legend('training dataset', 'validation dataset', 'Location', 'northeast');
	lg.FontSize = 16;
	title(title2, 'FontSize', 26);
	xlabel('Epoch', 'FontSize', 26);
	ylabel('Loss', 'FontSize', 26);
	set(gca, 'fontsize', 16);
	save_path = fullfile(vis_curve_dir, 'cross_entropy_loss_graph.eps');
	print(save_path, '-depsc');
	fprintf('save average reconstruction error vs training curve to %s\n', save_path);
	close(2);

	fprintf('\n\n############################################### done ###############################################\n\n');
end