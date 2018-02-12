% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

function main()
	startup();

	%% configuration
	fprintf('\n\n############################################### configuration ###############################################\n\n');
	config = struct();
	config.resume = false;
	config.resume_file = 'data_epoch_100_lr_0.01_weightdecay_1e-08_optim_momentum_seed_9237.mat';				% if one wants to test a trained model, please specify the file path

	config.num_epoch = 200;
	config.num_class = 10;
	config.num_units = 100;
	% config.num_units2 = 100;							% if one wants to use a two-layer network, please fill in this variable
	config.im_width = 28;
	config.im_height = 28;

	config.data_path = fullfile('dataset');
	config.save_dir = fullfile('outputs', get_timestamp());
	config.cache_path = fullfile('cache');
	config.debug_mode = true;
	config.vis = true;
	config.seed = floor(rand(1) * 10000);
	% config.seed = 5602;								% if one wants to reproduce the results, please use the same seed logged in the files
	config.initialization_method = 'gaussian';
	config.check_grad = true;
	config.num_check = 10;

	config.train.batch_size = 1;
	config.train.shuffle = true;
	config.train.optim = 'momentum';
	config.train.momentum = 0.9;
	config.train.lr = 0.01;
	config.train.weight_decay = 0.00000001;
	config.train.activation = 'sigmoid';

	%% save
	mkdir_if_missing(config.save_dir);
	if config.vis
		vis_dir = fullfile(config.save_dir, 'visualization');
		mkdir_if_missing(vis_dir);
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

	train_data_file = fullfile(config.data_path, 'digitstrain.txt');
	valid_data_file = fullfile(config.data_path, 'digitsvalid.txt');
	test_data_file = fullfile(config.data_path, 'digitstest.txt');

	% cache the data to MATLAB format
	if exist(config.cache_path, 'dir')
		fprintf('cache data found in %s.....\n', config.cache_path);		
		load(fullfile(config.cache_path, 'train_data.mat'), 'train_image_data', 'train_labels');
		load(fullfile(config.cache_path, 'valid_data.mat'), 'valid_image_data', 'valid_labels');
		load(fullfile(config.cache_path, 'test_data.mat'), 'test_image_data', 'test_labels');
	else
		train_data = parse_matrix_file(train_data_file, config.debug_mode);
		valid_data = parse_matrix_file(valid_data_file, config.debug_mode);
		test_data = parse_matrix_file(test_data_file, config.debug_mode);

		if config.debug_mode
			assert(size(train_data, 2) == config.im_height * config.im_width + 1, sprintf('image dimension is not right for training data: %d vs %d\n', size(train_data, 2), config.im_height * config.im_width + 1));
			assert(size(valid_data, 2) == config.im_height * config.im_width + 1, sprintf('image dimension is not right for validation data: %d vs %d\n', size(valid_data, 2), config.im_height * config.im_width + 1));
			assert(size(test_data, 2) == config.im_height * config.im_width + 1, sprintf('image dimension is not right for testing data: %d vs %d\n', size(test_data, 2), config.im_height * config.im_width + 1));
		end

		% reshape the data to image
		train_image_data = reshape(train_data(:, 1:end-1), [], config.im_height, config.im_width);
		train_labels = train_data(:, end);
		valid_image_data = reshape(valid_data(:, 1:end-1), [], config.im_height, config.im_width);
		valid_labels = valid_data(:, end);
		test_image_data = reshape(test_data(:, 1:end-1), [], config.im_height, config.im_width);
		test_labels = test_data(:, end);

		% cache the data
		mkdir_if_missing(config.cache_path);
		train_data_cache_path = fullfile(config.cache_path, 'train_data.mat');
		valid_data_cache_path = fullfile(config.cache_path, 'valid_data.mat');
		test_data_cache_path = fullfile(config.cache_path, 'test_data.mat');
		save(train_data_cache_path, 'train_image_data', 'train_labels');
		save(valid_data_cache_path, 'valid_image_data', 'valid_labels');
		save(test_data_cache_path, 'test_image_data', 'test_labels');
	end

	num_train = size(train_image_data, 1);
	num_valid = size(valid_image_data, 1);
	num_test = size(test_image_data, 1);

	% convert label from integer to one hot vector
	train_labels_matrix = zeros(num_train, config.num_class);
	valid_labels_matrix = zeros(num_valid, config.num_class);
	test_labels_matrix = zeros(num_test, config.num_class);
	lable_range = [0, 9];
	for index = 1:num_train
		label_data = train_labels(index, 1);
		label_vector = number2onehot(label_data, lable_range, config.debug_mode);
		train_labels_matrix(index, :) = label_vector';
	end
	for index = 1:num_valid
		label_data = valid_labels(index, 1);
		label_vector = number2onehot(label_data, lable_range, config.debug_mode);
		valid_labels_matrix(index, :) = label_vector';
	end
	for index = 1:num_test
		label_data = test_labels(index, 1);
		label_vector = number2onehot(label_data, lable_range, config.debug_mode);
		test_labels_matrix(index, :) = label_vector';
	end

	% check dimension
	if config.debug_mode
		assert(all(size(train_image_data) == [size(train_labels, 1), config.im_height, config.im_width]), sprintf('image dimension is not right for training data: [%d, %d, %d] vs [%d, %d, %d]\n', size(train_image_data, 1), size(train_image_data, 2), size(train_image_data, 3), size(train_labels, 1), config.im_height, config.im_width));
		assert(all(size(valid_image_data) == [size(valid_labels, 1), config.im_height, config.im_width]), sprintf('image dimension is not right for validation data: [%d, %d, %d] vs [%d, %d, %d]\n', size(valid_image_data, 1), size(valid_image_data, 2), size(valid_image_data, 3), size(valid_labels, 1), config.im_height, config.im_width));
		assert(all(size(test_image_data)  == [size(test_labels,  1), config.im_height, config.im_width]), sprintf('image dimension is not right for testing data: [%d, %d, %d] vs [%d, %d, %d]\n', size(test_image_data, 1), size(test_image_data, 2), size(test_image_data, 3), size(test_labels, 1), config.im_height, config.im_width));
		assert(all(size(train_labels_matrix) == [num_train, config.num_class]), sprintf('label dimension is not right for training data: [%d, %d] vs [%d, %d]\n',   size(train_labels_matrix, 1), size(train_labels_matrix, 2), num_train, config.num_class));
		assert(all(size(valid_labels_matrix) == [num_valid, config.num_class]), sprintf('label dimension is not right for validation data: [%d, %d] vs [%d, %d]\n', size(valid_labels_matrix, 1), size(valid_labels_matrix, 2), num_valid, config.num_class));
		assert(all(size(test_labels_matrix)  == [num_test , config.num_class]), sprintf('label dimension is not right for testing data: [%d, %d] vs [%d, %d]\n',    size(test_labels_matrix, 1) ,  size(test_labels_matrix, 2), num_test,  config.num_class));
	end
	fprintf('dimension of input training data (num_data, height, width): (%d, %d, %d)\n', size(train_image_data, 1), size(train_image_data, 2), size(train_image_data, 3));
	fprintf('dimension of input validation data (num_data, height, width): (%d, %d, %d)\n', size(valid_image_data, 1), size(valid_image_data, 2), size(valid_image_data, 3));
	fprintf('dimension of input testing data (num_data, height, width): (%d, %d, %d)\n', size(test_image_data, 1), size(test_image_data, 2), size(test_image_data, 3));

	% visualize the input image samples
	if config.vis 
		vis_input_path = fullfile(vis_dir, 'input_samples');
		mkdir_if_missing(vis_input_path);

		for image_index = 1:300:num_train
			image_tmp = squeeze(train_image_data(image_index, :, :));
			image_savepath = fullfile(vis_input_path, sprintf('input_sample_index_%010d.jpg', image_index));
			imwrite(image_tmp, image_savepath);
			fprintf('save sample input to %s\n', image_savepath);
		end
	end

	% convert the image data to vector for computation
	train_image_data = reshape(train_image_data, num_train, []);
	valid_image_data = reshape(valid_image_data, num_valid, []);
	test_image_data = reshape(test_image_data, num_test, []);

	if config.resume
		assert(ischar(config.resume_file), 'the resume file does not exist');
		load(config.resume_file);
	else
		fprintf('\n\n############################################ create the network ############################################\n\n');
		if isfield(config, 'num_units2')			% two-layers network
			config.net = [config.im_width*config.im_height, config.num_units, config.num_units2, config.num_class];
		else						
			config.net = [config.im_width*config.im_height, config.num_units, config.num_class];
		end
			
		num_layer = length(config.net) - 1;
		fc_weight = weights_initialization(config.net, config.initialization_method, config.debug_mode);	
		fprintf('network is:\n'); disp(config.net);
		fprintf('network has %d layers, including the input layer\n', num_layer);

		if config.check_grad
			fprintf('\n\n############################################### gradient checking ###############################################\n\n');
			epsilon = 0.0001;
			check_sample_id = randperm(size(train_image_data, 1), 1);
			data_temp = train_image_data(check_sample_id, :)';  		% N x 1
			label_temp = train_labels_matrix(check_sample_id, :)';      		% C x 1 
			[~, post_activation, ~] = forward_fc(fc_weight, data_temp, config.train, config.debug_mode);
			gradients = backward_fc(fc_weight, data_temp, label_temp, post_activation, config.train, config.debug_mode);

			% check the gradient of weight W for each layer
			fprintf('start checking %d randomly chosen weight W at each layer\n', config.num_check);
			fc_weight_check = struct();
			fc_weight_check.b = fc_weight.b;
			for layer_index = 1:num_layer
				fprintf('checking layer %d\n', layer_index);
				dim1 = randperm(size(fc_weight.W{layer_index}, 1), config.num_check);   % randomly choose some weight to check
				dim2 = randperm(size(fc_weight.W{layer_index}, 2), config.num_check);
				for check_index = 1:config.num_check           % check each weight individually
					new_W_plus = fc_weight.W;
					new_W_minus = fc_weight.W;
					new_W_plus{layer_index}(dim1(check_index), dim2(check_index)) = new_W_plus{layer_index}(dim1(check_index), dim2(check_index)) + epsilon;
					new_W_minus{layer_index}(dim1(check_index), dim2(check_index)) = new_W_minus{layer_index}(dim1(check_index), dim2(check_index)) - epsilon;

					fc_weight_check.W = new_W_plus;
					[output_W_plus, ~, ~] = forward_fc(fc_weight_check, data_temp, config.train, config.debug_mode);
					fc_weight_check.W = new_W_minus;
					[output_W_minus, ~, ~] = forward_fc(fc_weight_check, data_temp, config.train, config.debug_mode);

					loss_plus_W = -log(output_W_plus' * label_temp);        % new computed loss
					loss_minus_W = -log(output_W_minus' * label_temp);

					grad_check = (loss_plus_W - loss_minus_W) / (2 * epsilon);    % gradient with respect to W
					grad_W_temp = gradients.W{layer_index}(dim1(check_index), dim2(check_index));

					% compute the relative error 
					if grad_check == 0 && grad_W_temp == 0
						error_tmp = 0;
					else
						error_tmp = abs(grad_check - grad_W_temp) / max(abs(grad_check), abs(grad_W_temp));	
					end
					if error_tmp > epsilon
						disp('gradient error for weights!!!');
						keyboard;
					end
				end
			end
			disp('no gradient error found in weight W');

			% check the gradient of bias b for each layer
			fprintf('start checking %d randomly chosen bias b at each layer\n', config.num_check);
			fc_weight_check = struct();
			fc_weight_check.W = fc_weight.W;
			for layer_index = 1:num_layer
				fprintf('checking layer %d\n', layer_index);
				dim1 = randperm(size(fc_weight.b{layer_index}, 1), config.num_check);   % randomly choose some weight to check
				for check_index = 1:config.num_check           % check each weight individually
					new_b_plus = fc_weight.b;
					new_b_minus = fc_weight.b;
					new_b_plus{layer_index}(dim1(check_index)) = new_b_plus{layer_index}(dim1(check_index)) + epsilon;
					new_b_minus{layer_index}(dim1(check_index)) = new_b_minus{layer_index}(dim1(check_index)) - epsilon;

					fc_weight_check.b = new_b_plus;
					[output_b_plus, ~, ~] = forward_fc(fc_weight_check, data_temp, config.train, config.debug_mode);
					fc_weight_check.b = new_b_minus;
					[output_b_minus, ~, ~] = forward_fc(fc_weight_check, data_temp, config.train, config.debug_mode);

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

		% %% visualize the feature map after initialization
		% number_visualize = 64;
		% visualize_id = randperm(number_units, number_visualize);
		% featureMap = W{1}(visualize_id, :);
		% map = zeros(input_size, input_size, 1, number_visualize);
		% for i = 1:number_visualize
		% 	feature_temp = featureMap(i, :);
		% 	min_temp = min(feature_temp);
		% 	feature_temp = feature_temp - min_temp;
		% 	max_temp = max(feature_temp);
		% 	feature_temp = feature_temp ./ max_temp;
		% 	map_temp = reshape(feature_temp, input_size, input_size);
		% 	%     map_temp = (map_temp + min_temp)./max_temp;
		% 	map(:, :, 1, i) = map_temp;
		% end
		% figure;
		% montage(map);
		% % print('../writeup/feature_map_before.eps', '-depsc');

		fprintf('\n\n############################################### start training ###############################################\n\n');
		fprintf('training parameters are:\n'); disp(config.train);

		train_acc = zeros(config.num_epoch, 1);
		train_loss = zeros(config.num_epoch, 1);
		valid_loss = zeros(config.num_epoch, 1);
		valid_acc = zeros(config.num_epoch, 1);

		%% train the network
		for j = 1:config.num_epoch
			diary off;
			fc_weight = train(fc_weight, train_image_data, train_labels_matrix, config.train, config.debug_mode);

			[train_accuracy(j), train_loss_avg(j)] = eval_classification(fc_weight, train_image_data, train_labels_matrix, config, config.debug_mode);
			[valid_accuracy(j), valid_loss_avg(j)] = eval_classification(fc_weight, valid_image_data, valid_labels_matrix, config, config.debug_mode);

			diary on;
			fprintf('Epoch %d - accuracy (train, validation): (%.5f, %.5f) \t loss (train, validation): (%.5f, %.5f) \n', j, train_accuracy(j), valid_accuracy(j), train_loss_avg(j), valid_loss_avg(j));
		end

		%% save the model and parameters
		save_name = sprintf('data_epoch_%s_lr_%s_weightdecay_%s_optim_%s_seed_%s.mat', num2str(config.num_epoch), num2str(config.train.lr), num2str(config.train.weight_decay), config.train.optim, num2str(config.seed));
		save_path = fullfile(config.save_dir, save_name);
		save(save_path, 'fc_weight', 'config', 'train_accuracy', 'valid_accuracy', 'train_loss_avg', 'valid_loss_avg');
	end

	fprintf('\n\n############################################### testing ###############################################\n\n');
	[test_accuracy, test_loss_avg] = eval_classification(fc_weight, test_image_data, test_labels_matrix, config, config.debug_mode);
	fprintf('Testing dataset: accuracy is %2.2f%% \t loss is %.5f\n', test_accuracy * 100, test_loss_avg);

	% %% visualize the confusion matrix
	% outputs = Classify(W, b, test_data);		
	% [~, cm] = confusion(test_labels', outputs');
	% cm = cm./max(max(cm));
	% cm = imresize(cm, [classes*10, classes*10], 'nearest');
	% figure;
	% imshow(cm);
	% % print('../writeup/confusion_matrix.eps', '-depsc2');

	if config.vis
		fprintf('\n\n############################################### visualizing the weights ###############################################\n\n');
		vis_weight = fc_weight.W{1};		% 100 x 784
		num_map = size(vis_weight, 1);
		map = zeros(config.im_height, config.im_width, 1, num_map);
		for index_map = 1:num_map
			weight_tmp = vis_weight(index_map, :);

			% normalize
			min_val = min(weight_tmp);
			weight_tmp = weight_tmp - min_val;
			max_val = max(weight_tmp);
			weight_tmp = weight_tmp ./ max_val;

			map_tmp = reshape(weight_tmp, config.im_height, config.im_width);
			map(:, :, 1, index_map) = map_tmp;
		end
		figure(1);
		montage(map);

		save_path = fullfile(vis_dir, 'weight_visualization.eps');
		print(save_path, '-depsc');
		close(1);
	end

	fprintf('\n\n############################################### plot curve and save ###############################################\n\n');
	vis_curve_dir = fullfile(vis_dir, 'training_curve');
	mkdir_if_missing(vis_curve_dir);

	% param_tmp = sprintf('(lr: %1.2f, mo: %1.1f)', config.train.lr, config.train.momentum);
	% param_tmp = sprintf('(number of hidden: %d)', config.num_units);
	param_tmp = '';
	title1 = sprintf('Classification Error %s', param_tmp);
	title2 = sprintf('Cross-Entropy Loss %s', param_tmp);

	figure(1);
	plot(1:config.num_epoch, 1 - train_accuracy);
	hold on;
	plot(1:config.num_epoch, 1 - valid_accuracy);
	hold off;
	lg = legend('training dataset', 'validation dataset', 'Location', 'northeast');
	lg.FontSize = 16;
	title(title1, 'FontSize', 26);
	xlabel('Epoch', 'FontSize', 26);
	ylabel('Error', 'FontSize', 26);
	set(gca, 'fontsize', 16);
	save_path = fullfile(vis_curve_dir, 'classification_error_graph.eps');
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
	fprintf('save average cross-entropy loss vs training curve to %s\n', save_path);
	close(2);

	fprintf('\n\n############################################### done ###############################################\n\n');
end