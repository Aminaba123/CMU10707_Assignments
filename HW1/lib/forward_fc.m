% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu


% output is the one from softmax layer
% as well as the hidden layer pre activations, and the hidden layer post
% activations
function [output, post_activation, pre_activation] = forward_fc(fc_weight, train_sample, config, debug_mode)
	if nargin < 3
		debug_mode = true;
	end

	if debug_mode
		assert(isstruct(fc_weight), 'the weight should be a struct\n');
		assert(isfield(fc_weight, 'W'), 'the weights in fully connected do not exist');
		assert(isfield(fc_weight, 'b'), 'the bias in fully connected do not exist');
	end

	% W, b are cells of matrix to store the weight and bias
	W = fc_weight.W;
	b = fc_weight.b;

	% define the activation function
	if isfield(config, 'activation')
		if strcmp(config.activation, 'sigmoid')
			activation = @mysigmoid;
		elseif strcmp(config.activation, 'relu')
			activation = @myrelu;
		elseif strcmp(config.activation, 'tanh')
			activation = @mytanh;
		else
			assert(false, sprintf('the activation function is not correct in the configuration: %s', config.activation));
		end	
	else
		fprintf('No activation function is specified. Sigmoid is used here.');
		activation = @mysigmoid;
	end

	if debug_mode
		assert(length(W) == length(b), 'the number of layers in weight and bias is not equal\n');
	end
	num_layer = length(W);		% number of layers including output layer

	if debug_mode
		for layer_index = 1:num_layer
			assert(size(W{layer_index}, 1) == size(b{layer_index}, 1), sprintf('the input number of neurons should be equal in weight and bias: %d vs %d\n', size(W{layer_index}, 1), size(b{layer_index}, 1)));
			assert(1 == size(b{layer_index}, 2), 'the second dimension of bias in neuron should be equal 1\n');
		end
	end
	pre_activation = cell(num_layer-1, 1);
	post_activation = cell(num_layer-1, 1);

	for i = 1:num_layer
		weight = W{i};			% 100 x 784
		bias = b{i};			% 100 x 1
		% size(train_sample)	% 784 x 1

		output_pre = weight * train_sample + bias;				% 100 x 1			

		% output_pre
		if i < num_layer
			pre_activation{i} = output_pre;     				% 100 x 1
		end

		% output_pre(1:10)
		output_pos = activation(output_pre);
		% output_pos(1:10)
		% pause

		if i < num_layer
			post_activation{i} = output_pos;     				% 100 x 1
		elseif i == num_layer      % output the final softmax result
			output = mysoftmax(output_pre);						% 10 x 1
			break;
		end
		train_sample = output_pos;
	end

	if debug_mode
		assert(length(pre_activation) == length(W) - 1, 'the dimension of pre-activation is not correct');
		assert(length(post_activation) == length(W) - 1, 'the dimension of post-activation is not correct');
		for layer_index = 1:num_layer-1
			assert(all(size(pre_activation{layer_index}) == size(fc_weight.b{layer_index})), 'the dimension of pre-activation is not correct');
			assert(all(size(post_activation{layer_index}) == size(fc_weight.b{layer_index})), 'the dimension of post-activation is not correct');
		end
	end

end
