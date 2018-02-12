# CMU10707_HW1

This code is modularized. All forward, backward, optimizer and utility functions are separate from the main scripts.

In order to run the code, one have to run "startup.m" to load the self-contained library and then run "main.m". Also the MATLAB is required to install the "statistics and machine learning" toolbox to use the Gaussian function in the code.

All logging files are printed and stored.

###############
For different experiment settings, one just needs to modify the configuration in the first 37 lines of code.

For example, if one wants to define a two-layer network, the config.num_units2 at 16th line of code needs to be uncommented. 
If one wants to reproduce a results, the config.seed must be specified to be a seed used before (please find the seed and logging files in the path: experiments/**/**/log_***.txt)

All other parameters setting are as follows:

	Name 									Options 							Descriptions

	config.check_grad 						true or false						if check the gradient or not
	config.num_epoch 						integer								number of epochs to train
	config.num_units 						integer								number of units in the first hidden layer, required
	config.num_units2 (optional)			integer								number of units in the second hidden layer, comment it out if training a 1-layer network
	config.seed 							integer 							choose same seed to reproduce the results
	config.initialization_method 			'gaussian' or 'xavier'				method to initialize the weights

	config.train.batch_size 				integer 							only support 1
	config.train.shuffle 					true or false						if we shuffle the training data at every epoch
	config.train.optim 						'momentum' or 'sgd'					optimization method
	config.train.momentum 					float								
	config.train.lr 						float								learning rate
	config.train.weight_decay 				float								
	config.train.activation 				'sigmoid' or 'tanh' or 'relu'		activation function

	config.resume 							true or false						if using resume mode
	config.resume_file 						string								if one only wants to test a trained model, please specify the file path of the saved weights
