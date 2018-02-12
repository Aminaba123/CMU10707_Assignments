% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

% initialize the library path
function startup()
	clc;
	close all;
	clear;

	curdir = fileparts(mfilename('fullpath'));

	% addpath(genpath(fullfile(curdir, 'utils')));
	addpath(genpath(fullfile(curdir, 'libs')));
	addpath(genpath(fullfile(curdir, 'libs', 'matlab')));
	addpath(genpath(fullfile(curdir, 'libs', 'nn')));
	addpath(genpath(fullfile(curdir, 'libs', 'io')));
	addpath(genpath(fullfile(curdir, 'libs', 'math')));
	addpath(genpath(fullfile(curdir, 'libs', 'miscellaneous')));

	% addpath(genpath(fullfile(curdir, 'xinshuo_toolbox', 'xinshuo_matlab')));
	% addpath(genpath(fullfile(curdir, 'xinshuo_toolbox', 'xinshuo_learning')));
	% addpath(genpath(fullfile(curdir, 'xinshuo_toolbox', 'xinshuo_learning', 'neural_network')));
	% addpath(genpath(fullfile(curdir, 'xinshuo_toolbox', 'xinshuo_io')));
	% addpath(genpath(fullfile(curdir, 'xinshuo_toolbox', 'xinshuo_math')));
	% addpath(genpath(fullfile(curdir, 'xinshuo_toolbox', 'xinshuo_miscellaneous')));
end