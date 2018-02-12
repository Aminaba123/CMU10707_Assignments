% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

% initialize the library path
function startup()
	clc;
	close all;
	clear;

	curdir = fileparts(mfilename('fullpath'));

	% libdir = '/home/xinshuo/lib';
	% addpath(genpath(fullfile(libdir, 'xinshuo_toolbox', 'matlab')));
	% addpath(genpath(fullfile(libdir, 'xinshuo_toolbox', 'file_io')));
	addpath(genpath(fullfile(curdir, 'lib')));
	addpath(genpath(fullfile(curdir, 'utils')));
end