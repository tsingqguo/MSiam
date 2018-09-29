% -------------------------------------------------------------------------------------------------
function net = make_msiamfc(p)
%MAKE_SIAMESEFC
%   Creates Siamese Fully-Convolutional network,
%   made by duplicating a vanilla AlexNet in two branches
%   and joining the branches with a cross-correlation layer
%
%   Qing Guo, 2018
% -------------------------------------------------------------------------------------------------------

basepath = './model/';
netpath = fullfile(basepath,p.netm) ;
trainResults = load(netpath);
net = trainResults.net;

% load as DAGNN
net = dagnn.DagNN.loadobj(net);
net = remove_layers_from_block(net, 'dagnn.Loss');

net.move('gpu');
net.mode = 'test';

end