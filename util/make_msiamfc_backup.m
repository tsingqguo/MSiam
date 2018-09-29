% -------------------------------------------------------------------------------------------------
function net = make_msiamfc_backup(opts)
%MAKE_SIAMESEFC
%   Creates Siamese Fully-Convolutional network,
%   made by duplicating a vanilla AlexNet in two branches
%   and joining the branches with a cross-correlation layer
%
%   Qing Guo, 2018
% -------------------------------------------------------------------------------------------------------
net_opts.siamese = true;
net_opts.batchNormalization = true;
net_opts.strides = [2, 2, 1, 2];
net_opts = vl_argparse(net_opts, {opts.conf});

% create siamese branch of the network (5 conv layers)
fprintf('construct msiam network\n');
% four different net sizes to tradeoff accuracy/speed: vid_create_net, _small1, _small2 and _small3
branch = vid_create_net(...
    'exemplarSize',       opts.exemplarSize * [1 1], ...
    'instanceSize',       opts.instanceSize * [1 1], ...
    'batchNormalization', net_opts.batchNormalization, ...
    'networkType',        'simplenn', ...
    'weightInitMethod',   opts.init.weightInitMethod, ...
    'scale',              opts.init.scale, ...
    'initBias',           opts.init.initBias, ...
    'strides',            net_opts.strides);


branch = dagnn.DagNN.fromSimpleNN(branch);

% Add common final stream of the network
net = dagnn.DagNN();
rename_io_vars(branch, 'in', 'out');
add_triple_of_streams(net, branch,...
    {'instance_t0', 'instance_t1','instance_t2'}, ...
    {'t0_feat', 't1_feat','t2_feat'}, ...
    net_opts.siamese);
%%
net.addLayer('off_t01_l3',Off(),...
    {'a_x9','b_x9'},...
    {'t01_off_l3'});% output channels: 1152

net.addLayer('off_t01_conv3',dagnn.Conv('size',[3,3,384,384]),...
    {'t01_off_l3'},{'t01_off_l34'},{'off_conv3_weights','off_conv3_bias'});% outputs: 384

net.addLayer('off_t12_l3',Off('iscrop',true),...
    {'b_x9','c_x9'},...
    {'t12_off_l3'});

net.addLayer('off_t12_conv3',dagnn.Conv('size',[3,3,384,384]),...
    {'t12_off_l3'},{'t12_off_l34'},{'off_conv3_weights','off_conv3_bias'});

%%
net.addLayer('off_t01_l4',Off(),...
    {'a_x12','b_x12'},...
    {'t01_off_l4'});% output channels: 1152

net.addLayer('off_t01_concat34',dagnn.Concat(),...
    {'t01_off_l34','t01_off_l4'},...
    {'t01_off_l34c'});% output channels: 1536

net.addLayer('off_t01_conv4',dagnn.Conv('size',[3,3,512,768]),...
    {'t01_off_l34c'},{'t01_off_l45'},{'off_conv4_weights','off_conv4_bias'});

net.addLayer('off_t12_l4',Off('iscrop',true),...
    {'b_x12','c_x12'},...
    {'t12_off_l4'});

net.addLayer('off_t12_concat34',dagnn.Concat(),...
    {'t12_off_l34','t12_off_l4'},...
    {'t12_off_l34c'});% output channels: 1536

net.addLayer('off_t12_conv4',dagnn.Conv('size',[3,3,512,768]),...
    {'t12_off_l34c'},{'t12_off_l45'},{'off_conv4_weights','off_conv4_bias'});

%%
net.addLayer('off_t01_l5',Off(),...
    {'t0_feat','t1_feat'},...
    {'t01_off_l5'})%  output channels: 256

net.addLayer('off_t01_concat45',dagnn.Concat(),...
    {'t01_off_l45','t01_off_l5'},...
    {'t01_off_l45c'});% output channels: 768+256=1152

net.addLayer('off_t01_conv5',dagnn.Conv('size',[1,1,384,384]),...
    {'t01_off_l45c'},{'t01_off_l5o'},{'off_conv5_weights','off_conv5_bias'});

net.addLayer('off_t12_l1',Off('iscrop',true),...
    {'t1_feat','t2_feat'},...
    {'t12_off_l5'})

net.addLayer('off_t12_concat45',dagnn.Concat(),...
    {'t12_off_l45','t12_off_l5'},...
    {'t12_off_l45c'});

net.addLayer('off_t12_conv5',dagnn.Conv('size',[1,1,384,384]),...
    {'t12_off_l45c'},{'t12_off_l5o'},{'off_conv5_weights','off_conv5_bias'});

%% output
net.addLayer('xcorr', XCorr(), ...
    {'t12_off_l5o', 't01_off_l5o'}, ...
    {'xcorr_out'}, ...
    {});

add_adjust_layer(net, 'adjust', 'xcorr_out', 'score', ...
    {'adjust_f', 'adjust_b'}, 1e-3, 0, 0, 1);

% initialize the params
org_net = load('siamfcnet_gray.mat');
org_net = org_net.net;

for pi = 1:numel(org_net.params)
    net.params(net.getParamIndex(org_net.params(pi).name)).value=...
        org_net.params(pi).value;
    net.params(net.getParamIndex(org_net.params(pi).name)).learningRate=0;
end
    
for pi = 1:numel(net.params)
    switch net.params(pi).name      
        case 'off_conv3_weights'
            net.params(pi).value = init_weight(opts.init, 3, 3, 384, 384, 'single');
            net.params(pi).learningRate = 1;
        case 'off_conv3_bias'
            net.params(pi).value = zeros(384, 1, 'single');  
            net.params(pi).learningRate = 2;
        case 'off_conv4_weights'
            net.params(pi).value = init_weight(opts.init, 3, 3, 512, 768, 'single');
            net.params(pi).learningRate = 1;
        case 'off_conv4_bias'
            net.params(pi).value = zeros(768, 1, 'single');  
            net.params(pi).learningRate = 2;
        case 'off_conv5_weights'
            net.params(pi).value = init_weight(opts.init, 1, 1, 384, 384, 'single');
            net.params(pi).learningRate = 1;
        case 'off_conv5_bias'
            net.params(pi).value = zeros(384, 1, 'single');   
            net.params(pi).learningRate = 2;
    end
end

net.move('gpu');
net.mode = 'test';

end