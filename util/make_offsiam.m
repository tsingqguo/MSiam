% -------------------------------------------------------------------------------------------------
function  net = make_offsiam()
% OFF
%   Crosscorrelates two activations of different size exploiting  the API of vl_nnconv
%
%   Qing Guo, 2018
% -------------------------------------------------------------------------------------------------------------------------

net = dagnn.DagNN();
% OFF for template z
net.addLayer('off_z',Off(),{'zt','z1'},{'zt_off'},{});
% OFF for template x
net.addLayer('off_x',Off(),{'x1','xt'},{'xt_off'},{});
% fully convolution on OFF
net.addLayer('corr',XCorr(),{'zt_off','xt_off'},{'xcorr_off'});
% output
add_adjust_layer(net, 'adj_off', 'xcorr_off', 'score', ...
    {'adj_f_off', 'adj_b_off'}, 1e-3, 0, 0, 1);
net.move('gpu');
net.mode = 'test';
end