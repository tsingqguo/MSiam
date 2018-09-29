% -------------------------------------------------------------------------------------------------------------------------
classdef Off < dagnn.Layer
%XCORR
%   Crosscorrelates two activations of different size exploiting  the API of vl_nnconv
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------------------------------
    properties
        iscrop = false
        opts = {'cuDNN'}
    end

    methods
        function outputs = forward(obj, inputs, params)
            assert(numel(inputs) == 2, 'two inputs are needed');

            x1 = inputs{2}; % instance t-1
            x2 = inputs{1}; % instance t

            assert(size(x1,1) == size(x2,1), 'x1 and x2 have the same size');
            sobelx = repmat(single([-1 0 1; -1 0 1;-1 0 1]),1,1,1,size(x1,3));
            sobely = repmat(single([1 1 1;0 0 0;-1 -1 -1]),1,1,1,size(x1,3));
            sobelx = gpuArray(single(sobelx));
            sobely = gpuArray(single(sobely));
            
            x2_gradx = vl_nnconv(x2, sobelx, [],'pad',1);
            x2_grady = vl_nnconv(x2, sobely, [],'pad',1);
            x2_difft = x2-x1;
            
            outputs{1} = vl_nnconcat({x2_gradx,x2_grady,x2_difft},3);
            
            if obj.iscrop
               rect = [8,8,8,8];
               outputs{1} = vl_nncrop(outputs{1}, rect) ; 
            end
            
        end

        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            
        end

        function outputSizes = getOutputSizes(obj, inputSizes)
            x1_sz = inputSizes{1};
            y_sz = x1_sz;
            if obj.iscrop
               rect = [8,8,8,8];
               y_sz(1) = y_sz(1)-rect(1)-rect(2);
               y_sz(2) = y_sz(2)-rect(2)-rect(2);
            end
            outputSizes = {y_sz};
        end

        function rfs = getReceptiveFields(obj)
            rfs(1,1).size = [inf inf]; % could be anything
            rfs(1,1).stride = [1 1];
            rfs(1,1).offset = 1;
            rfs(2,1).size = [inf inf];
            rfs(2,1).stride = [1 1];
            rfs(2,1).offset = 1;
        end

        function obj = Off(varargin)
            obj.load(varargin);
        end

    end

end
