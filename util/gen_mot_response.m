                
function mot_response = gen_mot_response(net_m,x_imgs,z_imgs,targetPosition_m,targetPosition,targetSize_m,s_x_m,s_x,s_z_m,avgChans,stats,p)

    x_crops_m(:,:,:,1) = make_scale_pyramid(x_imgs{1}, targetPosition_m, s_x_m, p.instanceSize, avgChans, stats, p);
    x_crops_m(:,:,:,2) = make_scale_pyramid(x_imgs{2}, targetPosition_m, s_x_m, p.instanceSize, avgChans, stats, p);
    z_crops_m(:,:,:,1) = get_subwindow_tracking(z_imgs{1}, targetPosition_m, [p.exemplarSize p.exemplarSize], ...
        [round(s_z_m) round(s_z_m)], avgChans);
    z_crops_m(:,:,:,2) = get_subwindow_tracking(z_imgs{2}, targetPosition_m, [p.exemplarSize p.exemplarSize], ...
        [round(s_z_m) round(s_z_m)], avgChans);

    net_m.eval({'pair1_I1', z_crops_m(:,:,:,1), 'pair1_I2', z_crops_m(:,:,:,2),'pair2_I1',...
        x_crops_m(:,:,:,1), 'pair2_I2', x_crops_m(:,:,:,2)});

    mot_response = gpuArray(single(net_m.vars(94).value));
    mot_response = imresize(mot_response, p.responseUp, 'bicubic');
    [r_max, c_max] = find(mot_response == max(mot_response(:)), 1);
    [r_max, c_max] = avoid_empty_position(r_max, c_max, p);
    pos_max = [r_max, c_max];
    disp_instanceFinal = pos_max - ceil(p.scoreSize*p.responseUp/2);
    disp_instanceInput = disp_instanceFinal * p.totalStride/ p.responseUp;
    disp_instanceFrame = disp_instanceInput * s_x_m / p.instanceSize;
    targetPosition_m = targetPosition_m + disp_instanceFrame;
    diff_m = (targetPosition_m-targetPosition).*p.instanceSize./(s_x.*p.totalStride);
    response_sz_m = targetSize_m.*p.instanceSize./(s_x.*p.totalStride);
    response_pos_m = diff_m+p.scoreSize./2;
    
    rows = ceil(response_pos_m(1)-response_sz_m(1)./2):ceil(response_pos_m(1)+response_sz_m(1)./2);
    cols = ceil(response_pos_m(2)-response_sz_m(2)./2):ceil(response_pos_m(2)+response_sz_m(2)./2);
    
    rows(rows>p.scoreSize|rows<1)=[];
    cols(cols>p.scoreSize|cols<1)=[];
    
    [response_row,response_col] = ndgrid(rows,cols);
    
    mot_response = single(zeros(p.scoreSize,p.scoreSize));
    mot_response(response_row(:),response_col(:)) = 1;
    mot_response = gpuArray(mot_response);
    
end


function [r_max, c_max] = avoid_empty_position(r_max, c_max, params)
    if isempty(r_max)
        r_max = ceil(params.scoreSize/2);
    end
    if isempty(c_max)
        c_max = ceil(params.scoreSize/2);
    end
end