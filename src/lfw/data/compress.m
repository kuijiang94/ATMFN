% =========================================================================
% Test code for Super-Resolution Convolutional Neural Networks (SRCNN)
%
% Reference
%   Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. Learning a Deep Convolutional Network for Image Super-Resolution, 
%   in Proceedings of European Conference on Computer Vision (ECCV), 2014
%
%   Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. Image Super-Resolution Using Deep Convolutional Networks,
%   arXiv:1501.00092
%
% Chao Dong
% IE Department, The Chinese University of Hong Kong
% For any question, send email to ndc.forward@gmail.com
% =========================================================================

close all;
clear;
clc;

path1='E:\jiangkui\shiyan\face\GAN\src\lfw\data\lfw128\test\';
%path1='E:\jiangkui\shiyan\face\GAN\src\test\test\';
path2='E:\jiangkui\shiyan\face\GAN\src\lfw\data\lfw128_com100\test\';
%path2='E:\jiangkui\shiyan\face\GAN\src\test\test_com100_lr8\';
mkdir(path2);
list= dir(strcat(path1,'*.jpg'));
for i=1:length(list)
    img1_name=list(i).name;
    img1=imread(strcat(path1,img1_name));
    name=img1_name(1:end-4);

%img2 = single(im2double(img1));% + 25/255.0*randn(size(img1)));
    %% bicubic interpolation
%im_b = imresize(im_l, up_scale, 'bicubic');
%path = strcat(path2,img1_name);
im_l = imresize(img1, 1/8, 'bicubic');
imwrite(uint8(im_l), [path2,'/',img1_name], 'Quality',100)%20,30£¬50£¬75£¬100
%imwrite(img1,[path2,'/',img1_name]);

end
