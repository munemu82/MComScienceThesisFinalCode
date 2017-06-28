%img = imread('387.png');
grayedImg = rgb2gray(img);
histEqImg = histeq(grayedImg);

%plot show original image
% imshow(img);
% %plot show histogram equalized image
% imshow(histEqImg);
% %plot show histogram of original image
% imhist(grayedImg);
% %plot show histogram of histogram equalized image
% imhist(histEqImg);

figure
subplot(2,2,1)       % add first plot in 2 x 2 grid
imshow(img)            % Original image plot
title('Original Image')

subplot(2,2,2)       % add second plot in 2 x 2 grid
imhist(grayedImg)% Histogram Equalized Image plot
title('Histogram of original Image')

subplot(2,2,3)       % add third plot in 2 x 2 grid
imshow(histEqImg)       % Histogram equalized Image plot
title('Histogram equalized Image')

subplot(2,2,4)       % add fourth plot in 2 x 2 grid
imhist(histEqImg) %Histogram Equalized Image plot
title('Histogram of histogram equalized Image')