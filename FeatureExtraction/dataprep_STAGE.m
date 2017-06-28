%AUTHOR: Amos Munezero
%This script is used to prepare images for feature extraction

%Initial image folder setup
trainFolder = 'images/trainTemp'; %folder containing color training images
testFolder = 'images/testTemp';   %folder containing color test images

%validate specified training and test folders if exit
if ~isdir(trainFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', trainFolder);
  uiwait(warndlg(errorMessage));
  return;
end
if ~isdir(testFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', testFolder);
  uiwait(warndlg(errorMessage));
  return;
end

%extract trainingg images from folders 
%extract trainingg images from folders 
trainFilePattern1 = fullfile(trainFolder, '*.jpg'); trainFilePattern2 = fullfile(trainFolder, '*.png');   % to match images with .jpg and .png extensions
jpgTrainingImages= dir(trainFilePattern1);    pngTrainingImages= dir(trainFilePattern2);            %extract training images with jpg and .png
trainingImages = vertcat(pngTrainingImages, jpgTrainingImages); %combine list of .jpg and png images
    
 testFilePattern1 = fullfile(testFolder, '*.png');  testFilePattern2 = fullfile(testFolder, '*.jpg');
 pngTestingImages = dir(testFilePattern1);    jpgTestingImages = dir(testFilePattern2);
 testingImages = vertcat(pngTestingImages, jpgTestingImages); %combine list of .jpg and png images
 

%Declare lists
train_list = {};
%train_labels_cat = [];
train_labels= [];
test_list = {};
% test_labels_cat = [];
% test_labels = [];
% test_num_labels =[];
% classes = {1,2};
%Get all the image names from training and test image folders

%extract and store training image names and path to the list
for i = 1:length(trainingImages)
  baseTrainImageName = trainingImages(i).name;
  fullTrainImageName = fullfile(trainFolder, baseTrainImageName);
  
  %Display training images
  fprintf(1, 'Now reading %s\n', fullTrainImageName);
  imageArray = imread(fullTrainImageName);
  %Convert image into gray scale
  fprintf(1, 'Now converting %s\n', fullTrainImageName, ' to grayscale');
  grayedImg = rgb2gray(imageArray);
  %perform histogram equalization on grayed image
  fprintf(1, 'Now perform histograming on equalization on %s\n', fullTrainImageName);
  histEqImg = histeq(grayedImg);
  %display figure 
  %figure
  %imshow(imageArray);  % Display imageData preparation:.
  %subplot(2,2,1);imshow(imageArray);title('Original Image');
  %subplot(2,2,2);imhist(rgb2gray(imageArray));
  %subplot(2,2,3);imshow(histEqImg);title('Image after histogram equalization');
  %subplot(2,2,4);imhist(histEqImg);
  %drawnow; % Force display to update immediately.
  %save training histogram equalized image into a training dataset folder
  newFullTrainImageName = strrep(fullTrainImageName, 'trainTemp', 'train');
  fprintf(1, 'Now saving histogrammed image %s\n',' to train folder');
  imwrite(histEqImg,newFullTrainImageName);
  %write to log file
  header = 'Training Image Log File!';
  fid=fopen('train.txt','w');
  fprintf(fid, [ header '\n']);
  fprintf(fid, '%f %f \n', newFullTrainImageName);
  fclose(fid);
  
  %add processed image into the training list
  train_list{i} = newFullTrainImageName;
 end

%extract and store testing image names and path to the list
for j = 1:length(testingImages)
  baseTestImageName = testingImages(j).name;
  fullTestImageName = fullfile(testFolder, baseTestImageName);
  
  %Display Test images
  fprintf(1, 'Now reading %s\n', fullTestImageName);
  testImgArray = imread(fullTestImageName);
  %Convert image into gray scale
  testGrayedImg = rgb2gray(testImgArray);
  %perform histogram equalization on grayed image
  testHistEqImg = histeq(testGrayedImg);
  %display figure 
%   figure
%   %imshow(imageArray);  % Display imageData preparation:.
%   subplot(2,2,1);imshow(testImgArray);title('Original Image');
%   subplot(2,2,2);imhist(testGrayedImg);
%   subplot(2,2,3);imshow(testHistEqImg);title('Image after histogram equalization');
%   subplot(2,2,4);imhist(testHistEqImg);
%   drawnow; % Force display to update immediately.
  newFullTestImageName = strrep(fullTestImageName, 'testTemp', 'test');
  imwrite(testHistEqImg,newFullTestImageName);
  %write to a log file
   header = 'Training Image Log File!';
  fid=fopen('test.txt','w');
  fprintf(fid, [ header '\n']);
  fprintf(fid, '%f %f \n', newFullTestImageName);
  fclose(fid);
  %add processed image into the training list
  test_list{j} = newFullTestImageName;
end 

%Save important variables to .mat file
save('filelist.mat','train_list','test_list','train_labels','test_labels','classes','train_labels','train_num_labels','test_num_labels')

