%set the root folder of images
rootFolder ='images/DayImageDataSet';
%Set destination folder
destFolder ='images/DayImageDataSet/dayImagesFilelist.mat';
dataset ='DayImagesData';
%Categories of images in the root folder - each category must exist as a folder in the root folder
categories = {'Kangaroos', 'Background'};

%Specify number of images to be allocated to training set from each image category
numOfTrainingImages = 260;  %this can be a percent e.g 0.6 (i.e. 60%) or integer to be used for spliting images into training and test sets
%260 for night and day images
%425 for the original images
%Add path where the functions exist
%addpath(genpath('C:\Users\a582142\Documents\AMOS\UNE\2016\ThesisImplementation\MComScienceThesis2016\Feature Extraction'));
%call processImage function 
[trainingSet,testSet,trainLabels,testLabels,train_labels,train_num_labels,test_labels,test_num_labels] = processImages(rootFolder,destFolder,categories,numOfTrainingImages)

%PREPARE DATA FOR FEATURE EXTRACTION
 
% Initial image folder setup
%Initial image folder setup
trainFolder = 'images/DayImageDataSet/trainTemp'; %folder containing color training images
testFolder = 'images/DayImageDataSet/testTemp';   %folder containing color test images

%extract trainingg images from folders 
trainFilePattern1 = fullfile(trainFolder, '*.jpg'); trainFilePattern2 = fullfile(trainFolder, '*.png');   % to match images with .jpg and .png extensions
jpgTrainingImages= dir(trainFilePattern1);    pngTrainingImages= dir(trainFilePattern2);            %extract training images with jpg and .png
trainingImages = vertcat(pngTrainingImages, jpgTrainingImages); %combine list of .jpg and png images
    
 testFilePattern1 = fullfile(testFolder, '*.png');  testFilePattern2 = fullfile(testFolder, '*.jpg');
 pngTestingImages = dir(testFilePattern1);    jpgTestingImages = dir(testFilePattern2);
 testImages = vertcat(pngTestingImages, jpgTestingImages); %combine list of .jpg and png images
 
 %Image path list matrix varibales
 train_list = {};
 test_list = {};
 classes = {1,2};
 
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
for j = 1:length(testImages)
  baseTestImageName = testImages(j).name;
  fullTestImageName = fullfile(testFolder, baseTestImageName);
  
  %Display Test images
  fprintf(1, 'Now reading %s\n', fullTestImageName);
  testImgArray = imread(fullTestImageName);
  %Convert image into gray scale
  testGrayedImg = rgb2gray(testImgArray);
  %perform histogram equalization on grayed image
  testHistEqImg = histeq(testGrayedImg);
  
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
  %TRANSPOSE REQUIRED VARIABLES FOR FEATURE EXTRACTION
   train_labels = train_labels';
   train_num_labels = train_num_labels';
   test_labels = test_labels';
   test_num_labels = test_num_labels';
        
   %SAVE REQUIRED VARIABLES FOR FEATURE EXTRACTION
   save(destFolder,'train_list','test_list','train_labels','test_labels','classes','train_num_labels','test_num_labels','dataset')
