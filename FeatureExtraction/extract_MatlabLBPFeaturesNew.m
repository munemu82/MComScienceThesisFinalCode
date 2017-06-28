tic
addpath(genpath('.'));
info = load('images/DayImageDataSet/dayImagesFilelist.mat');
datasets = {'Kangaroo'}; 
feature = 'LBP';              %set type of feature
dataset = 'dayImages';

%Get list of images path
train_lists = {info.train_list};
test_lists = {info.test_list};

%Placeholder for feature vectors
train_features = [];
test_features = [];
newLBPFeatureVector=[];

%LBP features configuration settings
cellSize=[12 12];  % other options tried [12 12] and [56 56]   but the final choice was [44 44]
cellSize4Smals =[12 12];    
noOfNeigbors = 8;  %other options tried 8, and 24      but the final choice was 16
featureSize =1001;
radiusVal = 1;  % other options tried 1 and 5     but the final choice was 3
uprightRotFlag=false;
txt = 'Now computing LBP features for - ';  %status Message for each image

%initialize training features set with the first image from the list
firstImage = imread(train_lists{1}{1});

disp('Begin computing LBP features for training images')
disp('------------------------------------------------')
%extract LBP features from training images
for i=1:length(train_lists{1})
%for i=1:10
        %read image from training list
        newImg = imread(train_lists{1}{i});
        %extract features from the image
      newLBPFeatureVector= extractLBPFeatures(newImg,'CellSize',cellSize,'NumNeighbors',noOfNeigbors,'Radius',radiusVal, 'Upright',uprightRotFlag);
      disp(strcat(txt,train_lists{1}{i}))     %display status message
      %Draw random sample (without replacement) of 1000 feature points from the original LBP
      randsampl = datasample(newLBPFeatureVector,1000,2,'Replace',false);
      %assign the one-diminsional vector (sampled) to two the two-dimensional feature
      %vectors
      train_features =cat(1, train_features, randsampl);
       
end
disp('LBP feature Extraction for training images completed successfully')
disp('Begin computing LBP features for test images')
disp('------------------------------------------------')
%extract LBP features from training images
for j=1:length(test_lists{1})
%for j=1:15
       %read image from training list
       newImg = imread(test_lists{1}{j});
       %extract features from the image      
       newLBPFeatureVector2= extractLBPFeatures(newImg,'CellSize',cellSize,'NumNeighbors',noOfNeigbors,'Radius',radiusVal, 'Upright',uprightRotFlag);
       disp(strcat(txt,test_lists{1}{j}))     %display status message
        
      %Draw random sample (without replacement) of 1000 feature points from the original LBP
      randsampl2 = datasample(newLBPFeatureVector2,1000,2,'Replace',false);
      %assign the one-diminsional vector (sampled) to two the two-dimensional feature
      %vectors
       test_features =cat(1, test_features, randsampl2);
end
disp('LBP feature Extraction for test images completed successfully')
disp('LBP feature Extraction completed successfully')
%add features to the table 
trainingFeaturesDataTable = array2table(train_features);
trainingFeaturesDataTable.label = info.train_labels;
testFeaturesDataTable = array2table(test_features);
testFeaturesDataTable.label = info.test_labels;
toc

