tic
info = load('images/DayImageDataSet/dayImagesFilelist.mat');
datasets = {'Kangaroo'}; 
train_lists = {info.train_list};
test_lists = {info.test_list};
feature = 'SURF';              %set type of feature
dataset = 'dayImages';

%Construct an array of image sets based on the following categories
imageFolder ='images/DayImageDataSet/';
%imgSets = [ imageSet(fullfile('SceneData')), ...
imgSets = [ imageSet(fullfile(imageFolder, 'train')), ... 
            imageSet(fullfile(imageFolder, 'train')), ...
            imageSet(fullfile(imageFolder, 'test'))];
{imgSets.Description } % display all labels on one line
[imgSets.Count]   % show the corresponding count of image

% %create bag of visual features for the data set - 'trainig set'
bag = bagOfFeatures(imgSets(1), 'VocabularySize',5376, 'StrongestFeatures', 0.60, 'Upright',false);
%create bag of visual features for the data set - 'test set'
bag4testImages = bagOfFeatures(imgSets(3), 'VocabularySize',5376, 'StrongestFeatures', 0.60, 'Upright',false);

train_features = [];
test_features =[];

%computing SURF features for the training dataset
for j = 1:imgSets(1).Count
	    img = read(imgSets(1),j);
	    featureVector = encode(bag, img);
	    train_features = vertcat(train_features,featureVector);
end
%computing SURF features for the test dataset
for j = 1:imgSets(3).Count
	    img = read(imgSets(3),j);
	    testFeatureVector = encode(bag4testImages, img);
	    test_features = vertcat(test_features,testFeatureVector);
end

% %transforming SURF feature vectors from array to a table
trainingFeaturesDataTable = array2table(train_features);
%append class labels to the training features data table.
trainingFeaturesDataTable.label = info.train_labels;   %adding class label
%train_featuresTable.label = newLabel;   %adding class label - just for test

%transforming SURF feature vectors from array to a table
testFeaturesDataTable = array2table(test_features);
%append class labels to the ttest features data table.
testFeaturesDataTable.label = info.test_labels;   %adding class label

toc