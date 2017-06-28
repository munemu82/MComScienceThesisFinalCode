%Feature extraction
tic
addpath(genpath('.'));
info = load('images/DayImageDataSet/dayImagesFilelist.mat');
datasets = {'Kangaroo'}; 
train_lists = {info.train_list};
test_lists = {info.test_list};
feature = 'gist';              %set type of feature
dataset = 'dayImages';

% Load the configuration and set dictionary size to 20 (for fast demo)
c = conf();
%c.feature_config.(feature).dictionary_size=32;

% Compute train and test features
datasets_feature(datasets, train_lists, test_lists, feature, c);

% Load train and test features
train_features = load_feature(datasets{1}, feature, 'train', c);
test_features = load_feature(datasets{1}, feature, 'test', c);
%save training features into table structure for easy export to csv later
trainingFeaturesDataTable = array2table(train_features);
%trainingFeaturesDataTable.label = info.train_labels_cat;
trainingFeaturesDataTable.label = info.train_labels;

%save testing features into table structure for easy export to csv later
testFeaturesDataTable = array2table(test_features);
testFeaturesDataTable.label = info.test_labels;

%features table.
% for k=1:length(testFeaturesDataTable.Properties.VariableNames)
%     testFeaturesDataTable.Properties.VariableNames{k}= strcat('train_features',num2str(k));
% end
%FEATURE TRANSFORMATION
%compute transformed features - we use 1 / (1 + exp(x)) transformation
% transformed_train_features = 1 ./(1+exp(train_features));
% transformed_test_features = 1 ./(1+exp(test_features));
% %convert transformed features matrix into table
% transformedTrainingSet = array2table(transformed_train_features);
% trainingFeaturesDataTable.label = info.train_labels_cat;   %adding class label
% transformedTestSet = array2table(transformed_test_features);
toc

