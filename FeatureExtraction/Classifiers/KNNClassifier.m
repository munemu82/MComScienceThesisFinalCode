function [trainedClassifier,binomVector,finalResults,NoOfSuccesses,actualNoOfKangaroos,actualNoOfNotKangaroos,totalNoOfTestImages] =KNNClassifier(trainingData,train_labels,testDataInput,testLabels)

% Extract predictors and response
% This code processes the data into the right shape for training the
Xtrain = trainingData;
Ytrain = train_labels;
Xtest = testDataInput;
Ytest = testLabels; 

%Set parameters to start with
valueOfK =[1,10,10,10,10,3,7,10];
distanceFunction={'Euclidean','Euclidean','Cosine','Minkowski','Euclidean','Euclidean','Euclidean','Euclidean'};
distanceWeight={'Equal','Equal','Equal','Equal','SquaredInverse','Equal','Equal','Equal'};

% This code specifies all the classifier options and trains the classifier.
classificationKNN = fitcknn(...
    Xtrain, ...
    Ytrain, ...
    'Distance', distanceFunction{5}, ...
    'Exponent', [], ...
    'NumNeighbors', valueOfK(5), ...
    'DistanceWeight', distanceWeight{5}, ...
    'Standardize', true, ...
    'ClassNames', {'Kangaroo'; 'Not Kangaroo'});

% Create the result struct with predict function
knnPredictFcn = @(x) predict(classificationKNN, x);
trainedClassifier.classifierType = 'KNN';
trainedClassifier.classifierMethod =strcat('Distance function:',distanceFunction{5},'No. K:',num2str(valueOfK(5)),'Distance Weight:',distanceWeight{5})
trainedClassifier.predictFcn = @(x) knnPredictFcn(x);

% Add additional fields to the result struct
trainedClassifier.ClassificationKNN = classificationKNN;
trainedClassifier.About = 'This struct is a trained classifier exported from Classification Learner R2016a.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedClassifier''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationKNN, 'KFold', 10);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');

% Compute validation predictions and scores
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

%display validation accuracy
disp('Validation accuracy:')
disp(validationAccuracy)

% Compute resubstitution accuracy
resubstitutionAccuracy = 1 - resubLoss(trainedClassifier.ClassificationKNN, 'LossFun', 'ClassifError');

% Compute resubstitution predictions and scores
[resubstitutionPredictions, resubstitutionScores] = predict(trainedClassifier.ClassificationKNN, Xtrain);
disp('Training accuracy:')
disp(resubstitutionAccuracy)
%CALL FUNCTION TO PROCESS THE TRAINED CLASSIFIER AND PRODUCE METRICS 
[finalResults,binomVector,NoOfSuccesses,actualNoOfKangaroos,actualNoOfNotKangaroos,totalNoOfTestImages]= computeClassifierResults(trainedClassifier,Ytrain,validationAccuracy,resubstitutionAccuracy,resubstitutionPredictions,Xtest,validationPredictions,Ytest)
