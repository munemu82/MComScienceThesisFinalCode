function [trainedClassifier,finalResults,binomVector,NoOfSuccesses,actualNoOfKangaroos,actualNoOfNotKangaroos,totalNoOfTestImages] =randomForestClassifier(trainingData,train_labels,testDataInput,testLabels)

% Extract predictors and response
% This code processes the data into the right shape for training the
Xtrain = trainingData;
Ytrain = train_labels;
Xtest = testDataInput;
Ytest = testLabels; 

%Prepare input parameters and required variables
noOfLearners =[5,15,30,35,45,55,75,100,125, 150];
% Train a classifier
% This code specifies all the classifier options and trains the classifier.
 classificationEnsemble = fitensemble(...
     Xtrain, ...
     Ytrain, ...
     'Bag', ...
     noOfLearners(8), ...
     'Tree', ...
     'Type', 'Classification', ...
     'ClassNames', {'Kangaroo'; 'Not Kangaroo'});

    % Create the result struct with predict function
    ensemblePredictFcn = @(x) predict(classificationEnsemble, x);
    trainedClassifier.classifierType = 'Random Forest';
    trainedClassifier.classifierMethod =strcat('Number learners :',num2str(noOfLearners(8)));
    trainedClassifier.predictFcn = @(x) ensemblePredictFcn(x);

    % Add additional fields to the result struct
    trainedClassifier.ClassificationEnsemble = classificationEnsemble;
    trainedClassifier.About = 'This struct is a trained classifier exported from Classification Learner R2016a.';
    trainedClassifier.featureTrainedOn = 'HOG';
    trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedClassifier''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

    % Perform cross-validation
    partitionedModel = crossval(trainedClassifier.ClassificationEnsemble, 'KFold', 10);

    % Compute validation accuracy
    validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');

    % Compute validation predictions and scores
    [validationPredictions, validationScores] = kfoldPredict(partitionedModel);

    %display validation accuracy
    disp('Validation accuracy:')
    disp(validationAccuracy)
    %-----------------------------------------------------------------------------------------------
    % Compute resubstitution accuracy
    resubstitutionAccuracy = 1 - resubLoss(trainedClassifier.ClassificationEnsemble, 'LossFun', 'ClassifError');

    % Compute resubstitution predictions and scores
    [resubstitutionPredictions, resubstitutionScores] = predict(trainedClassifier.ClassificationEnsemble, Xtrain);
    disp('Training accuracy:')
    disp(resubstitutionAccuracy)
%-----------------------------------------------------------------------------------------------
%CALL FUNCTION TO PROCESS THE TRAINED CLASSIFIER AND PRODUCE METRICS 
[finalResults,binomVector,NoOfSuccesses,actualNoOfKangaroos,actualNoOfNotKangaroos,totalNoOfTestImages]= computeClassifierResults(trainedClassifier,Ytrain,validationAccuracy,resubstitutionAccuracy,resubstitutionPredictions,Xtest,validationPredictions,Ytest)
    