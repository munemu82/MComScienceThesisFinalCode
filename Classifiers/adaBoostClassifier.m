function [trainedClassifier,finalResults,binomVector,NoOfSuccesses,actualNoOfKangaroos,actualNoOfNotKangaroos,totalNoOfTestImages]= adaBoostClassifier(trainingData,train_labels, testDataInput,testLabels)

% Extract predictors and response
% This code processes the data into the right shape for training the
Xtrain = trainingData;
Ytrain = train_labels;
Xtest = testDataInput;
Ytest = testLabels; 

%Set parameters to start with
listOFLearners =[30,30,35,35,35,35];  %original  listOFLearners =[30,30,35,35,35,35];
listOFMaximumSplitValues=[20,25,25,20,35,15];
%adaBoostMethodType={'LogitBoost','GentleBoost','AdaBoostM1','RUSBoost','LPBoost','TotalBoost','RobustBoost','Subspace'};
adaBoostMethodType={'Subspace','LogitBoost','GentleBoost','AdaBoostM1','RUSBoost'};
     template = templateTree(...
        'MaxNumSplits', listOFMaximumSplitValues(2));
        classificationEnsemble = fitensemble(...
        Xtrain, ...
        Ytrain, ...
        adaBoostMethodType{2}, ...
        listOFLearners(2), ...   
        template, ...
        'Type', 'Classification', ...
        'ClassNames', {'Kangaroo'; 'Not Kangaroo'});
% end

    % Create the result struct with predict function
    ensemblePredictFcn = @(x) predict(classificationEnsemble, x);
    trainedClassifier.predictFcn = @(x) ensemblePredictFcn(x);

    % Add additional fields to the result struct
    trainedClassifier.ClassificationEnsemble = classificationEnsemble;
    trainedClassifier.classifierType = 'AdaBoost';
    trainedClassifier.featureTrainedOn = 'SURF';
    trainedClassifier.classifierMethod =strcat('Method :',adaBoostMethodType{2},'No. Of Learners :',num2str(listOFLearners(2)),'Max splits', num2str(listOFMaximumSplitValues(2))); 
    trainedClassifier.About = 'This struct is a trained classifier exported from Classification Learner R2016a.';
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
    % Compute resubstitution accuracy
    resubstitutionAccuracy = 1 - resubLoss(trainedClassifier.ClassificationEnsemble, 'LossFun', 'ClassifError');
   % Compute resubstitution predictions and scores
    [resubstitutionPredictions, resubstitutionScores] = predict(trainedClassifier.ClassificationEnsemble, Xtrain);
    disp('Training accuracy:')
    disp(resubstitutionAccuracy)
   
%CALL FUNCTION TO PROCESS THE TRAINED CLASSIFIER AND PRODUCE METRICS 
[finalResults,binomVector,NoOfSuccesses,actualNoOfKangaroos,actualNoOfNotKangaroos,totalNoOfTestImages]= computeClassifierResults(trainedClassifier,Ytrain,validationAccuracy,resubstitutionAccuracy,resubstitutionPredictions,Xtest,validationPredictions,Ytest)