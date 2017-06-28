function [trainedClassifier,finalResults,binomVector,NoOfSuccesses,actualNoOfKangaroos,actualNoOfNotKangaroos,totalNoOfTestImages] = NaiveBayesClassifier(trainingData,train_labels,testDataInput,testLabels)

% Extract predictors and response
% This code processes the data into the right shape for training the
Xtrain = trainingData;
Ytrain = train_labels;
Xtest = testDataInput;
Ytest = testLabels; 

%Setupt Input parameterskernelType
kernelSupport={[],[],[],[],[-20,20],[-5,5],[-10,10]}; %If a cell is empty ([]), then the software did not fit a kernel distribution to the corresponding predictor.
kernelType ={'normal','epanechnikov','triangle','box','triangle','triangle','triangle'};

%perform classification
    nbClassifier = fitcnb(Xtrain,Ytrain, 'distribution','kernel','kernel',kernelSupport{1},'kernel',kernelType{1});
    %More details on Naive Bayes can be found at http://au.mathworks.com/help/stats/naive-bayes-classification.html
    %and http://au.mathworks.com/help/stats/fitnaivebayes.html
    % Create the result struct with predict function
    nbPredictFcn = @(x) predict(nbClassifier, x);
    trainedClassifier.predictFcn = @(x) nbPredictFcn(x);

    % Add additional fields to the result struct
    trainedClassifier.NBClassification = nbClassifier;
	trainedClassifier.classifierType = 'Naive Bayes';
    trainedClassifier.classifierMethod =strcat('Kernel :',num2str(kernelSupport{1}),'Kernel Type :',kernelType{1}); 
    trainedClassifier.About = 'This struct is a trained classifier exported from Classification Learner R2016a.';
    trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedClassifier''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

    % Perform cross-validation
    partitionedModel = crossval(nbClassifier, 'KFold', 10);

    % Compute validation accuracy
    disp('Computing Validation Accuracy .....................')
    validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
    disp(validationAccuracy)
    % Compute validation predictions and scores
    disp('Computing Validation predictions .....................')
    [validationPredictions, validationScores] = kfoldPredict(partitionedModel);
   
    %display validation accuracy
    disp('Validation accuracy:')
    disp(validationAccuracy)
    %-----------------------------------------------------------------------------------------------
    % Compute resubstitution accuracy
    %resubstitutionAccuracy = 1 - resubLoss(trainedClassifier.nbClassifier, 'LossFun', 'ClassifError');

    % Compute resubstitution predictions and scores
    %[resubstitutionPredictions, resubstitutionScores] = predict(trainedClassifier.nbClassifier, predictors);
    disp('Training accuracy:')
    %disp(resubstitutionAccuracy)
    resubstitutionPredictions = {'NA'};
    resubstitutionAccuracy = 0;
    %-----------------------------------------------------------------------------------------------
   %CALL FUNCTION TO PROCESS THE TRAINED CLASSIFIER AND PRODUCE METRICS 
[finalResults,binomVector,NoOfSuccesses,actualNoOfKangaroos,actualNoOfNotKangaroos,totalNoOfTestImages]= computeClassifierResults(trainedClassifier,Ytrain,validationAccuracy,resubstitutionAccuracy,resubstitutionPredictions,Xtest,validationPredictions,Ytest)
    
   
