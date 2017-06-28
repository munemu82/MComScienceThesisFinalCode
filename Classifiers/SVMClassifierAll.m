function [allTrainedClassifiers, validationAccuracy,allConfMatrix,finalSVMModelsOutput,finalTestResultsforModels] = SVMClassifierAll(trainingData,testDataInput,testLabels)

% Extract predictors and response
% This code processes the data into the right shape for training the
% classifier.
inputTable = trainingData;
%Extract features/predictors from inputTable
predictorData = inputTable(:,1:end-1);
predictorNames = predictorData.Properties.VariableNames;
predictors = inputTable(:, predictorNames);
response = inputTable.label;

%prepare input parameters
kernelFunctions ={'linear','polynomial','polynomial','gaussian','gaussian','rbf'};
polynomialOrders ={[],2,3,[],[],[]};
boxConstraints =[5,5,5,5,5,Inf];
kernelScale ={'auto','auto','auto',23,91,'auto'};


%Output variables
finalSVMModelsOutput =[];
finalTestResultsforModels =[];
allConfMatrix={};
allTrainedClassifiers={}
% Train a classifier
%LOOP to create models for Naive Bayes
for i=1:length(kernelFunctions)
    disp('Now training a classifier for the individual models');
    classifierOutput =[]; 			%place holder for the current iteration
    % This code specifies all the classifier options and trains the classifier.
    classificationSVM = fitcsvm(...
        predictors, ...
        response, ...
        'KernelFunction', kernelFunctions{i}, ...
        'PolynomialOrder', polynomialOrders{i}, ...
        'KernelScale', kernelScale{i}, ...
        'BoxConstraint', boxConstraints(i), ...
        'Standardize', true, ...
        'ClassNames', {'Kangaroo'; 'Not Kangaroo'});

    % Create the result struct with predict function
    predictorExtractionFcn = @(t) t(:, predictorNames);
    svmPredictFcn = @(x) predict(classificationSVM, x);
    trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

    % Add additional fields to the result struct
    trainedClassifier.RequiredVariables =predictorNames;
    trainedClassifier.ClassificationSVM = classificationSVM;
    trainedClassifier.About = 'This struct is a trained classifier exported from Classification Learner R2016a.';
    trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedClassifier''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

    % Perform cross-validation
    partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 10);

    % Compute validation accuracy
    validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');

    % Compute validation predictions and scores
    [validationPredictions, validationScores] = kfoldPredict(partitionedModel);

    display validation accuracy
    disp('Validation accuracy:')
    disp(validationAccuracy)
    %training confusion matrix
    confMatrix4validation = confusionmat(response,validationPredictions);
    disp('Validation Confusion Matrix:')
    disp(confMatrix4validation)
    disp('----------------------------------------------------------------------------------------')
    truePositiveRate = confMatrix4validation(1) /(confMatrix4validation(1)+confMatrix4validation(3));
    falsePositiveRate =confMatrix4validation(2) /(confMatrix4validation(2)+confMatrix4validation(4));
    %-----------------------------------------------------------------------------------------------

    %add model and its properties to the current temp place holder vector
    disp('Add trained model to the final SVM models place holder');

   classifierOutput =[i,kernelFunctions{i},strcat('[',strtrim(cellstr(num2str(polynomialOrders{i}))),']'), kernelScale(i),boxConstraints(i),truePositiveRate,falsePositiveRate,validationAccuracy]

    %add model to the final NB Models output matrix
    disp('Add trained model to the final SVM models place holder');
    finalSVMModelsOutput =vertcat(finalSVMModelsOutput,classifierOutput)
    
    disp('----------------------------------------------------------------------------------------');
    % Compute resubstitution accuracy
    resubstitutionAccuracy = 1 - resubLoss(trainedClassifier.ClassificationSVM, 'LossFun', 'ClassifError');

    % Compute resubstitution predictions and scores
    [resubstitutionPredictions, resubstitutionScores] = predict(trainedClassifier.ClassificationSVM, predictors);
    disp('Training accuracy:')
    disp(resubstitutionAccuracy)
    %training confusion matrix
    confMatrix4training = confusionmat(response,resubstitutionPredictions);
    disp('Training Confusion Matrix:')
    disp(confMatrix4training)
    disp('----------------------------------------------------------------------------------------')
    %------------------------------------------------------------------------------------------------------
    %Prepare test data
    testResults = [];
    testdata = testDataInput;
    testdata.Properties.VariableNames = trainingData.Properties.VariableNames;
    
    testdata = testdata(:,1:end-1);
    %compute predictions for test data (data classifier never seen before)
    disp('Now computing predictions for the test dataset...........')
    testPredictions = trainedClassifier.predictFcn(testdata);
    %Computing test confusion matrix
    confMatrix4test = confusionmat(testLabels,testPredictions);
    %compute test accuracy
    truePositiveAndNegatives = confMatrix4test(1) + confMatrix4test(4);
    testAccuracy = truePositiveAndNegatives / numel(testLabels);
    disp(testAccuracy)
    truePositiveRate3 = confMatrix4test(1) /(confMatrix4test(1)+confMatrix4test(3));
    falsePositiveRate3 =confMatrix4test(2) /(confMatrix4test(2)+confMatrix4test(4));
    testResults =[i,truePositiveRate,falsePositiveRate,validationAccuracy...
    truePositiveRate3,falsePositiveRate3,testAccuracy]
    finalTestResultsforModels =vertcat(finalTestResultsforModels,testResults)
    disp('Test Confusion Matrix:')
    disp(confMatrix4test)
    allConfMatrix{i} = confMatrix4test;
    allTrainedClassifiers{i} = trainedClassifier;
end
%end of models training
%-----------------------------------------------------------------------------------------------
disp('convert the final SVM Models matrix to table........')
%convert the final robustBoostOutput to table
finalSVMModelsOutput=array2table(finalSVMModelsOutput);
%modify the column headings of the table
finalSVMModelsOutput.Properties.VariableNames = {'ModelNo','KernelFunction',...
    'PolynomialOrder','KernelScale','BoxConstraint','TPR','FPR','AP'};
%display the table
disp(finalSVMModelsOutput)
%convert the final Knn test models results to table
finalTestResultsforModels=array2table(finalTestResultsforModels);
%modify the column headings of the table
finalTestResultsforModels.Properties.VariableNames = {'ModelNo' 'ValidationTPR','ValidationFPR','ValidationAP','TestTPR',' TestFPR','TestAP'};
%display the table
disp(finalTestResultsforModels)
%save('newDayDataSIFTFeatSVMModels.mat','finalSVMModelsOutput','finalTestResultsforModels')