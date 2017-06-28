function [allTrainedClassifiers,finalKnnModels,finalTestResultsforModels,allConfMatrix] =KNNClassifierAll(trainingData,testDataInput,testLabels)

% Extract predictors and response
% This code processes the data into the right shape for training the
% classifier.
inputTable = trainingData;
%Extract features/predictors from inputTable
predictorData = inputTable(:,1:end-1);


predictorNames = predictorData.Properties.VariableNames;
predictors = inputTable(:, predictorNames);
response = inputTable.label;

%Set parameters to start with
%valueOfK =[1,10,10,10,10,3,7,10];
valueOfK =[1,3,5,10,10,7,9,10];
distanceFunction={'Euclidean','Euclidean','Cosine','Minkowski','Euclidean','Euclidean','Euclidean','Euclidean'};
distanceWeight='';
%Output variables
finalKnnModels =[];
finalTestResultsforModels = [];
allConfMatrix ={};
allTrainedClassifiers ={};
% Train a classifier
% This code specifies all the classifier options and trains the classifier.
for i=1:length(valueOfK)
    classifierOutput =[];
    disp(length(finalKnnModels))
    disp(length(classifierOutput))
    if i==5
        distanceWeight ='SquaredInverse';
    else
        distanceWeight ='Equal';
    end
    classificationKNN = fitcknn(...
        predictors, ...
        response, ...
        'Distance', distanceFunction{i}, ...
        'Exponent', [], ...
        'NumNeighbors', valueOfK(i), ...
        'DistanceWeight', distanceWeight, ...
        'Standardize', true, ...
        'ClassNames', {'Kangaroo'; 'Not Kangaroo'});

    % Create the result struct with predict function
    predictorExtractionFcn = @(t) t(:, predictorNames);
    knnPredictFcn = @(x) predict(classificationKNN, x);
    trainedClassifier.predictFcn = @(x) knnPredictFcn(predictorExtractionFcn(x));

    % Add additional fields to the result struct
    trainedClassifier.RequiredVariables = predictorNames;
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
    %training confusion matrix
    confMatrix4validation = confusionmat(response,validationPredictions);
    disp('Validation Confusion Matrix:')
    disp(confMatrix4validation)
    disp('----------------------------------------------------------------------------------------')
    truePositiveRate = confMatrix4validation(1) /(confMatrix4validation(1)+confMatrix4validation(3));
    falsePositiveRate =confMatrix4validation(2) /(confMatrix4validation(2)+confMatrix4validation(4));
    %-----------------------------------------------------------------------------------------------
    %add model to the final KNN models output
    classifierOutput =[i,distanceFunction(i), distanceWeight,num2str(valueOfK(i)),truePositiveRate,falsePositiveRate,validationAccuracy]
    finalKnnModels =vertcat(finalKnnModels,classifierOutput)
    %---------------------------------------------------------------------------------------------
    % Compute resubstitution accuracy
    resubstitutionAccuracy = 1 - resubLoss(trainedClassifier.ClassificationKNN, 'LossFun', 'ClassifError');

    % Compute resubstitution predictions and scores
    [resubstitutionPredictions, resubstitutionScores] = predict(trainedClassifier.ClassificationKNN, predictors);
    disp('Training accuracy:')
    disp(resubstitutionAccuracy)
    %training confusion matrix
    confMatrix4training = confusionmat(response,resubstitutionPredictions);
    disp('Training Confusion Matrix:')
    disp(confMatrix4training)
    disp('----------------------------------------------------------------------------------------')
    %------------------------------------------------------------------------------------------------------
    %Prepare test data
    testdata = testDataInput;
    testResults =[];
    % disp('Converting test data variable names to match training data, this may take some time...........')
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
    disp('Test Confusion Matrix:')
    disp(confMatrix4test)
    %add confusion matrix to the list
    allConfMatrix{i} = confMatrix4test;
    %Compute rates from test confusion matrix
    truePositiveRate3 = confMatrix4test(1) /(confMatrix4test(1)+confMatrix4test(3));
    falsePositiveRate3 =confMatrix4test(2) /(confMatrix4test(2)+confMatrix4test(4));
    testResults =[i,truePositiveRate,falsePositiveRate,validationAccuracy...
    truePositiveRate3,falsePositiveRate3,testAccuracy]
    finalTestResultsforModels =vertcat(finalTestResultsforModels,testResults)
    allTrainedClassifiers{i} = trainedClassifier;
end
%convert the final Knn validation models results to table
finalKnnModels=array2table(finalKnnModels);
%modify the column headings of the table
finalKnnModels.Properties.VariableNames = {'ModelNo' 'DistanceFunction','distanceWeight','ValueOfK','TPR','FPR','AP'};
%display the table
disp(finalKnnModels)

%convert the final Knn test models results to table
finalTestResultsforModels=array2table(finalTestResultsforModels);
%modify the column headings of the table
finalTestResultsforModels.Properties.VariableNames = {'ModelNo' 'ValidationTPR','ValidationFPR','ValidationAP','TestTPR',' TestFPR','TestAP'};
%display the table
disp(finalTestResultsforModels)
