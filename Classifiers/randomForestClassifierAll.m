function [allTrainedClassifiers, allConfMatrix,finalRFModelsOuput,finalTestResultsforModels] = randomForestClassifierAll(trainingData,testDataInput,testLabels)
%This classifier uses Matlab random forest implementation using Treebagger which is based on Breiman's random forest algorithm
% Extract predictors and response
% This code processes the data into the right shape for training the
% classifier.

inputTable = trainingData;
%Extract features/predictors from inputTable
predictorData = inputTable(:,1:end-1);
predictorNames = predictorData.Properties.VariableNames;
predictors = inputTable(:, predictorNames);
response = inputTable.label;

%Prepare input parameters and required variables
noOfLearners =[25,35,45,55,60,70,85,100,125, 150,200];  % number of decision trees 
leaf_size = [1 5 10 20 50];						   % number of leaf sizes for each number of decision trees
finalRFModelsOuput =[];     %place holder for all trained models for comparison
finalTestResultsforModels =[];
allConfMatrix={};
allTrainedClassifiers={};

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
for i=1:length(noOfLearners)
    modelNo = strcat('Currently training Model No.',num2str(i));
    disp(modelNo);
    classifierOutput =[];
    classificationEnsemble = fitensemble(...
        predictors, ...
        response, ...
        'Bag', ...
        noOfLearners(i), ...
        'Tree', ...
        'Type', 'Classification', ...
        'ClassNames', {'Kangaroo'; 'Not Kangaroo'});

    % Create the result struct with predict function
    predictorExtractionFcn = @(t) t(:, predictorNames);
    ensemblePredictFcn = @(x) predict(classificationEnsemble, x);
    trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

    % Add additional fields to the result struct
    trainedClassifier.RequiredVariables = predictorNames;
    trainedClassifier.ClassificationEnsemble = classificationEnsemble;
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
    %-----------------------------------------------------------------------------------------------
    % Compute resubstitution accuracy
    resubstitutionAccuracy = 1 - resubLoss(trainedClassifier.ClassificationEnsemble, 'LossFun', 'ClassifError');

    % Compute resubstitution predictions and scores
    [resubstitutionPredictions, resubstitutionScores] = predict(trainedClassifier.ClassificationEnsemble, predictors);
    disp('Training accuracy:')
    disp(resubstitutionAccuracy)
    %training confusion matrix
    confMatrix4training = confusionmat(response,resubstitutionPredictions);
    disp('Training Confusion Matrix:')
    disp(confMatrix4training)
    disp('----------------------------------------------------------------------------------------')
    disp('Computing validation confusion matrix for the model.........')
    confMatrix4validation = confusionmat(response,validationPredictions);
    disp('Validation Confusion Matrix:')
    disp(confMatrix4validation)
    disp('----------------------------------------------------------------------------------------')
    %-----------------------------------------------------------------------------------------------
    % Compute Rates for the validation
    disp('Computing rates from confusion matrix for the model.........');
    truePositiveRate = confMatrix4validation(1) /(confMatrix4validation(1)+confMatrix4validation(3));
    falsePositiveRate =confMatrix4validation(2) /(confMatrix4validation(2)+confMatrix4validation(4));
    disp('----------------------------------------------------------------------------------------');
    %-----------------------------------------------------------------------------------------------

    %Preparing current trained model
    classifierOutput =[i,noOfLearners(i),truePositiveRate,falsePositiveRate,validationAccuracy]
    modelDetails = strcat('Adding model to final out- Model No.',num2str(i))
    finalRFModelsOuput = vertcat(finalRFModelsOuput,classifierOutput)
    %Prepare test data
    testResults=[];
    disp('Now preparing test data for the model....');
    testdata = testDataInput;
    % disp('Converting test data variable names to match training data, this may take some time...........')    testdata.Properties.VariableNames = trainingData.Properties.VariableNames;
     testdata.Properties.VariableNames = trainingData.Properties.VariableNames;
    testdata = testdata(:,1:end-1);
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
%convert the final RF models output to table
finalRFModelsOuput=array2table(finalRFModelsOuput);
%modify the column headings of the table
finalRFModelsOuput.Properties.VariableNames = {'ModelNo','NoOfLearners','TPR','FPR','AP'};
%display the table
disp(finalRFModelsOuput)

%convert the final Knn test models results to table
finalTestResultsforModels=array2table(finalTestResultsforModels);
%modify the column headings of the table
finalTestResultsforModels.Properties.VariableNames = {'ModelNo' 'ValidationTPR','ValidationFPR','ValidationAP','TestTPR',' TestFPR','TestAP'};
%display the table
disp(finalTestResultsforModels)
