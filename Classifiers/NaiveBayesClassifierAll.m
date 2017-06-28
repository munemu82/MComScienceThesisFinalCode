function [allConfMatrix,allTrainedClassifiers,finalNBModelsOutput,finalTestResultsforModels] = NaiveBayesClassifierAll(trainingData,testDataInput,testLabels)

%prepare input variables
inputTable = trainingData;
%Extract features/predictors from inputTable
predictorData = inputTable(:,1:end-1);
predictorNames = predictorData.Properties.VariableNames;
predictors = inputTable(:, predictorNames);
response = inputTable.label;

%Setupt Input parameters
%kernelSupport={[],[],[],[],[-20,20],[-5,5],[-10,10]}; %If a cell is empty ([]), then the software did not fit a kernel distribution to the corresponding predictor.
kernelType ={'normal','epanechnikov','triangle','box','triangle','triangle','triangle'};
kernelSupport={[],[],[],[-2,2],[-1,1],[-5,5],[-10,10]};
%Set other variables
finalNBModelsOutput =[];
finalTestResultsforModels =[];
allTrainedClassifiers={};
allConfMatrix ={};

%LOOP to create models for Naive Bayes
for i=1:length(kernelSupport)
     
    disp('Now training a classifier for the individual models');
        classifierOutput =[]; 			%place holder for the current iteration
    
    %perform classification
    nbClassifier = fitcnb(predictors,response, 'distribution','kernel','kernel',kernelSupport{i},'kernel',kernelType{i});
    %More details on Naive Bayes can be found at http://au.mathworks.com/help/stats/naive-bayes-classification.html
    %and http://au.mathworks.com/help/stats/fitnaivebayes.html
    % Create the result struct with predict function
    predictorExtractionFcn = @(t) t(:, predictorNames);
    nbPredictFcn = @(x) predict(nbClassifier, x);
    trainedClassifier.predictFcn = @(x) nbPredictFcn(predictorExtractionFcn(x));

    % Add additional fields to the result struct
    trainedClassifier.RequiredVariables = predictorNames;
    trainedClassifier.NBClassification = nbClassifier;
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
   
    %validation confusion matrix
    disp('Computing validation confusion matrix for the model.........')
    confMatrix4validation = confusionmat(response,validationPredictions);
    disp('Validation Confusion Matrix:')
    disp(confMatrix4validation)
    disp('----------------------------------------------------------------------------------------');
     
    % Compute Rates for the validation
    disp('Computing rates from confusion matrix for the model.........')
    truePositiveRate = confMatrix4validation(1) /(confMatrix4validation(1)+confMatrix4validation(3));
    falsePositiveRate =confMatrix4validation(2) /(confMatrix4validation(2)+confMatrix4validation(4));
    disp('----------------------------------------------------------------------------------------');
    %-----------------------------------------------------------------------------------------------
      
    %add model and its properties to the current temp place holder vector
    disp('Add trained model to the final NB models place holder');
    classifierOutput =[i,'kernel',strcat('[',strtrim(cellstr(num2str(kernelSupport{i}))),']'),...
        kernelType(i),truePositiveRate,falsePositiveRate,validationAccuracy]
    % function cellstr(num2str(kernelSupport{i}))) output the cell array as
    % string
    disp('----------------------------------------------------------------------------------------');
    %add model to the final NB Models output matrix
    disp('Add trained model to the final NB models place holder');
    finalNBModelsOutput =vertcat(finalNBModelsOutput,classifierOutput)
    disp('----------------------------------------------------------------------------------------');
       
    %Prepare test data
    testResults=[];
    disp('Now preparing test data for the model....');
    testdata = testDataInput;
    % disp('Converting test data variable names to match training data, this may take some time...........')
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
     %add test confusion matrix to the matrices list
    allConfMatrix{i} = confMatrix4test;
     %add classifier to the classifiers list
    allTrainedClassifiers{i} = trainedClassifier;
end
%end of models training
%-----------------------------------------------------------------------------------------------
disp('convert the final NB Models matrix to table........')
%convert the final robustBoostOutput to table
finalNBModelsOutput=array2table(finalNBModelsOutput);
%modify the column headings of the table
finalNBModelsOutput.Properties.VariableNames = {'ModelNo','Distribution','KernelSupport','KernelSupportType','TPR','FPR','AP'};
%display the table
disp(finalNBModelsOutput)

%convert the final Knn test models results to table
finalTestResultsforModels=array2table(finalTestResultsforModels);
%modify the column headings of the table
finalTestResultsforModels.Properties.VariableNames = {'ModelNo' 'ValidationTPR','ValidationFPR','ValidationAP','TestTPR',' TestFPR','TestAP'};
%display the table
disp(finalTestResultsforModels)

save ('originalDataColorFeatNBModelsResults.mat','finalNBModelsOutput','finalTestResultsforModels')