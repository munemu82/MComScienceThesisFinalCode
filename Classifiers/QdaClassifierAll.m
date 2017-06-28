function [allTrainedClassifiers, validationAccuracy,allConfMatrix,finalQDAModelsOutput,finalTestResultsforModels] =QdaClassifierAll(trainingData,testDataInput,testLabels)
% Extract predictors and response
% This code processes the data into the right shape for training the
% classifier.
inputTable = trainingData;
%Extract features/predictors from inputTable
predictorData = inputTable(:,1:end-1);
predictorNames = predictorData.Properties.VariableNames;
predictors = inputTable(:, predictorNames);
response = inputTable.label;

%Variables for outputs
discrimType ={'diagQuadratic','pseudoQuadratic'};
finalQDAModelsOutput=[];
finalTestResultsforModels =[];
classifierOutput={};
allConfMatrix={};
allTrainedClassifiers = {};
for i=1:length(discrimType)
	classifierOutput=[];
	% Train a classifier
	% This code specifies all the classifier options and trains the classifier.
	classificationDiscriminant = fitcdiscr(...
		predictors, ...
		response, ...
		'DiscrimType', discrimType{i}, ...
		'FillCoeffs', 'off', ...
		'SaveMemory', 'on', ...
		'ClassNames', {'Kangaroo'; 'Not Kangaroo'});

	% Create the result struct with predict function
	predictorExtractionFcn = @(t) t(:, predictorNames);
	discriminantPredictFcn = @(x) predict(classificationDiscriminant, x);
	trainedClassifier.predictFcn = @(x) discriminantPredictFcn(predictorExtractionFcn(x));

	% Add additional fields to the result struct
	trainedClassifier.RequiredVariables = predictorNames;
	trainedClassifier.ClassificationDiscriminant = classificationDiscriminant;
	trainedClassifier.About = 'This struct is a trained classifier exported from Classification Learner R2016a.';
	trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedClassifier''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

	% Perform cross-validation
	partitionedModel = crossval(trainedClassifier.ClassificationDiscriminant, 'KFold', 10);

	% Compute validation accuracy
	validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');

	% Compute validation predictions and scores
	[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

	%display validation accuracy
	disp('Validation accuracy:')
	disp(validationAccuracy)
	%-----------------------------------------------------------------------------------------------
	% Compute resubstitution accuracy
	resubstitutionAccuracy = 1 - resubLoss(trainedClassifier.ClassificationDiscriminant, 'LossFun', 'ClassifError');

	% Compute resubstitution predictions and scores
	[resubstitutionPredictions, resubstitutionScores] = predict(trainedClassifier.ClassificationDiscriminant, predictors);
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
		
	classifierOutput ={i,discrimType{i}, truePositiveRate,falsePositiveRate,validationAccuracy}
	finalQDAModelsOutput =vertcat(finalQDAModelsOutput,classifierOutput)

	%Prepare test data
	testResults=[];
	disp('Now preparing test data for the model....');
	testdata = testDataInput;
	% disp('Converting test data variable names to match training data, this may take some time...........')    testdata.Properties.VariableNames = trainingData.Properties.VariableNames;
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
	testResults ={i, discrimType{i}, truePositiveRate,falsePositiveRate,validationAccuracy...
	truePositiveRate3,falsePositiveRate3,testAccuracy}
	allConfMatrix{i} = confMatrix4test;
	finalTestResultsforModels =vertcat(finalTestResultsforModels,testResults)
    allTrainedClassifiers{i} = trainedClassifier;
end

disp('convert the final SVM Models matrix to table........')
%convert the final Training model to table
finalQDAModelsOutput=array2table(finalQDAModelsOutput);
%modify the column headings of the table
finalQDAModelsOutput.Properties.VariableNames = {'ModelNo','DiscriminantType',...
    'TPR','FPR','AP'};
%display the table
disp(finalQDAModelsOutput)

%convert the final test results to table
finalTestResultsforModels=array2table(finalTestResultsforModels);
%modify the column headings of the table
finalTestResultsforModels.Properties.VariableNames = {'ModelNo','DiscriminantType','ValidationTPR','ValidationFPR',...
    'ValidationAP','TestTPR','TestFPR','TestAP'};
%display the table
disp(finalTestResultsforModels)

