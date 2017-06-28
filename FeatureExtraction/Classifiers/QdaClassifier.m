function [trainedClassifier,finalResults,binomVector,NoOfSuccesses,actualNoOfKangaroos,actualNoOfNotKangaroos,totalNoOfTestImages] = QdaClassifier(trainingData,train_labels,testDataInput,testLabels)


% Extract predictors and response
% This code processes the data into the right shape for training the
Xtrain = trainingData;
Ytrain = train_labels;
Xtest = testDataInput;
Ytest = testLabels; 

%Variables for outputs
discrimType ={'diagQuadratic','pseudoQuadratic'};
	% Train a classifier
	% This code specifies all the classifier options and trains the classifier.
	classificationDiscriminant = fitcdiscr(...
		Xtrain, ...
		Ytrain, ...
		'DiscrimType', discrimType{1}, ...
		'FillCoeffs', 'off', ...
		'SaveMemory', 'on', ...
		'ClassNames', {'Kangaroo'; 'Not Kangaroo'});

	% Create the result struct with predict function
	discriminantPredictFcn = @(x) predict(classificationDiscriminant, x);
	trainedClassifier.predictFcn = @(x) discriminantPredictFcn(x);

	% Add additional fields to the result struct
	trainedClassifier.ClassificationDiscriminant = classificationDiscriminant;
    trainedClassifier.classifierType = 'QDA';
    trainedClassifier.classifierMethod ='Diagnal Quadratic regularization';
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
	[resubstitutionPredictions, resubstitutionScores] = predict(trainedClassifier.ClassificationDiscriminant, Xtrain);
	disp('Training accuracy:')
	disp(resubstitutionAccuracy)

 %CALL FUNCTION TO PROCESS THE TRAINED CLASSIFIER AND PRODUCE METRICS 
[finalResults,binomVector,NoOfSuccesses,actualNoOfKangaroos,actualNoOfNotKangaroos,totalNoOfTestImages]= computeClassifierResults(trainedClassifier,Ytrain,validationAccuracy,resubstitutionAccuracy,resubstitutionPredictions,Xtest,validationPredictions,Ytest)