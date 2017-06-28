function [allTrainedClassifiers,adaBoostTrainedClassifiers,subspaceBoostTrainedClassfiers,logitBoostTrainedClassifies,gentleBoostTraClassifiers,rusBoostTrainedClassifiers,finalTestResultsforModels,gentleBoostOutput,adaBoostOutput,logitBoostOutput,rUSBoostOutput,subspaceOutput]= adaBoostAll(trainingData,testDataInput,testLabels)

% Extract predictors and response
% This code processes the data into the right shape for training the
inputTable = trainingData;
%Extract features/predictors from inputTable
predictorData = inputTable(:,1:end-1);
predictorNames = predictorData.Properties.VariableNames;
predictors = inputTable(:, predictorNames);
response = inputTable.label;

%Set parameters to start with
listOFLearners =[30,30,35,35,35,35];  %original  listOFLearners =[30,30,35,35,35,35];
listOFMaximumSplitValues=[20,25,25,20,35,15];
%adaBoostMethodType={'LogitBoost','GentleBoost','AdaBoostM1','RUSBoost','LPBoost','TotalBoost','RobustBoost','Subspace'};
adaBoostMethodType={'Subspace','LogitBoost','GentleBoost','AdaBoostM1','RUSBoost'};
%learningRate = 0.1;
rUSBoostOutput = [];  % place holder for the result of RUSBoost models
subspaceOutput = [];  % place holder for the result of Subspace models
gentleBoostOutput = [];  % place holder for the result of GentleBoost models
adaBoostOutput = [];  	 % place holder for the result of default adaBoost models
logitBoostOutput = [];   % place holder for the result of LogitBoost models
finalTestResultsforModels =[];
allConfMatrix = {};
allTrainedClassifiers={};
%Classifiers
subspaceBoostTrainedClassfiers = {};
logitBoostTrainedClassifies = {};
gentleBoostTraClassifiers = {};
adaBoostTrainedClassifiers = {};
rusBoostTrainedClassifiers = {};

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
for j=1:length(adaBoostMethodType)
    for i=1:length(listOFLearners)
         disp(adaBoostMethodType(j));
         modelNo = strcat('Currently training Model No.',num2str(i));
         disp(modelNo);
        classifierOutput =[]; 			%place holder for the current iteration
        
         if strcmp('Subspace',adaBoostMethodType{j})==1
            subspaceDimension = max(1, min(2688, width(predictors) - 1));
            classificationEnsemble = fitensemble(...
                predictors, ...
                response, ...
                'Subspace', ...
                listOFLearners(i), ...
                'KNN', ...
                'Type', 'Classification', ...
                'NPredToSample', subspaceDimension, ...
                'ClassNames', {'Kangaroo'; 'Not Kangaroo'});
         else

            template = templateTree(...
                'MaxNumSplits', listOFMaximumSplitValues(i));
            classificationEnsemble = fitensemble(...
                predictors, ...
                response, ...
                adaBoostMethodType{j}, ...
                listOFLearners(i), ...   
                template, ...
                'Type', 'Classification', ...
                'ClassNames', {'Kangaroo'; 'Not Kangaroo'});
          end

        % Create the result struct with predict function
        predictorExtractionFcn = @(t) t(:, predictorNames);
        ensemblePredictFcn = @(x) predict(classificationEnsemble, x);
        trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

        % Add additional fields to the result struct
        trainedClassifier.RequiredVariables =predictorNames;
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
            %training confusion matrix
            confMatrix4validation = confusionmat(response,validationPredictions);
            disp('Validation Confusion Matrix:')
            disp(confMatrix4validation)
            disp('----------------------------------------------------------------------------------------')
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
            % Compute Rates for the validation
            disp('Computing rates from confusion matrix for the model.........')
            truePositiveRate = confMatrix4validation(1) /(confMatrix4validation(1)+confMatrix4validation(3));
            falsePositiveRate =confMatrix4validation(2) /(confMatrix4validation(2)+confMatrix4validation(4));
            %-----------------------------------------------------------------------------------------------

            %add AdaBoost model types to the final output
            if j==2
                classifierOutput =[i,listOFMaximumSplitValues(i),listOFLearners(i),truePositiveRate,falsePositiveRate,validationAccuracy]
                logitBoostOutput =vertcat(logitBoostOutput,classifierOutput)
                logitBoostTrainedClassifies{i}= trainedClassifier;
            elseif j==3
                classifierOutput =[i,listOFMaximumSplitValues(i),listOFLearners(i),truePositiveRate,falsePositiveRate,validationAccuracy]
                gentleBoostOutput =vertcat(gentleBoostOutput,classifierOutput)
                gentleBoostTraClassifiers{i}= trainedClassifier;
            elseif j==4
                classifierOutput =[i,listOFMaximumSplitValues(i),listOFLearners(i),truePositiveRate,falsePositiveRate,validationAccuracy]
                adaBoostOutput =vertcat(adaBoostOutput,classifierOutput)
                adaBoostTrainedClassifiers{i}= trainedClassifier;
            elseif j==5
                classifierOutput =[i,listOFMaximumSplitValues(i),listOFLearners(i),truePositiveRate,falsePositiveRate,validationAccuracy]
                rUSBoostOutput =vertcat(rUSBoostOutput,classifierOutput)
                rusBoostTrainedClassifiers{i}= trainedClassifier;
            else
                classifierOutput =[i,listOFMaximumSplitValues(i),listOFLearners(i),truePositiveRate,falsePositiveRate,validationAccuracy]
                subspaceOutput =vertcat(subspaceOutput,classifierOutput)
                subspaceBoostTrainedClassfiers{i}= trainedClassifier;
            end


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
            truePositiveRate3 = confMatrix4test(1) /(confMatrix4test(1)+confMatrix4test(3));
            falsePositiveRate3 =confMatrix4test(2) /(confMatrix4test(2)+confMatrix4test(4));
             testAccuracy = truePositiveAndNegatives / numel(testLabels);
            disp(testAccuracy)
            disp('Test Confusion Matrix:')
            disp(confMatrix4test)
            allConfMatrix{i} = confMatrix4test;
            trainedClassiers{i} =trainedClassifier;
            testResults =[i,truePositiveRate,falsePositiveRate,validationAccuracy...
            truePositiveRate3,falsePositiveRate3,testAccuracy]
            finalTestResultsforModels =vertcat(finalTestResultsforModels,testResults);
            
    end
    allTrainedClassifiers = vertcat(allTrainedClassifiers,trainedClassiers);
end
%convert the final logitBoostOutput to table
logitBoostOutput=array2table(logitBoostOutput);
%modify the column headings of the table
logitBoostOutput.Properties.VariableNames = {'ModelNo' 'MaximumSplits' 'NoOfLearners','TPR','FPR','AP'};
%display the table
disp(logitBoostOutput)
%convert the final gentleBoostOutput to table
gentleBoostOutput=array2table(gentleBoostOutput);
%modify the column headings of the table
gentleBoostOutput.Properties.VariableNames = {'ModelNo' 'MaximumSplits' 'NoOfLearners','TPR','FPR','AP'};
%display the table
disp(gentleBoostOutput)
%--------------------------------------------------------------------------
%convert the final adaBoostOutput to table
adaBoostOutput=array2table(adaBoostOutput);
%modify the column headings of the table
adaBoostOutput.Properties.VariableNames = {'ModelNo' 'MaximumSplits' 'NoOfLearners','TPR','FPR','AP'};
%display the table
disp(adaBoostOutput)
%--------------------------------------------------------------------------
%convert the final robustBoostOutput to table
rUSBoostOutput=array2table(rUSBoostOutput);
%modify the column headings of the table
rUSBoostOutput.Properties.VariableNames = {'ModelNo' 'MaximumSplits' 'NoOfLearners','TPR','FPR','AP'};
%display the table
disp(rUSBoostOutput)
%--------------------------------------------------------------------------
%convert the final robustBoostOutput to table
subspaceOutput=array2table(subspaceOutput);
%modify the column headings of the table
subspaceOutput.Properties.VariableNames = {'ModelNo' 'MaximumSplits' 'NoOfLearners','TPR','FPR','AP'};
%display the table
disp(subspaceOutput)
%--------------------------------------------------------------------------

%convert the final Knn test models results to table
finalTestResultsforModels=array2table(finalTestResultsforModels);
%modify the column headings of the table
finalTestResultsforModels.Properties.VariableNames = {'ModelNo' 'ValidationTPR','ValidationFPR','ValidationAP','TestTPR',' TestFPR','TestAP'};
%display the table
disp(finalTestResultsforModels)
