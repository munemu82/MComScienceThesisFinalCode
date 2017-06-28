function [finalResults,binomVector,NoOfSuccesses,actualNoOfKangaroos,actualNoOfNotKangaroos,totalNoOfTestImages] = computeClassifierResults(trainedClassifier,response,validationAccuracy,resubstitutionAccuracy,resubstitutionPredictions,testdata,validationPredictions,testLabels)
finalResults =[];
%Validation confusion matrix
confMatrix4validation = confusionmat(response,validationPredictions);
disp('Validation Confusion Matrix:')
disp(confMatrix4validation)
disp('----------------------------------------------------------------------------------------')
 truePositiveRate = confMatrix4validation(1) /(confMatrix4validation(1)+confMatrix4validation(3));
 falsePositiveRate =confMatrix4validation(2) /(confMatrix4validation(2)+confMatrix4validation(4));
%-----------------------------------------------------------------------------------------------
validationResults ={'Training-10KFold Validation',num2str(truePositiveRate),num2str(falsePositiveRate),num2str(validationAccuracy)}

% confMatrix4training = confusionmat(response,resubstitutionPredictions);
% disp('Training Confusion Matrix:')
% disp(confMatrix4training)
% disp('----------------------------------------------------------------------------------------')
%  truePositiveRate2 = confMatrix4training(1) /(confMatrix4training(1)+confMatrix4training(3));
%  falsePositiveRate2 =confMatrix4training(2) /(confMatrix4training(2)+confMatrix4training(4));
%  trainingResults ={'Training-No Validation',num2str(truePositiveRate2),num2str(falsePositiveRate2),num2str(resubstitutionAccuracy)}

%------------------------------------------------------------------------------------------------------
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
 truePositiveRate3 = confMatrix4test(1) /(confMatrix4test(1)+confMatrix4test(3));
 falsePositiveRate3 =confMatrix4test(2) /(confMatrix4test(2)+confMatrix4test(4));
testResults ={'Test Dataset',num2str(truePositiveRate3),num2str(falsePositiveRate3),num2str(testAccuracy)}
finalResults =vertcat(validationResults,testResults)

%COMPUTING NUMBERS FOR THE BINOMIAL PROBABILITY ANALYSIS
binomVector =[];
NoOfSuccesses =0;
actualNoOfKangaroos =0;
actualNoOfNotKangaroos =0;
totalNoOfTestImages = length(testLabels);
for i=1:length(testLabels)
    if strcmp(testLabels{i},'Kangaroo')==1
        actualNoOfKangaroos = actualNoOfKangaroos + 1;
    else
        actualNoOfNotKangaroos = actualNoOfNotKangaroos +1;
    end
    if strcmp(testLabels{i},'Kangaroo')==1 && strcmp(testPredictions{i},'Kangaroo')==1
        binomVector{i} = 1;
        NoOfSuccesses = NoOfSuccesses + 1;
    elseif strcmp(testLabels{i},'Not Kangaroo')==1 && strcmp(testPredictions{i},'Not Kangaroo')==1
        binomVector{i} = 1;
        NoOfSuccesses = NoOfSuccesses + 1;
    else
        binomVector{i} = 0;
    end
end 
binomVector = binomVector';