function [finalOutput,finalTestOutput,allTestPredictionOutputs,allTrainingPredictionOutputs] = elmTrainAll(trainingFeatureMatrix,trainLabels,testFeatureMatrix,testLabels)               %trainingset is filename of training dataset which in .txt format

% this function takes 4 inputs  - 1) Training Images features matrix 2)
% Training images labels containing "0" and "1" , 0=Not Kangaroo, and 1=Kangaroo
%3)Test images features matrix , 4) Test images labels containing "0" and "1" , 0=Not Kangaroo, and 1=Kangaroo

%prepare dataset  - this will combine the 4 inputs and create two files
%suitable for working with ELMs toolbox
%1) for the training trainingDataset.txt
%2) For testDataset.txt

%Apppend numeric labels to the training dataset features
trainingDataset  = [trainLabels,trainingFeatureMatrix];
testDataset =[testLabels,testFeatureMatrix];
%Convert the trainingDataset and testDataset matrices to flat .txt file
dlmwrite('trainingDataset.txt',trainingDataset)
dlmwrite('testDataset.txt',testDataset)

%Set Parameters required
elm_type = 1;               %this is a type of elm model to be run - 1 for classification and 0 for regression
noOfHiddenNeurons =[150,150,150,100,100,100,200,200];           % the array contains 8 no. of hidden neurons to be used for each of 8 models
%activationFunction ={'sig','sin','hardlim'};  % contains various activation functions to be tried, we are using 3 different types of functions, Sigmoid, Sine, and Hardlim 
%noOfHiddenNeurons =[150,150,150,100,100,100,200,200];           % the array contains 8 no. of hidden neurons to be used for each of 8 models
%noOfHiddenNeurons =[135,155,175,200,225,275,500,500]; 
activationFunction ={'sig','sin','hardlim','sig','sin','hardlim','sin','sig'};  % contains various activation functions to be tried, we are using 3 different types of functions, Sigmoid, Sine, and Hardlim 

finalOutput =[];
finalTestOutput =[];
allTestPredictionOutputs ={};
allTrainingPredictionOutputs ={};

%Depending on the size of the array of noOfHiddenNeurons we iterate
%through and each iteration produce a model with out put
for i=1:length(noOfHiddenNeurons)
    %for j=50:500  %number of hidden neurons
    thisClassifierOutput =[];           %set empty place holder for the model
    thisTestOutput=[];
    %Calling elm train function to train the current model
    [TrainingTime,TrainingAccuracy] = elm_train('trainingDataset.txt', elm_type,noOfHiddenNeurons(i), activationFunction{i});
     %perform prediction on the training data set using the modelload('elm_output.mat')
    [TestingTime, TestingAccuracy] = elm_predict('trainingDataset.txt');


       load('elm_output.mat');      %load the model predictions output 
       trainOutput =output';             %transpose the model output to match our actual labels
       
       %add the training prediction output to the matrix
       allTrainingPredictionOutputs{i} = trainOutput;
       
       %preprare confusion matrix 
       confMatrix4validation = confusionmat(trainLabels,trainOutput)

       % Compute Rates for the validation
       disp('Computing rates for the model')
       truePositiveRate = confMatrix4validation(1) /(confMatrix4validation(1)+confMatrix4validation(3));
       falsePositiveRate =confMatrix4validation(2) /(confMatrix4validation(2)+confMatrix4validation(4));
       

       %perform prediction for test dataset
       [TestingTime, TestingAccuracy] = elm_predict('testDataset.txt');

       load('elm_output.mat');      %load the model predictions output 
       testOutput =output';             %transpose the model output to match our actual labels
       
       %add the test prediction output to the matrix
       allTestPredictionOutputs{i} = testOutput;
       
       %preprare confusion matrix 
       testConfMatrix = confusionmat(testLabels,testOutput)

       % Compute Rates for the validation
       disp('Computing rates for the model')
       testTPR = testConfMatrix(1) /(testConfMatrix(1)+testConfMatrix(3));
       testFPR =testConfMatrix(2) /(testConfMatrix(2)+testConfMatrix(4));
       
       %Create the model output
       thisClassifierOutput = [i,activationFunction(i),noOfHiddenNeurons(i),truePositiveRate,falsePositiveRate,TrainingAccuracy];
       %add the model output to the final output
        finalOutput =vertcat(finalOutput,thisClassifierOutput);
        %----------------------------------------------------------------------------------------------
       %create test results
       thisTestOutput = [i,truePositiveRate,falsePositiveRate,TrainingAccuracy,testTPR,testFPR,TestingAccuracy];
       %add the model output to the final output
       finalTestOutput =vertcat(finalTestOutput,thisTestOutput);
end

%convert the final output to table
finalOutput=array2table(finalOutput);
%modify the column headings of the table
finalOutput.Properties.VariableNames = {'ModelNo' 'ActivationFunction' 'NoOfHiddenNeurons','TPR','FPR','AP'};
%display the table
disp(finalOutput)

%convert the final output to table
finalTestOutput=array2table(finalTestOutput);
%modify the column headings of the table
finalTestOutput.Properties.VariableNames = {'ModelNo' 'ValidationTPR' 'ValidationFPR','ValidationAP','TestTPR','TestFPR','TestAP'};
%display the table
disp(finalTestOutput)
