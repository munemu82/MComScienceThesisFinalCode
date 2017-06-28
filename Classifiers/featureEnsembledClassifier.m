function [listOfConfMatrices,classifiersList,classifiersAccuracies,modelPredNonAgreementCount,modelPredAgreementCount] = featureEnsembledClassifier(trainColorFeat,testColorFeat,trainHOGFeat,testHOGFeat,trainLBPFeat,testLBPFeat, trainGISTFeat, testGISTFeat,trainSIFTFeat, testSIFTFeat,train_labels,test_labels)
%Timer for comparison between ensembled classifier vs Deep learning training
tic          %set a clock

%Training data matrix for each feature
Ytrain = train_labels;          % Common labels indicating whether image is kangaroo or not Kangaroos
XtrainColor = trainColorFeat;   % n-by-m matrix containing Color features representing image  
XtrainHOG = trainHOGFeat;       % n-by-m matrix containing HOG features representing image  
XtrainLBP = trainLBPFeat;       % n-by-m matrix containing LBP features representing image  
XtrainGIST = trainGISTFeat;     % n-by-m matrix containing GIST features representing image  
XtrainSIFT = trainSIFTFeat;     % n-by-m matrix containing SIFT features representing image  

%Test data matrix for each feature
Ytest = test_labels;            % Ground truth labels indicating whether image is kangaroo or not Kangaroos
XtestColor = testColorFeat;   % n-by-m matrix containing Color features representing image  
XtestHOG = testHOGFeat;       % n-by-m matrix containing HOG features representing image  
XtestLBP = testLBPFeat;       % n-by-m matrix containing LBP features representing image
XtestGIST = testGISTFeat;       % n-by-m matrix containing GIST features representing image
XtestSIFT = testSIFTFeat;       % n-by-m matrix containing SIFT features representing image


%Set required variables
finalTestPredictions = [];
testResults = [];
opts =statset('MaxIter',30000);
%train the model 1 -  color-SVM  (using rbf kernel function)
color_SVM_classifier = fitcsvm(XtrainColor,Ytrain, 'KernelFunction','rbf', ...
    'KernelScale', 'Auto','BoxConstraint',Inf, 'Standardize', true, ...
    'ClassNames', {'Kangaroo'; 'Not Kangaroo'});

%train the model 2 -  HOG-SVM  (using rbf kernel function)
hog_SVM_classifier = fitcsvm(XtrainHOG,Ytrain, 'KernelFunction','gaussian', ...
    'KernelScale', 91,'BoxConstraint',5, 'Standardize', true, ...
    'ClassNames', {'Kangaroo'; 'Not Kangaroo'});

%train the model 3 -  LBP-SVM  (using rbf kernel function)
lbp_SVM_classifier = fitcsvm(XtrainLBP,Ytrain, 'KernelFunction','rbf', ...
    'KernelScale', 'Auto','BoxConstraint',Inf, 'Standardize', true, ...
    'ClassNames', {'Kangaroo'; 'Not Kangaroo'});

%train the model 4 -  GIST-SVM  (using linear kernel function)
gist_SVM_classifier = fitcsvm(XtrainGIST,Ytrain, 'KernelFunction','polynomial', ...
    'PolynomialOrder',3,'KernelScale', 'Auto','BoxConstraint',5, 'Standardize', true, ...
    'ClassNames', {'Kangaroo'; 'Not Kangaroo'});

%train the model 5 - SIFT-SVM  (using rbf kernel function)
sift_SVM_classifier = fitcsvm(XtrainSIFT,Ytrain, 'KernelFunction','rbf', ...
    'KernelScale', 'Auto','BoxConstraint',Inf, 'Standardize', true, ...
    'ClassNames', {'Kangaroo'; 'Not Kangaroo'});

%predictions for each models
[classifier1Predictions,classifier1Scores] = predict(color_SVM_classifier,XtestColor);
[classifier2Predictions,classifier2Scores] = predict(hog_SVM_classifier,XtestHOG);
[classifier3Predictions,classifier3Scores] = predict(lbp_SVM_classifier,XtestLBP);
[classifier4Predictions,classifier4Scores] = predict(gist_SVM_classifier,XtestGIST);
[classifier5Predictions,classifier5Scores] = predict(sift_SVM_classifier,XtestSIFT);


%models confusion matrices
disp('Computing models confusion matrices')
disp('--------------------------------------------------------------------')
classifier1ConfMatrix = confusionmat(Ytest,classifier1Predictions)
classifier2ConfMatrix = confusionmat(Ytest,classifier2Predictions)
classifier3ConfMatrix = confusionmat(Ytest,classifier3Predictions)
classifier4ConfMatrix = confusionmat(Ytest,classifier4Predictions)
classifier5ConfMatrix = confusionmat(Ytest,classifier5Predictions)

listOfConfMatrices = [];
listOfConfMatrices{1} = classifier1ConfMatrix; 
listOfConfMatrices{2} = classifier2ConfMatrix;
listOfConfMatrices{3} = classifier3ConfMatrix; 
listOfConfMatrices{4} = classifier4ConfMatrix; 
listOfConfMatrices{5} = classifier5ConfMatrix; 

%models accuracies
disp('Computing individuals confusion matrices')
disp('--------------------------------------------------------------------')
classifier1Accuracy = (classifier1ConfMatrix(1) + classifier1ConfMatrix(4)) / numel(Ytest)
classifier2Accuracy = (classifier2ConfMatrix(1) + classifier2ConfMatrix(4)) / numel(Ytest)
classifier3Accuracy = (classifier3ConfMatrix(1) + classifier3ConfMatrix(4)) / numel(Ytest)
classifier4Accuracy = (classifier4ConfMatrix(1) + classifier4ConfMatrix(4)) / numel(Ytest)
classifier5Accuracy = (classifier5ConfMatrix(1) + classifier5ConfMatrix(4)) / numel(Ytest)

%add accuracies to the list
classifiersAccuracies =[classifier1Accuracy classifier2Accuracy classifier3Accuracy...
    classifier4Accuracy classifier5Accuracy];

%add classifiers to the list
classifiersList = [];
classifiersList{1} = color_SVM_classifier;
classifiersList{2} = hog_SVM_classifier;
classifiersList{3} = lbp_SVM_classifier;
classifiersList{4} = gist_SVM_classifier;
classifiersList{5} = sift_SVM_classifier;

%Create Binomial distribution for each model and create vector
classifier1Binom = [];
classifier2Binom = [];
classifier3Binom = [];


for i=1:length(Ytest)
    %for classifier 1
   if strcmp(Ytest{i},'Kangaroo')==1 && strcmp(classifier1Predictions{i},'Kangaroo')==1
        classifier1Binom{i} = 1;
    elseif strcmp(Ytest{i},'Not Kangaroo')==1 && strcmp(classifier1Predictions{i},'Not Kangaroo')==1
        classifier1Binom{i} = 1;
    else
        classifier1Binom{i} = 0;
   end
   %for classifier 2
   if strcmp(Ytest{i},'Kangaroo')==1 && strcmp(classifier2Predictions{i},'Kangaroo')==1
        classifier2Binom{i} = 1;
    elseif strcmp(Ytest{i},'Not Kangaroo')==1 && strcmp(classifier2Predictions{i},'Not Kangaroo')==1
        classifier2Binom{i} = 1;
    else
        classifier2Binom{i} = 0;
   end
   %for classifier 3
   if strcmp(Ytest{i},'Kangaroo')==1 && strcmp(classifier3Predictions{i},'Kangaroo')==1
        classifier3Binom{i} = 1;
    elseif strcmp(Ytest{i},'Not Kangaroo')==1 && strcmp(classifier3Predictions{i},'Not Kangaroo')==1
        classifier3Binom{i} = 1;
    else
        classifier3Binom{i} = 0;
   end
  
end

%Checking correlations of predictions between models to determine which
modelPredAgreementCount = 0;
modelPredNonAgreementCount = 0;
%models combinations should be used for predicitons
%Lets consider 3 models
for k=1:length(Ytest)
    if classifier1Binom{k}== classifier2Binom{k} 
        modelPredAgreementCount = modelPredAgreementCount + 1;
    elseif classifier1Binom{k}== classifier3Binom{k}
        modelPredAgreementCount = modelPredAgreementCount + 1;
    elseif classifier2Binom{k}== classifier3Binom{k}
        modelPredAgreementCount = modelPredAgreementCount + 1;
    elseif classifier2Binom{k}== classifier3Binom{k} && classifier2Binom{k}== classifier1Binom{k}
        modelPredAgreementCount = modelPredAgreementCount + 1;
    else
        modelPredNonAgreementCount = modelPredNonAgreementCount +1;
    end
end

%using combined classifiers to predict a label
for i=1:length(Ytest)
    kangaroosCount = 0;
    notKangaroosCount = 0;
    %for each of the pre-trained classifier, perform prediction against the
    %test i'th record/image feature vector
    
    %For color feature SVM prediction
     [colorPrediction,colorProbScore] =  predict(color_SVM_classifier,XtestColor(i,:)); 
     
     %For HOG feature
     [hogPrediction,hogProbScore] =  predict(hog_SVM_classifier,XtestHOG(i,:)); 
     
      %For LBP feature
     [lbpPrediction,lbpProbScore] =  predict(lbp_SVM_classifier,XtestLBP(i,:)); 
     
      %For GIST feature
     [gistPrediction,gistProbScore] =  predict(gist_SVM_classifier,XtestGIST(i,:)); 
     
      %For SIFT feature
     [siftPrediction,siftProbScore] =  predict(sift_SVM_classifier,XtestSIFT(i,:)); 
     
     if strcmp(colorPrediction{1},'Kangaroo')==1                %ned to add weights
        kangaroosCount = kangaroosCount + colorProbScore + 1;
     else 
        notKangaroosCount = notKangaroosCount + colorProbScore + 1;
         
     end
     
     if strcmp(gistPrediction{1},'Kangaroo')==1                %ned to add weights
        kangaroosCount = kangaroosCount + gistProbScore + 1;
     else 
        notKangaroosCount = notKangaroosCount + gistProbScore + 1;
         
     end
     
      if strcmp(siftPrediction{1},'Kangaroo')==1                %ned to add weights
        kangaroosCount = kangaroosCount +siftProbScore + 1;
     else 
        notKangaroosCount = notKangaroosCount + siftProbScore + 1;
         
     end
     
     if strcmp(hogPrediction{1},'Kangaroo')==1                %ned to add weights
%        if strcmp(lbpPrediction{1},'Kangaroo')==1  
%             kangaroosCount = kangaroosCount + lbpProbScore + 4;
%        else
           kangaroosCount = kangaroosCount + hogProbScore + 3;
%        end
     else 
%          if strcmp(lbpPrediction{1},'Kangaroo')==0               %ned to add weights
%             notKangaroosCount = notKangaroosCount + lbpProbScore + 3;
%          else
             notKangaroosCount = notKangaroosCount + hogProbScore + 3; 
%          end
     end
            
     if strcmp(lbpPrediction{1},'Kangaroo')==1                %ned to add weights
         kangaroosCount = kangaroosCount + lbpProbScore + 3;
     else
         notKangaroosCount = notKangaroosCount + lbpProbScore + 1;
     end
             
    %perform majority voting computation
    if kangaroosCount > notKangaroosCount
        finalTestPredictions{i} = 'Kangaroo';
    else
       finalTestPredictions{i} = 'Not Kangaroo';
    end
end

disp('Combining classifiers test results and computing confusion matrix......')
disp('-----------------------------------------------------------------------')
%combined classifier test results and confusion matrix
confMatrix4test = confusionmat(Ytest,finalTestPredictions);
%compute test accuracy
truePositiveAndNegatives = confMatrix4test(1) + confMatrix4test(4);
testAccuracy = truePositiveAndNegatives / numel(Ytest);
disp(testAccuracy)
disp('Test Confusion Matrix:')
disp(confMatrix4test)
truePositiveRate = confMatrix4test(1) /(confMatrix4test(1)+confMatrix4test(3));
falsePositiveRate =confMatrix4test(2) /(confMatrix4test(2)+confMatrix4test(4));
finalResults ={'Test Dataset',num2str(truePositiveRate),num2str(falsePositiveRate),num2str(testAccuracy)}
toc  %close a clock 