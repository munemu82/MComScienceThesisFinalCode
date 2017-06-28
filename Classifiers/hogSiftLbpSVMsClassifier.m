function [hogSiftLbpSVMClassifiers,modelPredAgreementCount,modelPredDisagreementCount,onlyTwoAgreement,allThreeAgremement,lbp_HogAgreement,sift_HogAgreement] = hogSiftLbpSVMsClassifier(trainSIFTFeat,trainHOGFeat,trainLBPFeat,train_labels)
%Timer for comparison between ensembled classifier vs Deep learning training
tic          %set a clock

%Training data matrix for each feature
Ytrain = train_labels;          % Common labels indicating whether image is kangaroo or not Kangaroos
XtrainSIFT = trainSIFTFeat;   % n-by-m matrix containing Color features representing image  
XtrainHOG = trainHOGFeat;       % n-by-m matrix containing HOG features representing image  
XtrainLBP = trainLBPFeat;     % n-by-m matrix containing GIST features representing image

modelPredAgreementCount = 0;
modelPredDisagreementCount = 0;
onlyTwoAgreement = 0;
allThreeAgremement = 0;
sift_HogAgreement = 0;
lbp_HogAgreement = 0;

%SIFT FEATURE
disp('Currently training classifiers .........')
%train the classifier 1 -  SIFT-SVM  (using rbf kernel function) and
sift_SVM_classifier = fitcsvm(XtrainSIFT,Ytrain, 'KernelFunction','gaussian', ...
    'KernelScale', 91,'BoxConstraint',5, 'Standardize', true, ...
    'ClassNames', {'Kangaroo'; 'Not Kangaroo'});
%HOG FEATURE 
%train the classifier 4 -  HOG-SVM  (using gaussian kernel function)
hog_SVM_classifier = fitcsvm(XtrainHOG,Ytrain, 'KernelFunction','polynomial', ...
    'PolynomialOrder',3,'KernelScale','auto','BoxConstraint',5, 'Standardize', true, ...
    'ClassNames', {'Kangaroo'; 'Not Kangaroo'});

%LBP FEATURE

%train the model 7 -  LBP-SVM  (using Gaussian kernel)siftSVMCVModel
lbp_SVM_classifier = fitcsvm(XtrainLBP,Ytrain, 'KernelFunction','gaussian', ...
    'KernelScale', 23,'BoxConstraint',5, 'Standardize', true, ...
    'ClassNames', {'Kangaroo'; 'Not Kangaroo'});

disp('Training classifiers complete!')
disp('.............................................................................. .........')

disp('Currently cross validating classifiers .........')
%Cross validation of each classifiers
siftSVMCVModel = crossval(sift_SVM_classifier, 'KFold', 10);
hogSVMCVModel = crossval(hog_SVM_classifier, 'KFold', 10);
lbpSVMCVModel = crossval(lbp_SVM_classifier, 'KFold', 10);
disp('Cross validation of classifiers complete!')

disp('Currently computing classifiers cross validation predictions.........')
%Cross validation Classifiers accuracies
[siftSVMCVModelValPredictions, siftSVMCVModelValScores] = kfoldPredict(siftSVMCVModel);
[hogSVMCVModelValPredictions, hogSVMCVModelValScores] = kfoldPredict(hogSVMCVModel);
[lbpSVMCVModelValPredictions, lbpSVMCVModelValScores] = kfoldPredict(lbpSVMCVModel);
disp('Cross validation predictions of classifiers complete!')

%Cross validation Confusion matrices for each classifiers
disp('Computing models cross validation confusion matrices for each of the classifiers')
disp('--------------------------------------------------------------------')
siftSVMConfMatrix = confusionmat(Ytrain,siftSVMCVModelValPredictions)
hogConfMatrix = confusionmat(Ytrain,hogSVMCVModelValPredictions)
lbpSVMConfMatrix = confusionmat(Ytrain,lbpSVMCVModelValPredictions)

%add cross validated classifiers to the list
hogSiftLbpSVMClassifiers = [];
hogSiftLbpSVMClassifiers{1} = sift_SVM_classifier;
hogSiftLbpSVMClassifiers{2} = hog_SVM_classifier;
hogSiftLbpSVMClassifiers{3} = lbp_SVM_classifier;

disp('Computing cross validation binomial trials.................')
%Binomial trials on cross validation
model1Binom = [];
model2Binom = [];
model3Binom = [];
for i=1:length(Ytrain)
    %for Model 1
   if strcmp(Ytrain{i},'Kangaroo')==1 && strcmp(siftSVMCVModelValPredictions{i},'Kangaroo')==1
        model1Binom{i} = 1;
    elseif strcmp(Ytrain{i},'Not Kangaroo')==1 && strcmp(siftSVMCVModelValPredictions{i},'Not Kangaroo')==1
        model1Binom{i} = 1;
    else
        model1Binom{i} = 0;
   end
   %for Model 2
   if strcmp(Ytrain{i},'Kangaroo')==1 && strcmp(hogSVMCVModelValPredictions{i},'Kangaroo')==1
        model2Binom{i} = 1;
    elseif strcmp(Ytrain{i},'Not Kangaroo')==1 && strcmp(hogSVMCVModelValPredictions{i},'Not Kangaroo')==1
        model2Binom{i} = 1;
    else
        model2Binom{i} = 0;
   end
   %for Model 3
   if strcmp(Ytrain{i},'Kangaroo')==1 && strcmp(lbpSVMCVModelValPredictions{i},'Kangaroo')==1
        model3Binom{i} = 1;
    elseif strcmp(Ytrain{i},'Not Kangaroo')==1 && strcmp(lbpSVMCVModelValPredictions{i},'Not Kangaroo')==1
        model3Binom{i} = 1;
    else
        model3Binom{i} = 0;
   end
end
disp('Computing cross validation binomial trials complete!')

disp('Now comparing cross validation predictions between classifiers for model agreements..........')
%compare the number of times each classification models agrees on a cross-validated set
%models combinations should be used for predicitons
for k=1:length(Ytrain)
    if model1Binom{k}== model2Binom{k} 
        modelPredAgreementCount = modelPredAgreementCount + 1;
        onlyTwoAgreement = onlyTwoAgreement + 1;
        sift_HogAgreement = sift_HogAgreement + 1;
    elseif model1Binom{k}== model3Binom{k}
        modelPredAgreementCount = modelPredAgreementCount + 1;
        onlyTwoAgreement = onlyTwoAgreement + 1;
    elseif model2Binom{k}== model3Binom{k}
        modelPredAgreementCount = modelPredAgreementCount + 1;
        onlyTwoAgreement = onlyTwoAgreement + 1;
        lbp_HogAgreement = lbp_HogAgreement + 1;
    elseif model2Binom{k}== model3Binom{k} && model2Binom{k}== model1Binom{k}
        modelPredAgreementCount = modelPredAgreementCount + 1;
        allThreeAgremement = allThreeAgremement + 1;
    else
        modelPredDisagreementCount = modelPredDisagreementCount +1;
    end
end
