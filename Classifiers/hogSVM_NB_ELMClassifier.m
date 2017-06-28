function [hogSVM_NB_ELMClassifiers,modelPredAgreementCount,modelPredDisagreementCount,onlyTwoAgreement,allThreeAgremement,nb_SvmAgreement,elm_SvmAgreement] = hogSVM_NB_ELMClassifier(trainHOGFeat,train_labels,train_num_labels)
%Timer for comparison between ensembled classifier vs Deep learning training
tic          %set a clock

%Training data matrix for each feature
Ytrain = train_labels;          % Common labels indicating whether image is kangaroo or not Kangaroos
Ynum_train = train_num_labels;   % numerical labels 0, 1 whereby 1=Kangaroo and 0=Not Kangaroo required for ELM classifier
XtrainHOG = trainHOGFeat;       % n-by-m matrix containing HOG features representing image  

modelPredAgreementCount = 0;
modelPredDisagreementCount = 0;
onlyTwoAgreement = 0;
allThreeAgremement = 0;
nb_SvmAgreement = 0;
elm_SvmAgreement = 0;

%prepare data for ELM model
trainingDataset  = [train_num_labels,XtrainHOG]; 
dlmwrite('dayHogTrainingData.txt',trainingDataset)


disp('Currently training classifiers .........')

hog_SVM_classifier = fitcsvm(XtrainHOG,Ytrain, 'KernelFunction','polynomial', ...
    'PolynomialOrder',3,'KernelScale','auto','BoxConstraint',5, 'Standardize', true, ...
    'ClassNames', {'Kangaroo'; 'Not Kangaroo'});

hog_NB_classifier = fitcnb(XtrainHOG,Ytrain, 'distribution','kernel','kernel',[],'kernel','normal');

[TrainingTime,TrainingAccuracy] = elm_train('dayHogTrainingData.txt', 1,100,'sin');
    
disp('Training classifiers complete!')
disp('.............................................................................. .........')

disp('Currently cross validating classifiers .........')
%Cross validation of each classifiers
hogNBCVModel = crossval(hog_NB_classifier, 'KFold', 10);
hogSVMCVModel = crossval(hog_SVM_classifier, 'KFold', 10);

disp('Cross validation of classifiers complete!')

disp('Currently computing classifiers cross validation predictions.........')
%Cross validation Classifiers accuracies
[hogNBCVModelValPredictions, hogNBCVModelValScores] = kfoldPredict(hogNBCVModel);
[hogSVMCVModelValPredictions, hogSVMCVModelValScores] = kfoldPredict(hogSVMCVModel);
[TestingTime, TestingAccuracy] = elm_predict('dayHogTrainingData.txt');
load('elm_output.mat');      %load the model ELM model predictions output 
hogELMCVModelValPredictions = output';         %transpose the model output to match our actual labe

disp('Cross validation predictions of classifiers complete!')

%Cross validation Confusion matrices for each classifiers
disp('Computing models cross validation confusion matrices for each of the classifiers')
disp('--------------------------------------------------------------------')
hogSVMConfMatrix = confusionmat(Ytrain,hogNBCVModelValPredictions)
hogSVMConfMatrix = confusionmat(Ytrain,hogSVMCVModelValPredictions)
hogELMConfMatrix = confusionmat(Ynum_train,hogELMCVModelValPredictions)

%add cross validated classifiers to the list
hogSVM_NB_ELMClassifiers = [];
hogSVM_NB_ELMClassifiers{1} = hog_NB_classifier;
hogSVM_NB_ELMClassifiers{2} = hog_SVM_classifier;


disp('Computing cross validation binomial trials.................')
%Binomial trials on cross validation
model1Binom = [];
model2Binom = [];
model3Binom = [];
for i=1:length(Ytrain)
    %for Model 1
   if strcmp(Ytrain{i},'Kangaroo')==1 && strcmp(hogNBCVModelValPredictions{i},'Kangaroo')==1
        model1Binom{i} = 1;
    elseif strcmp(Ytrain{i},'Not Kangaroo')==1 && strcmp(hogNBCVModelValPredictions{i},'Not Kangaroo')==1
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
   if Ynum_train(i) ==1 && hogELMCVModelValPredictions(i)==1
        model3Binom{i} = 1;
    elseif Ynum_train(i)==0 && hogELMCVModelValPredictions(i) ==0
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
        nb_SvmAgreement = nb_SvmAgreement + 1;
    elseif model1Binom{k}== model3Binom{k}
        modelPredAgreementCount = modelPredAgreementCount + 1;
        onlyTwoAgreement = onlyTwoAgreement + 1;
    elseif model2Binom{k}== model3Binom{k}
        modelPredAgreementCount = modelPredAgreementCount + 1;
        onlyTwoAgreement = onlyTwoAgreement + 1;
        elm_SvmAgreement = elm_SvmAgreement + 1;
    elseif model2Binom{k}== model3Binom{k} && model2Binom{k}== model1Binom{k}
        modelPredAgreementCount = modelPredAgreementCount + 1;
    else
        modelPredDisagreementCount = modelPredDisagreementCount +1;
        allThreeAgremement = allThreeAgremement + 1;
    end
end
