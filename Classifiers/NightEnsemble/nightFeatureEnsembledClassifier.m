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

%COLOR FEATURE

%train the classifier 1 -  color-SVM  (using rbf kernel function)
color_SVM_classifier = fitcsvm(XtrainColor,Ytrain, 'KernelFunction','rbf', ...
    'KernelScale', 'Auto','BoxConstraint',5, 'Standardize', true, ...
    'ClassNames', {'Kangaroo'; 'Not Kangaroo'});

%train the classifier 2  - color -Adaboost
template = templateTree('MaxNumSplits', 20); 
color_adaboost_classifier = fitensemble(XtrainColor,Ytrain,'LogitBoost',30, ...   
template,'Type', 'Classification','ClassNames', {'Kangaroo'; 'Not Kangaroo'});

%train the classifier 3  - color -Random Forest
color_randomForest_classifier = fitensemble(XtrainColor,Ytrain,'Bag',125, ...
        'Tree', 'Type', 'Classification','ClassNames', {'Kangaroo'; 'Not Kangaroo'});

%HOG FEATURE

%train the classifier 4 -  HOG-SVM  (using gaussian kernel function)
hog_SVM_classifier = fitcsvm(XtrainHOG,Ytrain, 'KernelFunction','polynomial', ...
    'PolynomialOrder',3,'KernelScale','auto','BoxConstraint',5, 'Standardize', true, ...
    'ClassNames', {'Kangaroo'; 'Not Kangaroo'});

%train the classifier 5  - HOG -Adaboost
template = templateTree('MaxNumSplits', 35); 
hog_adaboost_classifier = fitensemble(XtrainHOG,Ytrain,'GentleBoost',35, ...   
template,'Type', 'Classification','ClassNames', {'Kangaroo'; 'Not Kangaroo'});

%train the classifier 6  - HOG -Random Forest
hog_randomForest_classifier = fitensemble(XtrainHOG,Ytrain,'Bag',125, ...
        'Tree', 'Type', 'Classification','ClassNames', {'Kangaroo'; 'Not Kangaroo'});
    
%SIFT FEATURE

%train the model 7 -  SIFT-SVM  (using Gaussian kernel)
lbp_SVM_classifier = fitcsvm(XtrainLBP,Ytrain, 'KernelFunction','gaussian', ...
    'KernelScale', 91,'BoxConstraint',5, 'Standardize', true, ...
    'ClassNames', {'Kangaroo'; 'Not Kangaroo'});

%train the classifier 8  - SIFT -Adaboost
template = templateTree('MaxNumSplits', 15); 
lbp_adaboost_classifier = fitensemble(XtrainLBP,Ytrain,'LogitBoost',35, ...   
template,'Type', 'Classification','ClassNames', {'Kangaroo'; 'Not Kangaroo'});

%train the classifier 9  - HOG -Random Forest
lbp_randomForest_classifier = fitensemble(XtrainLBP,Ytrain,'Bag',150, ...
        'Tree', 'Type', 'Classification','ClassNames', {'Kangaroo'; 'Not Kangaroo'});
    
    %GIST FEATURE

%train the model 7 -  SIFT-SVM  (using Gaussian kernel)
gist_SVM_classifier = fitcsvm(XtrainGIST,Ytrain, 'KernelFunction','gaussian', ...
    'KernelScale', 91,'BoxConstraint',5, 'Standardize', true, ...
    'ClassNames', {'Kangaroo'; 'Not Kangaroo'});

%train the classifier 8  - SIFT -Adaboost
template = templateTree('MaxNumSplits', 35); 
gist_adaboost_classifier = fitensemble(XtrainGIST,Ytrain,'AdaBoostM1',35, ...   
template,'Type', 'Classification','ClassNames', {'Kangaroo'; 'Not Kangaroo'});

%train the classifier 9  - HOG -Random Forest
gist_randomForest_classifier = fitensemble(XtrainGIST,Ytrain,'Bag',125, ...
        'Tree', 'Type', 'Classification','ClassNames', {'Kangaroo'; 'Not Kangaroo'});

 % Perform cross-validation for color models
 disp('Performing cross validation for color models')
 cvColorModel1 = crossval(color_SVM_classifier, 'KFold', 10);
 cvColorModel2 = crossval(color_adaboost_classifier, 'KFold', 10);
 cvColorModel3 = crossval(color_randomForest_classifier, 'KFold', 10);

  disp('Performing cross validation for Hog models')
 % Perform cross-validation for HOG modesl
 cvHogModel1 = crossval(hog_SVM_classifier, 'KFold', 10);
 cvHogModel2 = crossval(hog_adaboost_classifier, 'KFold', 10);
 cvHogModel3 = crossval(hog_randomForest_classifier, 'KFold', 10);
 
  disp('Performing cross validation for SIFT models')
  % Perform cross-validation for SIFT models
 cvSiftModel1 = crossval(lbp_SVM_classifier, 'KFold', 10);
 cvSiftModel2 = crossval(lbp_adaboost_classifier, 'KFold', 10);
 cvSiftModel3 = crossval(lbp_randomForest_classifier, 'KFold', 10);
 
 % Perform cross-validation for color models
 disp('Performing cross validation for color models')
 cvGistModel1 = crossval(gist_SVM_classifier, 'KFold', 10);
 cvGistModel2 = crossval(gist_adaboost_classifier, 'KFold', 10);
 cvGistModel3 = crossval(gist_randomForest_classifier, 'KFold', 10);
 
  disp('Computing cross validation predictions for color models')
  [colorModel1CVPredictions, colorModel1CVScores] = kfoldPredict(cvColorModel1);
  [colorModel2CVPredictions, colorModel2CVScores] = kfoldPredict(cvColorModel2);
  [colorModel3CVPredictions, colorModel3CVScores] = kfoldPredict(cvColorModel3);
  
    disp('Computing cross validation predictions for HOG models')
  [hogModel1CVPredictions, hogModel1CVScores] = kfoldPredict(cvHogModel1);
  [hogModel2CVPredictions, hogModel2CVScores] = kfoldPredict(cvHogModel2);
  [hogModel3CVPredictions, hogModel3CVScores] = kfoldPredict(cvHogModel3);
  
    disp('Computing cross validation predictions for SIFT models')
  [lbpModel1CVPredictions, lbpModel1CVScores] = kfoldPredict(cvSiftModel1);
  [lbpModel2CVPredictions, lbpModel2CVScores] = kfoldPredict(cvSiftModel2);
  [lbpModel3CVPredictions, lbpModel3CVScores] = kfoldPredict(cvSiftModel3);
  disp('Computing cross validation predictions for GIST models')
  [gistModel1CVPredictions, gistModel1CVScores] = kfoldPredict(cvGistModel1);
  [gistModel2CVPredictions, gistModel2CVScores] = kfoldPredict(cvGistModel2);
  [gistModel3CVPredictions, gistModel3CVScores] = kfoldPredict(cvGistModel3);



%add classifiers to the list
dayClassifiersList = [];
dayClassifiersList{1} = color_SVM_classifier;
dayClassifiersList{2} = color_adaboost_classifier;
dayClassifiersList{3} = color_randomForest_classifier;
dayClassifiersList{4} = hog_SVM_classifier;
dayClassifiersList{5} = hog_adaboost_classifier;
dayClassifiersList{6} = hog_randomForest_classifier;
dayClassifiersList{7} = gist_SVM_classifier;
dayClassifiersList{8} = gist_adaboost_classifier;
dayClassifiersList{9} = gist_randomForest_classifier;
dayClassifiersList{10} = lbp_SVM_classifier;
dayClassifiersList{11} = lbp_adaboost_classifier;
dayClassifiersList{12} = lbp_randomForest_classifier;


%Color Models correlations
colorModelsPredAgreement = 0;
colorAtleatTwoAgreement = 0;
colorModelsDisagreement = 0;
colorModel1AND2Agreement = 0;
colorModel1AND3Agreement = 0;
colorModel2AND3Agreement = 0;
for i=1:length(train_labels)
    if strcmp(colorModel1CVPredictions{i},colorModel2CVPredictions{i})==1  % 1 and 2 are equal
        colorAtleatTwoAgreement = colorAtleatTwoAgreement + 1;
        colorModel1AND2Agreement = colorModel1AND2Agreement + 1;
    elseif strcmp(colorModel1CVPredictions{i},colorModel3CVPredictions{i})==1 %1 and 3 are equal
       colorAtleatTwoAgreement = colorAtleatTwoAgreement + 1;
       colorModel1AND3Agreement = colorModel1AND3Agreement + 1;
    elseif strcmp(colorModel2CVPredictions{i},colorModel3CVPredictions{i})==1 %2 and 3 are equal
       colorAtleatTwoAgreement = colorAtleatTwoAgreement + 1;
       colorModel2AND3Agreement = colorModel2AND3Agreement + 1;
    elseif strcmp(colorModel2CVPredictions{i},colorModel3CVPredictions{i})==1 && ...
            strcmp(colorModel2CVPredictions{i},colorModel1CVPredictions{i})==1  % 1, 2 and 3 are equal
        colorModelsPredAgreement = colorModelsPredAgreement + 1;
    else
        colorModelsDisagreement = colorModelsDisagreement + 1;
    end
end

%HOG Models correlations
hogModelsPredAgreement = 0;
hogAtleatTwoAgreement = 0;
hogModelsDisagreement = 0;
hogModel1AND2Agreement = 0;
hogModel1AND3Agreement = 0;
hogModel2AND3Agreement = 0;
for j=1:length(train_labels)
    if strcmp(hogModel1CVPredictions{j},hogModel2CVPredictions{j})==1  % 1 and 2 are equal
        hogAtleatTwoAgreement = hogAtleatTwoAgreement + 1;
        hogModel1AND2Agreement = hogModel1AND2Agreement + 1;
    elseif strcmp(hogModel1CVPredictions{j},hogModel3CVPredictions{j})==1 %1 and 3 are equal
       hogAtleatTwoAgreement = hogAtleatTwoAgreement + 1;
       hogModel1AND3Agreement = hogModel1AND3Agreement + 1;
    elseif strcmp(hogModel2CVPredictions{j},hogModel3CVPredictions{j})==1 %2 and 3 are equal
       hogAtleatTwoAgreement = hogAtleatTwoAgreement + 1;
       hogModel2AND3Agreement = hogModel2AND3Agreement + 1;
    elseif strcmp(hogModel2CVPredictions{j},hogModel3CVPredictions{j})==1 && ...
            strcmp(hogModel2CVPredictions{j},hogModel1CVPredictions{j})==1  % 1, 2 and 3 are equal
        hogModelsPredAgreement = hogModelsPredAgreement + 1;
    else
        hogModelsDisagreement = hogModelsDisagreement + 1;
    end
end

%SIFT Models correlations
lbpModelsPredAgreement = 0;
lbpAtleatTwoAgreement = 0;
lbpModelsDisagreement = 0;
lbpModel1AND2Agreement = 0;
lbpModel1AND3Agreement = 0;
lbpModel2AND3Agreement = 0;
for k=1:length(train_labels)
    if strcmp(lbpModel1CVPredictions{k},lbpModel2CVPredictions{k})==1  % 1 and 2 are equal
        lbpAtleatTwoAgreement = lbpAtleatTwoAgreement + 1;
        lbpModel1AND2Agreement = lbpModel1AND2Agreement + 1;
    elseif strcmp(lbpModel1CVPredictions{k},lbpModel3CVPredictions{k})==1 %1 and 3 are equal
       lbpAtleatTwoAgreement = lbpAtleatTwoAgreement + 1;
       lbpModel1AND3Agreement = lbpModel1AND3Agreement + 1;
    elseif strcmp(lbpModel2CVPredictions{k},lbpModel3CVPredictions{k})==1 %2 and 3 are equal
       lbpAtleatTwoAgreement = lbpAtleatTwoAgreement + 1;
       lbpModel2AND3Agreement = lbpModel2AND3Agreement + 1;
    elseif strcmp(lbpModel2CVPredictions{k},lbpModel3CVPredictions{k})==1 && ...
            strcmp(lbpModel2CVPredictions{k},lbpModel1CVPredictions{k})==1  % 1, 2 and 3 are equal
        lbpModelsPredAgreement = lbpModelsPredAgreement + 1;
    else
        lbpModelsDisagreement = lbpModelsDisagreement + 1;
    end
end

%GIST Models correlations
gistModelsPredAgreement = 0;
gistAtleatTwoAgreement = 0;
gistModelsDisagreement = 0;
gistModel1AND2Agreement = 0;
gistModel1AND3Agreement = 0;
gistModel2AND3Agreement = 0;
for i=1:length(train_labels)
    if strcmp(gistModel1CVPredictions{i},gistModel2CVPredictions{i})==1  % 1 and 2 are equal
        gistAtleatTwoAgreement = gistAtleatTwoAgreement + 1;
        gistModel1AND2Agreement = gistModel1AND2Agreement + 1;
    elseif strcmp(gistModel1CVPredictions{i},gistModel3CVPredictions{i})==1 %1 and 3 are equal
       gistAtleatTwoAgreement = gistAtleatTwoAgreement + 1;
       gistModel1AND3Agreement = gistModel1AND3Agreement + 1;
    elseif strcmp(gistModel2CVPredictions{i},gistModel3CVPredictions{i})==1 %2 and 3 are equal
       gistAtleatTwoAgreement = gistAtleatTwoAgreement + 1;
       gistModel2AND3Agreement = gistModel2AND3Agreement + 1;
    elseif strcmp(gistModel2CVPredictions{i},gistModel3CVPredictions{i})==1 && ...
            strcmp(gistModel2CVPredictions{i},gistModel1CVPredictions{i})==1  % 1, 2 and 3 are equal
        gistModelsPredAgreement = gistModelsPredAgreement + 1;
    else
        gistModelsDisagreement = gistModelsDisagreement + 1;
    end
end

