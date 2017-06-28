function [binomVector,finalResults,NoOfSuccesses,confMatrix4test] = nightFinalEnsembledClassifier(nightClassifiersList,testColorFeat,testHOGFeat,testLBPFeat,testGISTFeat,testSIFTFeat,Ytest)
    
finalResults = [];
finalTestPredictions =[];
%using combined classifiers to predict a label
for i=1:length(Ytest)
    kangaroosCount = 0;
    notKangaroosCount = 0;
    probs = [];
    equalVotes = 0;
     
    %HOG Classification prediction
      for k=1:3
         [hogPrediction,hogProbScore] = predict(nightClassifiersList{k},testHOGFeat(i,:));
         if k==1
             if strcmp(hogPrediction{1},'Kangaroo')==1                %ned to add weights
                 kangaroosCount = kangaroosCount + 4;
             else 
                notKangaroosCount = notKangaroosCount + 4.1;
             end
         elseif k==2
             if strcmp(hogPrediction{1},'Kangaroo')==1                %ned to add weights
                 kangaroosCount = kangaroosCount +  1;
             else 
                notKangaroosCount = notKangaroosCount + 1;
             end
         elseif k==3
             if strcmp(hogPrediction{1},'Kangaroo')==1                %ned to add weights
                 kangaroosCount = kangaroosCount +  3;
             else 
                notKangaroosCount = notKangaroosCount + 3.1;
             end
         end     
      end
     
      %GIST Classification prediction
      for j=4:5
         [gistPrediction,gistProbScore] = predict(nightClassifiersList{j},testGISTFeat(i,:));
         
         if j==4
             if strcmp(gistPrediction{1},'Kangaroo')==1                %ned to add weights
             kangaroosCount = kangaroosCount + 1;
             else 
                notKangaroosCount = notKangaroosCount + 1;
             end
         else
             if strcmp(gistPrediction{1},'Kangaroo')==1                %ned to add weights
                kangaroosCount = kangaroosCount + 1;
             else 
                notKangaroosCount = notKangaroosCount + 1;
             end
         end   
       end
   
 %display final counts
    disp(kangaroosCount)
    disp(notKangaroosCount)
    if kangaroosCount > notKangaroosCount
        finalTestPredictions{i} = 'Kangaroo';
    else
        finalTestPredictions{i} = 'Not Kangaroo';

    end
    
    if kangaroosCount == notKangaroosCount
        equalVotes = equalVotes + 1;
        disp('Equal votes found')
    end
end
disp('-----------------------------------------------------------Equal votes---------------------------------------------')
 disp(equalVotes)
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

%COMPUTING NUMBERS FOR THE BINOMIAL PROBABILITY ANALYSIS
binomVector =[];
NoOfSuccesses =0;
actualNoOfKangaroos =0;
actualNoOfNotKangaroos =0;
totalNoOfTestImages = length(Ytest);
for i=1:length(Ytest)
    if strcmp(Ytest{i},'Kangaroo')==1
        actualNoOfKangaroos = actualNoOfKangaroos + 1;
    else
        actualNoOfNotKangaroos = actualNoOfNotKangaroos +1;
    end
    if strcmp(Ytest{i},'Kangaroo')==1 && strcmp(finalTestPredictions{i},'Kangaroo')==1
        binomVector{i} = 1;
        NoOfSuccesses = NoOfSuccesses + 1;
    elseif strcmp(Ytest{i},'Not Kangaroo')==1 && strcmp(finalTestPredictions{i},'Not Kangaroo')==1
        binomVector{i} = 1;
        NoOfSuccesses = NoOfSuccesses + 1;
    else
        binomVector{i} = 0;
    end
end 
binomVector = binomVector';
toc  %close a clock 