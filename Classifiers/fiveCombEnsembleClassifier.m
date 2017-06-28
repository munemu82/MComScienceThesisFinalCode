function [finalResults,binomVector,NoOfSuccesses] = fiveCombEnsembleClassifier(FiveCombClassifiersList,testHOGFeat, testColorFeat, testSIFTFeat,testGISTFeat,testLBPFeat,test_labels)
tic    
finalResults = [];
finalTestPredictions =[];

%using combined classifiers to predict a label
for i=1:length(test_labels)
    kangaroosCount = 0;
    notKangaroosCount = 0;
    equalVotes = 0;
     %for HOG feature
     for j=1:length(FiveCombClassifiersList)
         if j == 1 %for gist feature
             [classPrediction,predProScore] = predict(FiveCombClassifiersList{j},testHOGFeat(i,:));
      
            if strcmp(classPrediction{1},'Kangaroo')==1                %ned to add weights
                kangaroosCount = kangaroosCount + 2.5;
             else 
                notKangaroosCount = notKangaroosCount + 2.5;                                                          %              ;
            end 
         
         elseif j == 2 %for gist feature
             [classPrediction,predProScore] = predict(FiveCombClassifiersList{j},testHOGFeat(i,:));
      
            if strcmp(classPrediction{1},'Kangaroo')==1                %ned to add weights
                kangaroosCount = kangaroosCount + 2;
             else 
                notKangaroosCount = notKangaroosCount + 2;                                                          %              ;
            end 
         elseif j==4   %for HOG feature
              [classPrediction,predProScore] = predict(FiveCombClassifiersList{j},testGISTFeat(i,:));
      
            if strcmp(classPrediction{1},'Kangaroo')==1                %ned to add weights
                kangaroosCount = kangaroosCount + 1;
             else 
                notKangaroosCount = notKangaroosCount + 1;                                                          %              ;
            end 
        elseif j==5    %for HOG feature
              [classPrediction,predProScore] = predict(FiveCombClassifiersList{j},testHOGFeat(i,:));
      
            if strcmp(classPrediction{1},'Kangaroo')==1                %ned to add weights
                kangaroosCount = kangaroosCount + 1;
             else 
                notKangaroosCount = notKangaroosCount + 1;                                                          %              ;
            end 
%          elseif j==4   %for HOG feature
%               [classPrediction,predProScore] = predict(FiveCombClassifiersList{j},testLBPFeat(i,:));
%       
%             if strcmp(classPrediction{1},'Kangaroo')==1                %ned to add weights
%                 kangaroosCount = kangaroosCount + 1;
%              else 
%                 notKangaroosCount = notKangaroosCount + 1;                                                          %              ;
%             end 
%          elseif j==3    %for HOG feature
%               [classPrediction,predProScore] = predict(FiveCombClassifiersList{j},testHOGFeat(i,:));
%       
%             if strcmp(classPrediction{1},'Kangaroo')==1                %ned to add weights
%                 kangaroosCount = kangaroosCount + 2.1;
%              else 
%                 notKangaroosCount = notKangaroosCount + 2.1;                                                          %              ;
%             end 
%          else    %for HOG feature
%               [classPrediction,predProScore] = predict(FiveCombClassifiersList{j},testLBPFeat(i,:));
%       
%             if strcmp(classPrediction{1},'Kangaroo')==1                %ned to add weights
%                 kangaroosCount = kangaroosCount + 1;
%              else 
%                 notKangaroosCount = notKangaroosCount + 1;                                                          %              ;
%             end 

         end
             
          
     end
    
    
 %display final counts
    disp(kangaroosCount)
    disp(notKangaroosCount)
    if kangaroosCount > notKangaroosCount
        finalTestPredictions{i} = 'Kangaroo';
    else
%         if kangaroosCount == notKangaroosCount  %where equal votes, then select randomnly
% %             label = Ytest(randi(numel(Ytest)));
% %             finalTestPredictions{i} = label{1};
%               if gistProbScore(1) > hogProbScore(1) || lbpProbScore(1) > hogProbScore(1) 
%                   finalTestPredictions{i} = 'Kangaroo';
%               else
%                    finalTestPredictions{i} = 'Not Kangaroo';
%               end
%                   
%         else
%             
            finalTestPredictions{i} = 'Not Kangaroo';
%         end
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
confMatrix4test = confusionmat(test_labels,finalTestPredictions);
%compute test accuracy
truePositiveAndNegatives = confMatrix4test(1) + confMatrix4test(4);
testAccuracy = truePositiveAndNegatives / numel(test_labels);
disp(testAccuracy)
disp('Test Confusion Matrix:')
disp(confMatrix4test)
truePositiveRate = confMatrix4test(1) /(confMatrix4test(1)+confMatrix4test(3));
falsePositiveRate =confMatrix4test(2) /(confMatrix4test(2)+confMatrix4test(4));
finalResults ={'Test Dataset',num2str(truePositiveRate),num2str(falsePositiveRate),num2str(testAccuracy)}

x`
toc  %close a clock 