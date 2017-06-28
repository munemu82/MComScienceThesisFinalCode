function [trainingSet,testSet,trainLabels,testLabels,train_labels,train_num_labels,test_labels,test_num_labels] = processImages(rootFolder,destFolder,categories,
)
%validate specified training and test folders if exit
if ~isdir(rootFolder)            % Begin of main condition to be met for the rest of codes to run
    errorMessage = sprintf('Error: The following folder does not exist:\n%s', rootFolder);
    uiwait(warndlg(errorMessage));
    return;
else
    %Categories of images in the root folder - each category must exist as a folder in the root folder
    imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

    %Check quick summary of data for the categories
    originalDataLabels = countEachLabel(imds)

    %Because imds above contains an unequal number of images per category, let's first adjust it, so that the number of images in the training set is balanced.

    minSetCount = min(originalDataLabels{:,2}); % determine the smallest amount of images in a category

    % Use splitEachLabel method to trim the set.
    imds = splitEachLabel(imds, minSetCount, 'randomize');

    % Notice that each set now has exactly the same number of images.
    balancedDataLabels = countEachLabel(imds)

    %Prepare Training and Test Image Sets
    %Split the sets into training and validation data. Pick 260 of images from each label set for the training data and the remainder, 70%, 
    %for the validation data. Randomize the split to avoid biasing the %%%%results. The training and test sets will be processed by the CNN model.
    [trainingSet, testSet] = splitEachLabel(imds,numOfTrainingImages, 'randomize');

    %Check quick summary of each dataset
    trainLabels = countEachLabel(trainingSet)
    testLabels = countEachLabel(testSet)
    % VARIABLES REQUIRED FOR FEATURE EXTRACTION TOOLBOX
    %Declare lists
    train_labels= [];
    train_num_labels = [];
    test_labels = [];
    test_num_labels =[];
    sets = {length(trainingSet.Files), length(testSet.Files)}
    %PREPARE REQUIRED VARIABLES TRAINING SET
    for j=1:length(sets)                %Begin of image sets for LOOP

        for i = 1:sets{j}   % Begin of specific set (training or test set) iteration for LOOP

          if j==1  %Begin training set processing 
              %get original image path
              fullImagePath = trainingSet.Files(i);

              %Display training images
              disp(fullImagePath);

              %Reading Images
              imageArray = imread(fullImagePath{1});
              %moving the images to the training folder
              newfullImagePath = strrep(fullImagePath, cellstr(trainingSet.Labels(i)), 'trainTemp');
              fprintf(1, 'Now saving image %s\n',' to train folder');
              imwrite(imageArray,newfullImagePath{1});
          else  %hard coded for test set processing
              %get original image path
              fullImagePath = testSet.Files(i);
              %Display test images
              disp(fullImagePath);
              %Reading Images
              imageArray = imread(fullImagePath{1});
              %moving the images to the training folder
              newfullImagePath = strrep(fullImagePath, cellstr(testSet.Labels(i)), 'testTemp');
              fprintf(1, 'Now saving image %s\n',' to train folder');
              imwrite(imageArray,newfullImagePath{1});
          end %End of training set processing 

          %EXTRACT LABELS FROM EACH IMAGE SET (TRAINING AND TEST SETS)
         if j==1           %add list of training images paths to matrix
              if trainingSet.Labels(i) =='Kangaroos'
                  train_labels{i} = 'Kangaroo';    %a kangaroo catagorical label
                  train_num_labels(i) = 1;         %a kangaroo numerical label
              else
                  train_labels{i} = 'Not Kangaroo'; %not a kangaroo catagorical label
                  train_num_labels(i) = 0;          %Not a kangaroo numerical label
              end
          else              %add list of test images paths to matrix
              if testSet.Labels(i) =='Kangaroos'
                  test_labels{i} = 'Kangaroo';    %a kangaroo catagorical label
                  test_num_labels(i) = 1;         %a kangaroo numerical label
              else
                  test_labels{i} = 'Not Kangaroo';    %not a kangaroo catagorical label
                  test_num_labels(i) = 0;             %Not a kangaroo numerical label 
              end
              
          end   %End of adding image paths to the matrix
          
        end  %End of specific set (training or test set) iteration for LOOP
        
    end    %End of Main image sets for LOOP
        
end  %End of Main condition to be met 
