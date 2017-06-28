%load('newDayImagesColorFeatures.mat') %load the data required
X = train_features;     % Feature vectors 
Y = train_num_labels;       % Train class labels 
rng(1945,'twister')
leaf = [1 5 10 20 50];
col = 'rbcmy';
figure       %%This plots out of the bag classification error for each leaf size given number of trees
for i=1:length(leaf)
    forest = TreeBagger(100,X,Y,'Method','C','OOBPred','On',...
			    'MinLeafSize',leaf(i));
    plot(oobError(forest),col(i))
    hold on
end
xlabel 'Number of Decision Trees'
ylabel 'Out-of-Bag Classification Error'
legend({'1', '5' '10' '20' '50' '50'},'Location','NorthEast')
hold off

figure %%This plots out of the bag classification error for given number of trees
plot(oobError(forest))
xlabel('Number of Decision Trees')
ylabel('Out-of-Bag Classification Error')