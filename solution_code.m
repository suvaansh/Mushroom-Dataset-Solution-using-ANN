clc;
close all;
clear all;
% Using the given data and the classes I have firstly predicted the missing
% values in the stalk-root data type
%For this I have firstly separated the categorical data and numerical (including the labeled class)
%Then I separate the datas with missing root-stalk values from the ones having it.

% In the whole code :
% Numerical values means he values of weight and radius attribute
% Categorical values means the rest of attributes

load('train_data');
mod_cat_data = categoricaldata;  %mod_cat_data => modified categorical data

%Removing the attributes that are not of much use because of repetition of
%data
mod_cat_data(:,1) = []; %This is the labeled class itself
mod_cat_data(:,6) = []; %gill-attachment have 97.64% of the values as f
mod_cat_data(:,15) = []; %veil-type have 100% of the values as p
mod_cat_data(:,15) = []; %veil-colour have 97.73% of the values as w

%Converting alphabetical data to numerical form
proc_cat_data = grp2idx(mod_cat_data(:,1));   %proc_cat_data => processed categorical data
for i = 2:size(mod_cat_data,2),
proc_cat_data = [proc_cat_data grp2idx(mod_cat_data(:,i))];
end;


%Finding the rows with missing data and separating from the other data
index = find(proc_cat_data(:,10) == 1);
missing_cat_data = proc_cat_data(index,:);  %missing_cat_data => data entry having missing values of stalk root 
non_missing_cat_data = proc_cat_data;  % opposite of missing
non_missing_cat_data(index,:) = [];

missing_num_data = numericdata(index,:);  %missing_num_data => data entry having missing values of stalk root
non_missing_num_data = numericdata;
non_missing_num_data(index,:) = [];

norm_non_missing_num_data = featureNormalize(non_missing_num_data); %norm_non_missing_num_data => feature normalized non missing numerical data
norm_missing_num_data = featureNormalize(missing_num_data);  %norm_missing_data => feature normalized numerical data 


%X1 => data entries with the values of stalk roots
%Y1 => stalk root values of the same data
X1 = [non_missing_cat_data(:,1:9) non_missing_cat_data(:,11:end) norm_non_missing_num_data]; 
Y1 = non_missing_cat_data(:,10);
Y1 = Y1-1;

%Separating data to be trained into two parts - training set and Cross-validation set
X1train = X1(1:floor(size(X1,1)*60/100),:); 
X1CV = X1(size(X1train,1)+1:end,:);
Y1train = Y1(1:floor(size(Y1,1)*60/100),:); 
Y1CV = Y1(size(Y1train,1)+1:end,:);

%X1predict => data with missing stalk root values that is to be predicted
X1predict = [missing_cat_data(:,1:9) missing_cat_data(:,11:end) missing_num_data];

% Using these non missing datas to train a model to predict the values of missing stalk root

input_layer_size1  = 20;   % 20 attributes 
hidden_layer_size1 = 40;   % 40 hidden units and a single hidden layer
num_labels1 = 4;          % 4 labels, from 1 to 4   

%initialiazing weights randomly in order to break any kind of symmetry 
initial_Theta1 = randInitializeWeights(input_layer_size1, hidden_layer_size1); 
initial_Theta2 = randInitializeWeights(hidden_layer_size1, num_labels1);

% Unroll parameters to be passed to the cost function
%initial_nn_params => initial neural network parameters
initial_nn_params1 = [initial_Theta1(:) ; initial_Theta2(:)];

                          
options = optimset('MaxIter', 200);

%I have done regularization in order to keep the value of weights low
lambda1 = 1;  %regularisation parameter


%Here first I used feed forwarding to get the predictions and then used
%backpropagation to get the values of weights

%Neural Network Cost Funcion/Loss Function
costFunction = @(p) nnCostFunction(p,input_layer_size1,hidden_layer_size1,num_labels1, X1train, Y1train, lambda1);

%using an inbuilt function of machine learning library to minimize the
%value of cost and returning the weights 
%Here neural network training is done
%nn_params => neural networks parameters
[nn_params1, cost1] = fmincg(costFunction, initial_nn_params1, options);


%Theta1 => weights from input layer to the hidden layer
%Theta2 => weights from hidden layer to the output layer

Theta1 = reshape(nn_params1(1:hidden_layer_size1 * (input_layer_size1 + 1)), ...
                 hidden_layer_size1, (input_layer_size1 + 1));

Theta2 = reshape(nn_params1((1 + (hidden_layer_size1 * (input_layer_size1 + 1))):end), ...
                 num_labels1, (hidden_layer_size1 + 1));
             
%Then I used them to predict the missing values CV set to check the
%accuracy of training and I reached 99.56% by certain parametric tuning
testpred1 = predict(Theta1, Theta2, X1CV);
%testpred1 => predicted values of stalk roots for Cross-Validation
%set for accuracy check

fprintf('\nAccuracy: %f\n', mean(double(testpred1 == Y1CV)) * 100);
% pause;
%Then I used these weights to predict the missing values of stalk root
Y1predicted = predict(Theta1,Theta2,X1predict);
%Y1predicted => predicted values for missing stalk-roots
%Now we have predicted the missing data

% Now we will put this data in the dataset and use all data to train for 
%class and then predict the classes for the training set

%Now the new dataset will become

newdataset = [X1 Y1 ; X1predict Y1predicted];
X2 = newdataset;
%X2 contains the data along with tha values of stalk-roots so as to predict
%the classes for data

%Here we are converting the given classes to the same format as other data
%so that no mismatching takes place
class = categoricaldata(:,1);
proc_class = grp2idx(class); %This is the label-encoder function in matlab 
%proc_class => processed class values
class_mis = proc_class(index,:); %class_mis => classes values for data entries with missing value of stalk roots
class_nonmis = proc_class; %class_nonmis => classes values for data entries with stalk roots values present
class_nonmis(index,:) = [];

Y2 = [class_nonmis ; class_mis]; 


%Now this dataset can be used to train a model and predict a class for the
%actual training set

%Here I am randomly selecting data for training and Cross-validation set to
%break any kind of symmetry in the dataset because I found that in some
%attributes  I found that some kinds of data are concentrated mostly in the
%vottom of dataset and some in the above part
newindex = randperm(size(X2,1));
newindex = newindex';
newindextrain = newindex(1:size(newindex,1)*70/100,1);
newindexCV = newindex((size(newindextrain,1)+1):end,:); 

X2train = X2(newindextrain,:); 
X2CV = X2(newindexCV,:);
Y2train = Y2(newindextrain,:); 
Y2CV = Y2(newindexCV,:);

%Now we will train a neural network using this dataset and check its
%accuracy by using Cross-Validation set
input_layer_size2  = 21;   % 21 attributes 
hidden_layer_size2 = 16;   % 17 hidden units and a single hidden layer
num_labels2 = 4;          % 4 labels, from 1 to 4   

%Randomly initializing weights for the same reason mentioned before
initial_Theta3 = randInitializeWeights(input_layer_size2, hidden_layer_size2);
initial_Theta4 = randInitializeWeights(hidden_layer_size2, num_labels2);

% Unroll parameters
initial_nn_params2 = [initial_Theta3(:) ; initial_Theta4(:)];

                          
options = optimset('MaxIter', 250);
lambda2 = 0;  %regularisation parameter

costFunction = @(p1) nnCostFunction(p1,input_layer_size2,hidden_layer_size2,num_labels2, X2train, Y2train, lambda2);


[nn_params2, cost2] = fmincg(costFunction, initial_nn_params2, options);

%Theta3 => weights from input layer to the hidden layer
%Theta4 => weights from hidden layer to the output layer

Theta3 = reshape(nn_params2(1:hidden_layer_size2 * (input_layer_size2 + 1)), ...
                 hidden_layer_size2, (input_layer_size2 + 1));

Theta4 = reshape(nn_params2((1 + (hidden_layer_size2 * (input_layer_size2 + 1))):end), ...
                 num_labels2, (hidden_layer_size2 + 1));
             
testpred2 = predict(Theta3, Theta4, X2CV);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(testpred2 == Y2CV)) * 100);


%Now we are loading the mushroom_test dataset

load('test_data.mat');


%categoricaltestdata => the categorical data present in the test set
%Then we again do the same preprocessing on the data as done in the earlier
%case

%Now we are removing the unwanted/non-informative attributes to as we have
%done earlier so as to increase the performance of our neural net
categorictestdata(:,6) = []; 
categorictestdata(:,15) = [];
categorictestdata(:,15) = [];

proc_cat_test_data = grp2idx(categorictestdata(:,1)); %proc_cat_test_data => processed categorical data in test set
for i = 2:size(categorictestdata,2), 
proc_cat_test_data = [proc_cat_test_data grp2idx(categorictestdata(:,i))];
end;

testindex = find(proc_cat_test_data(:,10) == 3);

missing_cat_test_data = proc_cat_test_data(testindex,:);%missig_cat_test_data => categorical test data with missing stalk root values
non_missing_cat_test_data = proc_cat_test_data;%non_missig_cat_test_data => categorical test data with having stalk root values
non_missing_cat_test_data(testindex,:) = [];

missing_num_test_data = numerictestdata(testindex,:); %missing_num_test_data => missing stalk root numerical test data
non_missing_num_test_data = numerictestdata; %non_missing_num_test_data => non missing stalk root numerical test data
non_missing_num_test_data(testindex,:) = [];

norm_non_missing_num_test_data = featureNormalize(non_missing_num_test_data); %here norm means feature normalized
norm_missing_num_test_data = featureNormalize(missing_num_test_data);

X3 = [non_missing_cat_test_data(:,1:9) non_missing_cat_test_data(:,11:end) norm_non_missing_num_test_data];
Y3 = non_missing_cat_test_data(:,10);
for i = 1: length(Y3),
    if(Y3(i,1)>3),
        Y3 = Y3-1;
    end;
end;


X3predict = [missing_cat_test_data(:,1:9) missing_cat_test_data(:,11:end) missing_num_test_data];
%X3predict => test set data with missing stalk root values that is to be predicted


%Here we predict the missing values of stalk root using the previously
%trained neural network
Y3predicted = predict(Theta1,Theta2,X3predict);
%Y3predict => predicted missing stalk root values from the test set

newdataset1 = [X3 Y3 ; X3predict Y3predicted];
X4 = newdataset1;
%X4 contains the dataset containing the missing stalk roots values

%Now we are using the previously trained neural net to predict the final
%class

Y4 = predict(Theta3, Theta4, X4);

%Y4 is the final predicted values of classes for the test data set


for i=1:size(Y4,1),
    if(Y4(i,1)==1),
        Y4(i,1)='p';
    else
        Y4(i,1)='e';
    end;
end;
Y4 = char(Y4);


