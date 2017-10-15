# dsg_predictive_modelling

Solution for the mushroom dataset using matlab machine learning toolbox.

The solution is contained in the file "solution.csv" which contains the predicted values of classes for the given data.

Then the file 'solution_code.m' contains the source code

In this repository I have put the complete solution of the mushroom dataset provided by data science group iitr.
I have used deep learning to solve this classification problem.
The repository contains a file names "solution_code" that contains the complete code to carry out the process of ddep learning by using 
a neural network with single hidden layer.

I have first analyzed the dataset and inferred that tha attributes - 
gill-attachment have 97.64% of the values as 'f',
veil-type have 100% of the values as 'p', and 
veil-colour have 97.73% of the values as 'w' 
which provides the information that these are not informative for our model so I removed them in the beginning.

Then I realized that in some attributes, some particular values are concentrated more at the bottom of the dataset and some at the top, 
so I distributed them randomly in order to break any kind of symmetry.

Then I separated the numerical data types like radius and weight from the categorical data type and had to do preprocessing separately
on both.

I used feature normalization for the numerical data and used label-encoder for the categorical data.

Then I separated the data entries which had missing values of stalk-root and used the entries with stalk-root value to predict the value of
the missing stalk root values.

I divided the training set into two parts -
1.training set 
2.cross-validation set

Then I used the whole data set to train a neural network and predicted the values for the Cross-Validation set.
So I checked the accuracy and tuned the parameters accordingly and reached an accuracy of approx. 95 - 97 %

Then I used the previously trained model to find the missing stalk-root values in the test set.

Then I used the latter model to predict the values of classes in the mushroom_test set.

Then I converted the numerical values to the categories i.e. 'e' and 'p'

For the training purpose I wrote the cost function myself and then used fmincg() an inbuilt function of matlab machine learning toolbox to 
minimize the cost value with repect to the parametric weights i.e. Theta's.
