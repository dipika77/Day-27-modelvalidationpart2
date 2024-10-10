#Cross Validation

#The problems with holdout phase
#instructions
'''Create samples sample1 and sample2 with 200 observations that could act as possible testing datasets.
Use the list comprehension statement to find out how many observations these samples have in common.
Use the Series.value_counts() method to print the values in both samples for column Class.'''

# Create two different samples of 200 observations 
sample1 = tic_tac_toe.sample(200, random_state=1111)
sample2 = tic_tac_toe.sample(200, random_state=1171)

# Print the number of common observations 
print(len([index for index in sample1.index if index in sample2.index]))

# Print the number of observations in the Class column for both samples 
print(sample1['Class'].value_counts())
print(sample2['Class'].value_counts())


#Cross Validation
#instructions
'''Call the KFold() method to split data using five splits, shuffling, and a random state of 1111.
Use the split() method of KFold on X.
Print the number of indices in both the train and validation indices lists.'''

from sklearn.model_selection import KFold

# Use KFold
kf = KFold(n_splits=5, shuffle=True, random_state=1111)

# Create splits
splits = kf.split(X)

# Print the number of indices
for train_index, val_index in splits:
    print("Number of training indices: %s" % len(train_index))
    print("Number of validation indices: %s" % len(val_index))
    
    
#instructions
'''Use train_index and val_index to call the correct indices of X and y when creating training and validation data.
Fit rfc using the training dataset
Use rfc to create predictions for validation dataset and print the validation accuracy'''

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

rfc = RandomForestRegressor(n_estimators=25, random_state=1111)

# Access the training and validation indices of splits
for train_index, val_index in splits:
    # Setup the training and validation data
    X_train, y_train = X[train_index], y[train_index]
    X_val, y_val = X[val_index], y[val_index]
    # Fit the random forest model
    rfc.fit(X_train, y_train)
    # Make predictions, and print the accuracy
    predictions = rfc.predict(X_val)
    print("Split accuracy: " + str(mean_squared_error(y_val, predictions)))
    
    
    
#Sklearn's cross-val score
#instructions
'''Load the method for calculating the scores of cross-validation.
Load the random forest regression method.
Load the mean square error metric.
Load the method for creating a scorer to use with cross-validation.'''

# Instruction 1: Load the cross-validation method
from sklearn.model_selection import cross_val_score

# Instruction 2: Load the random forest regression model
from sklearn.ensemble import RandomForestRegressor

# Instruction 3: Load the mean squared error method
# Instruction 4: Load the function for creating a scorer
from sklearn.metrics import mean_squared_error, make_scorer


#instructions
'''Fill in cross_val_score().
Use X_train for the training data, and y_train for the response.
Use rfc as the model, 10-fold cross-validation, and mse for the scoring function.
Print the mean of the cv results.'''

rfc = RandomForestRegressor(n_estimators=25, random_state=1111)
mse = make_scorer(mean_squared_error)

# Set up cross_val_score
cv = cross_val_score(estimator= rfc,
                     X= X_train,
                     y=y_train,
                     cv= 10,
                     scoring=mse)

# Print the mean error
print(cv.mean())


#Leave-one-out-cross-validation (LOOCV)
#instructions
'''Create a scorer using mean_absolute_error for cross_val_score() to use.
Fill out cross_val_score() so that the model rfr, the newly defined mae_scorer, and LOOCV are used.
Print the mean and the standard deviation of scores using numpy (loaded as np).'''

from sklearn.metrics import mean_absolute_error, make_scorer

# Create scorer
mae_scorer = make_scorer(mean_absolute_error)

rfr = RandomForestRegressor(n_estimators=15, random_state=1111)

# Implement LOOCV
scores = cross_val_score(rfr, X=X, y=y, cv=85, scoring=mae_scorer)

# Print the mean and standard deviation
print("The mean of the errors is: %s." % np.mean(scores))
print("The standard deviation of the errors is: %s." % np.std(scores))



#Introduction to hyperparameter tuning
#instructions
'''Print.get_params() in the console to review the possible parameters of the model that you can tune.
Create a maximum depth list, [4, 8, 12] and a minimum samples list [2, 5, 10] that specify possible values for each hyperparameter.
Create one final list to use for the maximum features.
Use values 4 through the maximum number of features possible (10), by 2.'''

# Review the parameters of rfr
print(rfr.get_params())

# Maximum Depth
max_depth = [4, 8, 12]

# Minimum samples for a split
min_samples_split = [2, 5, 10]

# Max features 
max_features = [4,6,8,10]


# instructions
'''Randomly select a max_depth, min_samples_split, and max_features using your range variables.
Print out all of the parameters for rfr to see which values were randomly selected.'''

from sklearn.ensemble import RandomForestRegressor

# Fill in rfr using your variables
rfr = RandomForestRegressor(
    n_estimators=100,
    max_depth=random.choice(max_depth),
    min_samples_split=random.choice(min_samples_split),
    max_features=random.choice(max_features))

# Print out the parameters
print(rfr.get_params())


#Randomized Search CV
#instructions
'''Finalize the parameter dictionary by adding a list for the max_depth parameter with options 2, 4, 6, and 8.
Create a random forest regression model with ten trees and a random_state of 1111.
Create a mean squared error scorer to use.'''

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error

# Finish the dictionary by adding the max_depth parameter
param_dist = {"max_depth": [2,4,6,8],
              "max_features": [2, 4, 6, 8, 10],
              "min_samples_split": [2, 4, 8, 16]}

# Create a random forest regression model
rfr = RandomForestRegressor(n_estimators=10, random_state=1111)

# Create a scorer to use (use the mean squared error)
scorer = make_scorer(mean_squared_error)


#instructions
'''Load the method for conducting a random search in sklearn.
Complete a random search by filling in the parameters: estimator, param_distributions, and scoring.
Use 5-fold cross validation for this random search.'''

# Import the method for random search
from sklearn.model_selection import RandomizedSearchCV

# Build a random search using param_dist, rfr, and scorer
random_search =\
   RandomizedSearchCV(
        estimator=rfr,
        param_distributions=param_dist,
        n_iter=10,
        cv=5,
        scoring=scorer)
   
   
#Selecting your final model
#instructions
'''Create a precision scorer, precision using make_scorer(<scoring_function>).
Complete the random search method by using rfc and param_dist.
Use rs.cv_results_ to print the mean test scores.
Print the best overall score.'''

from sklearn.metrics import precision_score, make_scorer

# Create a precision scorer
precision = make_scorer(precision_score)
# Finalize the random search
rs = RandomizedSearchCV(
  estimator=rfc, param_distributions=param_dist,
  scoring = precision,
  cv=5, n_iter=10, random_state=1111)
rs.fit(X, y)

# print the mean test scores:
print('The accuracy for each run was: {}.'.format(rs.cv_results_['mean_test_score']))
# print the best model score:
print('The best accuracy for a single model was: {}'.format(rs.best_score_))