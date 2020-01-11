import pandas as pd

print("\n#******Iowa Data******#\n")
iowa_file = 'train.csv'
iowa_data = pd.read_csv(iowa_file)
print(iowa_data.columns)
# view data
iowa_data.describe()
print(iowa_data.describe())
print(iowa_data.head)

# Specify Prediction Target
y = iowa_data.SalePrice

# Specify features
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = iowa_data[features]

print(X.describe())
print(X.head)

# Specify and Fit Model
from sklearn.tree import DecisionTreeRegressor

iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(X, y)

# predict from model
prediction = iowa_model.predict(X)
print(prediction)
print(y)

# Check accuracy of model
from sklearn.metrics import mean_absolute_error

err = mean_absolute_error(y, prediction)
print(f'Mean error: {err}')
# Error is coming less as same data used in training is used for validating also
# Need to separate training data and validation data

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
iowa_model.fit(train_X, train_y)
prediction_new = iowa_model.predict(val_X)
err_new = mean_absolute_error(val_y, prediction_new)
print(f'\nNew Mean Error with different training and validation sets: {err_new}\n')


# New Error is much more this time as training and validating data are different


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    predictions = model.predict(val_X)
    mae = mean_absolute_error(val_y, predictions)
    return mae


candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
i1 = 0
mae_new = [0] * len(candidate_max_leaf_nodes)
for i in candidate_max_leaf_nodes:
    mae_new[i1] = get_mae(i, train_X, val_X, train_y, val_y)
    print(f'MAE with {i} leaf nodes: {mae_new[i1]}')
    i1 += 1

mae_min = min(mae_new)
i_min = mae_new.index(mae_min)
optimum_leaf_nodes = candidate_max_leaf_nodes[i_min]
print(f'\nMinimum MAE is {mae_min} for {optimum_leaf_nodes} leaf nodes')


# Optimize entire model including validation data with optimum leaf nodes

model_final = DecisionTreeRegressor(max_leaf_nodes=optimum_leaf_nodes, random_state=1)
model_final.fit(X, y)
prediction_final = model_final.predict(X)
mae_final = mean_absolute_error(y, prediction_final)
print(f'\nFinal MAE with optimized leaf nodes({optimum_leaf_nodes}) : {mae_final}')

# Random forest use multiple decision trees instead of one to give better results
from sklearn.ensemble import RandomForestRegressor
model_rf = RandomForestRegressor(random_state=1)
model_rf.fit(train_X, train_y)
prediction_rf = model_rf.predict(val_X)
mae_rf = mean_absolute_error(val_y, prediction_rf)
print(f'\nMAE using Random Forests: {mae_rf}')

model_rf_final = RandomForestRegressor(random_state=1)
model_rf.fit(X, y)
prediction_rf = model_rf.predict(X)
mae_rf = mean_absolute_error(y, prediction_rf)
print(f'\nFinal MAE using Random Forests: {mae_rf}')