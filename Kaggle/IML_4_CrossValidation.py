import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
train_data = pd.read_csv('train.csv', index_col='Id')
test_data = pd.read_csv('test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice
train_data.drop(['SalePrice'], axis=1, inplace=True)

# Select numeric columns only
numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
X = train_data[numeric_cols].copy()
X_test = test_data[numeric_cols].copy()

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators=50, random_state=0))
])

from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("Average MAE score:", scores.mean())


# Define a function for all the above tasks
def get_score(n_estimators):
    """Return the average MAE over 3 CV folds of random forest model.

    Keyword argument:
    n_estimators -- the number of trees in the forest
    """
    my_p = Pipeline(
        steps=[
            ('prep', SimpleImputer()),
            ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))
        ])
    scores = -1 * cross_val_score(my_p, X, y,
                                  cv=3,
                                  scoring='neg_mean_absolute_error')
    return scores.mean()


results = {}
for n in range(1, 9):
    results[n * 50] = (get_score(n * 50))

print(results)

# Lets plot all the values for different n_estimators
import matplotlib.pyplot as plt

plt.plot(list(results.keys()), list(results.values()))

plt.show()


# Lets add more parameters to def_score function
def get_score2(n_estimators, cv, X, y):
    """Return the average MAE over 3 CV folds of random forest model.

    Keyword argument:
    n_estimators -- the number of trees in the forest
    """
    my_p = Pipeline(
        steps=[
            ('prep', SimpleImputer()),
            ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))
        ])
    scores = -1 * cross_val_score(my_p, X, y,
                                  cv=cv,
                                  scoring='neg_mean_absolute_error')
    return scores.mean()


import numpy as np
print('Warning: Wait for 15 mins for below code to execute')
results2 = np.zeros((8, 4))
for n in range(1, 9):
    for m in range(1, 5):
        results2[n - 1][m - 1] = (get_score2(n * 50, m + 4, X, y))

print(results2)
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
xline = [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4 + [5] * 4 + [6] * 4 + [7] * 4 + [8] * 4
yline = [1, 2, 3, 4] * 8
zline = results2.reshape(32)
# ax.plot3D(xline, yline, zline, 'gray')
ax.scatter3D(xline, yline, zline, c=zline, cmap='Accent_r')
