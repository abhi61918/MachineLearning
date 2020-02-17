import pandas as pd 

X_full = pd.read_csv('train.csv', index_col='Id')
X_test_full = pd.read_csv('test.csv', index_col='Id')
print("\nRead Data:\n")
print(X_full)
print(X_test_full)

# Remove rows with missing target ie SalePrice
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)  # inplace when true stores result in same variable ie X_full
print('\nRemoved rows with Sale Price Missing:\n')
print(X_full)

# Set target
y = X_full.SalePrice

# remove target from data set
X_full.drop(['SalePrice'], axis=1, inplace=True)
print('\nTaking this data for prediction (minus SalePrice) :\n')
print(X_full)

# only consider numerical predictors remove others
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])
print('\nRemoved columns with non numerical data: \n')
print(X)

# Divide training and validation data
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
print('\nFinal Training Data: \n')
print(train_X.head)

# Shape of training data (num_rows, num_columns)
print(train_X.shape)

missing_values_by_column = train_X.isnull().sum()
print(missing_values_by_column[missing_values_by_column > 0])

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


def score_dataset(X_train, X_val, y_train, y_val):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    return mean_absolute_error(y_val, pred)


# Approach 1

# this step, you'll preprocess the data in X_train and X_valid to remove columns with missing values.
#  Set the preprocessed DataFrames to reduced_X_train and reduced_X_valid, respectively.
columns_with_missing_values = [cols for cols in train_X
                               if train_X[cols].isnull().any()]
print(columns_with_missing_values)
reduced_X_train = train_X.drop(columns_with_missing_values, axis=1)
reduced_X_valid = val_X.drop(columns_with_missing_values, axis=1)
# Get MAE for this data set
print("MAE (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, train_y, val_y))

# Approach 2

# Try another approach by replacing missing values with mean values of their respective columns
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(train_X))
imputed_X_val = pd.DataFrame(my_imputer.transform(val_X))
# imputation removed column names; put them back
imputed_X_train.columns = train_X.columns
imputed_X_val.columns = train_X.columns
# Get MAE for this imputed data set
print("MAE (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_val, train_y, val_y))

# MAE got worse. Considering colums to be imputed a better strategy would be to replace with median values
final_imputer = SimpleImputer(strategy='median')
final_imputed_X_train = pd.DataFrame(final_imputer.fit_transform(train_X))
final_imputed_X_val = pd.DataFrame(final_imputer.transform(val_X))
# imputation removed column names; put them back
final_imputed_X_train.columns = train_X.columns
final_imputed_X_val.columns = train_X.columns
# Get MAE for this imputed data set
print("MAE (Imputation With Median):")
print(score_dataset(final_imputed_X_train, final_imputed_X_val, train_y, val_y))

# Genrate MAE for test data using model
model = RandomForestRegressor(random_state=0)
model.fit(imputed_X_train, train_y)

final_X_test = pd.DataFrame(final_imputer.fit_transform(X_test))
preds_final = model.predict(final_X_test)
preds_final.shape
# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_final})
output.to_csv('submission.csv', index=False)

# Approach 3
train_X_plus = train_X.copy()
val_X_plus = val_X.copy()
X_test_plus = X_test.copy()

for cols in columns_with_missing_values:
    train_X_plus[cols + 'was_missing'] = train_X[cols].isnull()
    val_X_plus[cols + 'was_missing'] = val_X[cols].isnull()
    X_test_plus[cols + 'was_missing'] = X_test[cols].isnull()

my_imputer2 = SimpleImputer()
imputed_train_X_plus = pd.DataFrame(my_imputer2.fit_transform(train_X_plus))
imputed_val_X_plus = pd.DataFrame(my_imputer2.fit_transform(val_X_plus))
imputed_X_test_plus = pd.DataFrame(my_imputer2.fit_transform(X_test_plus))

imputed_train_X_plus.columns = train_X_plus.columns
imputed_val_X_plus.columns = val_X_plus.columns
imputed_X_test_plus.columns = X_test_plus.columns

print('MAE with extended Imputation Model:')
print(score_dataset(imputed_train_X_plus, imputed_val_X_plus, train_y, val_y))

# Genrate MAE for test data using model
model = RandomForestRegressor(random_state=0)
model.fit(imputed_train_X_plus, train_y)
# impute test data
preds_final = model.predict(imputed_X_test_plus)
preds_final.shape
# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_final})
output.to_csv('submission2.csv', index=False)
