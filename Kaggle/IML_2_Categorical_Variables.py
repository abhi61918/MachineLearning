import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X = pd.read_csv('train.csv', index_col='Id')
X_test = pd.read_csv('test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll drop columns with missing values ie no imputation for this exercise
cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
X.drop(cols_with_missing, axis=1, inplace=True)
X_test.drop(cols_with_missing, axis=1, inplace=True)

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


# APPROACH 1
# Drop columns with categorical data
drop_X_train = X_train.select_dtypes(exclude='object')
drop_X_valid = X_valid.select_dtypes(exclude='object')

print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))

# APPROACH 2
# Encoding categorical data (non-numerical data)
print(X_train.head())
print(drop_X_train.head())
# From print output can see total 27 columns dropped
categorical_columns = set(X_train.columns) - set(drop_X_train.columns)
print(f'\nNumber of columns with categorical data: {len(categorical_columns)}')
print(list(categorical_columns))

print('\nLet\'s see unique values in \'Condition2\' column in training and validation data respectively:')
print(X_train['Condition2'].unique())
print(X_valid['Condition2'].unique())
print('\nValidation set has values that don\'t appear in training data and hence scikit encoder will throw error for '
      'these values. Let\'s take a simple approach and drop all such columns which exhibit such difference in '
      'training and validation data')

# Another way to get categorical columns
categorical_columns = [cols for cols in X_train.columns
                       if X_train[cols].dtype == 'object']
print(f'\nCategorical columns: {categorical_columns}')

# Find columns that can be safely encoded that is which have same set of data in training and validation
good_label_cols = [cols for cols in categorical_columns
                   if set(X_train[cols].unique()) == set(X_valid[cols].unique())]
print(f'\nGood Label Columns that dont have different unique values in training and validation data: {good_label_cols}')
print(f'No. of good label columns: {len(good_label_cols)}')
bad_label_cols = set(categorical_columns)-set(good_label_cols)
print(f'\nTherefore discarded columns that will be dropped: {bad_label_cols}')

# Let's drop bad label cols from both training and Validation data
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

# Let's use a label encoder
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

# Do label encoding of each column with categorical data
for cols in good_label_cols:
    label_X_train[cols] = label_encoder.fit_transform(label_X_train[cols])
    label_X_valid[cols] = label_encoder.fit_transform(label_X_valid[cols])

print(f'\nLabel Encoded Data:\n {label_X_train}')

print("MAE from Approach 2 (Label Encoding):")
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))

# APPROACH 3
# One hot encoding
# Make a column for each unique value in a categorical column and assign 1 or 0 to it depending on what value it is
# For this approach to work the no of unique values should be less, else the no. of columns added will be huge
# Hence lets explore the cardinality of each column ie the number of unique values in each column
for cols in good_label_cols:
    print(f'{cols}: {(X_train[cols].nunique())}')

# Lets take cols with cardinality less than 10
low_cardinality_cols = [cols for cols in good_label_cols
                        if X_train[cols].nunique() < 10]

# We are using good_label_cols to avoid the discussed scenario earlier but One Hot Encoder has option to ignore
# validation categorical data that is not there in training and hence we can use all object_cols instead of just
# good_label_cols
low_cardinality_cols = [cols for cols in categorical_columns
                        if X_train[cols].nunique() < 10]
print(low_cardinality_cols)

from sklearn.preprocessing import OneHotEncoder
# handle_unknown setting to ignore, ignores if categories other than testing data are there in validation data
# sparse if true returns a sparse matrix instead of numpy. We need numpy
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
# Now lets encode the columns we need ie low_cardinality_cols
OH_train_cols = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_valid_cols = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))
# OH Encoding removes index. Add back index
OH_train_cols.index = X_train.index
OH_valid_cols.index = X_valid.index

# Now since we have columns representing the categorical columns we can drop them from original data and append OH_cols
num_X_train = X_train.drop(categorical_columns, axis=1)
num_X_valid = X_valid.drop(categorical_columns, axis=1)

OH_X_train = pd.concat([num_X_train, OH_train_cols], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_valid_cols], axis=1)

print("MAE from Approach 3 (One-Hot Encoding):")
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))

# Run model on test data
# X_test_plus = X_test.copy()
# columns_with_missing_values = [cols for cols in X_test
#                                if X_test[cols].isnull().any()]
# for cols in columns_with_missing_values:
#     X_test_plus[cols + 'was_missing'] = X_test[cols].isnull()
# Categorical data in test data has missing values. Need to impute them using most frequent values in those columns
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer(strategy='most_frequent')
final_X_test = pd.DataFrame(my_imputer.fit_transform(X_test))
final_X_test.columns = X_test.columns
# Imputed data is now in final_X_test. Using this for OH encoding
OH_test_cols = pd.DataFrame(OH_encoder.transform(final_X_test[low_cardinality_cols]))
OH_test_cols.index = final_X_test.index
num_X_test = final_X_test.drop(categorical_columns, axis=1)
OH_X_test = pd.concat([num_X_test, OH_test_cols], axis=1)

model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(OH_X_train, y_train)

preds_final = model.predict(OH_X_test)
output = pd.DataFrame({'Id': OH_X_test.index,
                       'SalePrice': preds_final})
output.to_csv('submission3.csv', index=False)

