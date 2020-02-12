import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('train.csv', index_col='Id')
X_test_full = pd.read_csv('test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)

# Select categorical data with less cardinality
categorical_cols = [cols for cols in X_train_full.columns
                    if X_train_full[cols].dtype == 'object' and
                    X_train_full[cols].nunique() < 10]

numerical_cols = [cols for cols in X_train_full.columns
                  if X_train_full[cols].dtype in ['int64', 'float64']]

# Remove columns other than these 2
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# Pipelining basically combines preprocessing and modelling in a single step so the code is less cluttered
# The code below:
# imputes missing values in numerical data, and
# imputes missing values and applies a one-hot encoding to categorical data.

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical data ie imputation
numerical_transformer = SimpleImputer(strategy='median')

# Preprocessing categorical data ie imputation + OneHotEncoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

# Bundle processing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Define model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Both preprocessing and model is now defined.
# Bundle into one pipeline
my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Use pipeline to preprocess and model
my_pipeline.fit(X_train, y_train)

# Predict
preds = my_pipeline.predict(X_valid)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_valid, preds)

print(f'MAE: {mae}')

# Predict test data directly
preds_final = my_pipeline.predict(X_test)


# Cross validation on model
my_pipeline2 = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators=50, random_state=0))
])
from sklearn.model_selection import cross_val_score
# Combine training and validation data sets as cross validation will do that by itself
X = X_full[numerical_cols]
scores = -1 * cross_val_score(my_pipeline2, X, y, cv=5, scoring='neg_mean_absolute_error')
print(scores.mean())
