import catboost as cb
import numpy as np
import pandas as pd
import pickle
from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler, PolynomialFeatures

# load it up
df = pd.read_csv("abalone.csv")
df.columns = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight',
       'viscera_weight', 'shell_weight', 'rings']
y = df['rings']
X = df.drop('rings', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=100)

# test case
test_case = pd.DataFrame({
    'sex': ['M'],
    'length': [2],
    'diameter': [2],
    'height': [2],
    'whole_weight': [2],
    'shucked_weight': [2],
    'viscera_weight': [2],
    'shell_weight': [2]
})

# naive bench:
y_pred = [y_train.mean()] * len(y_train)
y_pred = pd.DataFrame(data = y_pred,
                      index = y_train.index.values)
print(f'RMSE Naive: {np.sqrt(mean_squared_error(y_train,y_pred))}')

# mapper
mapper = DataFrameMapper([
    ('sex', LabelBinarizer()),
    (['length'], StandardScaler()),
    (['diameter'], StandardScaler()),
    (['height'], StandardScaler()),
    (['whole_weight'], StandardScaler()) ,
    (['shucked_weight'], StandardScaler()),
    (['viscera_weight'], StandardScaler()),
    (['shell_weight'], StandardScaler())
], df_out=True)

Z_train = mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)

# poly features
pnf = PolynomialFeatures(include_bias=False,interaction_only=True)

B_train = pnf.fit_transform(Z_train)
pnfcol = pnf.get_feature_names(input_features=Z_train.columns)
Z_train = pd.DataFrame(B_train, columns=pnfcol).set_index(Z_train.index)

B_test = pnf.fit_transform(Z_test)
pnfcol = pnf.get_feature_names(input_features=Z_test.columns)
Z_test = pd.DataFrame(B_test, columns=pnfcol).set_index(Z_test.index)

# linear regression
model = LinearRegression()
model.fit(Z_train, y_train)
print(f'Score Train: {model.score(Z_train, y_train)}')
print(f'Score Test: {model.score(Z_test, y_test)}')
print(f'RMSE Train: {np.sqrt(mean_squared_error(y_train,model.predict(Z_train)))}')
print(f'RMSE Test: {np.sqrt(mean_squared_error(y_test,model.predict(Z_test)))}')

# ElasticNet
model_e = ElasticNet()
params = {
#'alpha':[10,1,.1,.01,.001,.001,.0001], 
'alpha':[.001], 
'l1_ratio':[1], 
'max_iter': [10000]
}

grid = GridSearchCV(model_e, param_grid=params, cv=10)
grid.fit(Z_train, y_train)
print(grid.best_params_)
model = grid.best_estimator_
print(f'Score Train: {model.score(Z_train, y_train)}')
print(f'Score Test: {model.score(Z_test, y_test)}')
print(f'RMSE Train: {np.sqrt(mean_squared_error(y_train,model.predict(Z_train)))}')
print(f'RMSE Test: {np.sqrt(mean_squared_error(y_test,model.predict(Z_test)))}')

# catboost
A_train = Z_train.values.astype(int).astype(object)
A_test = Z_test.values.astype(int).astype(object)
cat_features = list(range(0, Z_train.shape[1]))
model = cb.CatBoostRegressor(
    iterations=150,
    learning_rate=.1,
)

train_pool = cb.Pool(data=A_train, 
                  label=y_train, 
                  cat_features=cat_features)

test_pool = cb.Pool(data=A_test,
                       label=y_test,
                       cat_features=cat_features)
model.fit(
    train_pool,
    eval_set=test_pool,
    verbose=False,
    plot=True
)
print(f'Score Train: {model.score(A_train, y_train)}')
print(f'Score Test: {model.score(A_test, y_test)}')
print(f'RMSE Train: {np.sqrt(mean_squared_error(y_train,model.predict(A_train)))}')
print(f'RMSE Test: {np.sqrt(mean_squared_error(y_test,model.predict(A_test)))}')

# Random Forest
model = RandomForestRegressor(n_estimators=100)
model.fit(Z_train, y_train)
print(f'Score Train: {model.score(Z_train, y_train)}')
print(f'Score Test: {model.score(Z_test, y_test)}')
print(f'RMSE Train: {np.sqrt(mean_squared_error(y_train,model.predict(Z_train)))}')
print(f'RMSE Test: {np.sqrt(mean_squared_error(y_test,model.predict(Z_test)))}')

# pipe it out
pipe = Pipeline([("mapper", mapper), ("pnf", pnf), ("model", model)])
pickle.dump(pipe, open("pipe.pkl", "wb"))
