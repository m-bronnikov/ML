import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option('max_columns', 100)

#Считаем данные
sample= pd.read_csv("/kaggle/input/mlcourse-2019-mai-autumn/sample.csv")
train= pd.read_csv("/kaggle/input/mlcourse-2019-mai-autumn/train.csv")
test= pd.read_csv("/kaggle/input/mlcourse-2019-mai-autumn/test.csv")

#Используемые колонки
cols = ['Id', 'MSSubClass', 'OverallQual', 'OverallCond',
'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 
'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',  'GrLivArea',
'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',
'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 
'ScreenPorch', 'MoSold', 'YrSold', '2ndFlrSF',
        'EnclosedPorch', 
        
        'LowQualFinSF','3SsnPorch',
         'LotArea','MiscVal','BsmtUnfSF',
        'PoolArea', 'SalePrice']

# Колонка 'Id' вообще не несет никакой информации, ее сразу отбрасываем, также без поля PoolArea - оно не заполнено в тестовых данных
train_columns = cols[1:-2]
target_column = cols[-1]

# Нормируем фичи
# В метрике задачи используются алгоритмы, пэтому лучше использовать нормализация минмакс

X_scaled = (train[train_columns] - train[train_columns].min(axis=0))/(train[train_columns].max(axis=0) - train[train_columns].min(axis=0))
y = train[target_column]

# Добавим единичный столбец Intercept.
X_scaled['Intercept'] = 1

#Предсказание
X_test_scaled =  ((test[train_columns] - test[train_columns].min(axis=0))/(test[train_columns].max(axis=0) - test[train_columns].min(axis=0)))
X_test_scaled['Intercept'] = 1
X_test_scaled = X_test_scaled.values

# Наш финальынй вектор с фичами для обучения 
X =  X_scaled.values
y = y.values

# Сразу создадим отложенную выборку, на которой будем проверять качество нашего решения

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=14)

# Избавимся от выбросов
x_train = x_train[y_train < 300000]
y_train = y_train[y_train < 300000]

# Функция ошибки в задаче - берется из условия
def rmsle_error(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    terms_to_sum = [(np.log(y_pred[i] + 1) - np.log(y_true[i] + 1)) ** 2.0 for i, pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0 / len(y))) ** 0.5


#Инициализируем вектор весов w - случайный вес для кжадой из нашей фичи
w = np.random.rand(x_train.shape[1])

#сколько раз будем спускаться по градиентному спуску - сколько сделаем шажков
steps = 500

# здесь храним историю ошибок нашей модели
costHistory = []
rmsleHistory = []

# Регуляризатор, который штрафует за большие веса при X
lmbda = 0.01
# Размер шага
alpha = 0.1

for step in range(steps):
    # считаем, что вообще модель предсказывает на конкретном шаге  
    # все вычисления идут уже в матричной форме
    prediction = x_train.dot(w)
    prediction_test = x_test.dot(w)
    
    # считаем ошбику модели
    error = prediction - y_train
    cost = error**2 + lmbda / 2 * w.T.dot(w)
    
    rmsle = rmsle_error(y_train, prediction)
    rmsle_test = rmsle_error(prediction_test, y_test)
    costHistory.append(cost)
    rmsleHistory.append(rmsle)
    
    print("[INFO] шаг {}, RMSLE = {}, RMSLE отложенная выборка {}".format(step + 1, rmsle,rmsle_test))
    
    # наша функция градиента сразу в матричном виде - разберитесь, почему именно так выглядит
    gradient = x_train.T.dot(error) / x_train.shape[0] + lmbda * w
    
    # спускаемся по "склону" градиентного спуска
    w += - alpha * gradient
 
prediction = X_test_scaled.dot(w)

sample['SalePrice'] = prediction
sample.to_csv('simple_solution.csv', index = False)