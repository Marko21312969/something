import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import numpy as np

df = pd.read_csv(r'C:\Users\Mark0\OneDrive\Рабочий стол\Boston.csv')
#уберём первую колонку, так как она просто нумерует строки
df = df.drop('Unnamed: 0', axis=1)
X = df.drop("medv", axis=1).values
y = df["medv"].values

#посплитим наши данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Используем линейную регрессию
reg = LinearRegression()

reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print("Predictions: {}, Actual Values: {}".format(y_pred[:10], y_test[:10]))

#как мы видим, линейная регрессия в нашем случае не очень точна, посчитаем насколько
r_squared = reg.score(X_test, y_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)

print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))
#да, 0,7 это не очень...
#посмотрим на кросс валидация нашей линейной регресси
kf = KFold(n_splits=10, shuffle=True, random_state=5)

reg = LinearRegression()

cv_results = cross_val_score(reg, X, y, cv=kf)

print(cv_results)

print(np.mean(cv_results))

print(np.std(cv_results))

print(np.quantile(cv_results, [0.025, 0.975]))
#как мы видим результат не сильно зависил от нашего выбора тестовой выборки
#посмотрим, что сильнее всего влияет на цены при помощи Lasso
lasso = Lasso(alpha=0.3)

lasso = lasso.fit(X, y)

columns = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat',]

lasso_coef = lasso.coef_
#посмотрим на графике результат использования lasso
plt.bar(columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()
#Как мы видим rm (среднее количество комнат на жилище) самая важная переменная и в основном от неё зависят цены
#посмотрим при помощи линейной регрессии, как зависит цена от количества комнат
X = df["rm"].values.reshape(-1, 1)
y = df["medv"].values

reg = LinearRegression()
reg.fit(X, y)
predictions = reg.predict(X)

plt.scatter(X, y, color="blue")
plt.plot(X, predictions, color="red")
plt.xlabel("average number of rooms per dwelling")
plt.ylabel("median value of owner-occupied homes in ($)")
plt.show()
#как мы видим на графике направление мы угадали верно
