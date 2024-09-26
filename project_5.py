import pandas as pd
import numpy as np
from sklearn import linear_model
import warnings  

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
df = pd.read_csv(r'C:\Users\leand\OneDrive\Documentos\FormacaoDSA\f_projeto5\Scripts-05\Projeto\dados\dataset.csv')

# print(df.head())
# print(df.info())


## Applying One-Hot Encoding

# df_dummies = pd.get_dummies(df['Modelo'],dtype=float)
# print(df_dummies)
# df = pd.concat([df, df_dummies], axis=1)
# df_ml = df.drop(columns='Modelo')
# print(df_ml)

# x = df_ml.drop(columns='Preco_Venda')  #x is the input variable
# y = df_ml['Preco_Venda']  #y is the output variable, the target
# print(x)
# print(y)

#traning Model Linear Regression, setting a x, y
# modelo_1 = linear_model.LinearRegression()
# modelo_1.fit(x,y)

# print(modelo_1.predict(x))

#test model giving 5 variables to hem
# print(modelo_1.predict([[7000,4,0,0,1]]))  
# print(modelo_1.predict([[45000,4,0,1,0]]))
# print(modelo_1.predict([[86000,7,1,0,0]]))

# print(modelo_1.score(x,y))  # 94% of precision


## Applying Label Encoding
# from sklearn.preprocessing import LabelEncoder
# copy = df
# copy['Modelo'] = LabelEncoder().fit_transform(copy['Modelo'])
# print(copy)

#separating the columns input and output
# x = copy[['Modelo','Kilometragem', 'Idade_Veiculo']].values
# y = copy['Preco_Venda'].values

#training model_2
# modelo_2 = linear_model.LinearRegression()
# modelo_2.fit(x,y)

# print(modelo_2.predict(x))  # geral vision of model
# print(modelo_2.score(x,y))  # 88% of precision

## One-Hot Enconding showed better result. Now, we will follow improving this model

## One-Hot-Encode with Transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

OHEncoder = ColumnTransformer([('Modelo', OneHotEncoder(), [0])], remainder= 'passthrough')

df_copy = df
x = df_copy.drop(columns='Preco_Venda')  #x is the input variable
y = df_copy['Preco_Venda']  #y is the output variable, the target

X = OHEncoder.fit_transform(x)
# print(X)

modelo_3 = linear_model.LinearRegression()
modelo_3.fit(X,y)

print(modelo_3.predict([[0,0,1,45000, 4]]))
print(modelo_3.predict([[0,1,0,86000, 7]]))
print(modelo_3.score(X,y))   # 94% of precision

#Save the model in machine
import joblib
joblib.dump(modelo_3, 'Projeto/modelin.pkl')

## Conclusion
## For this dataset, the method One-Hot-Encoder was most effective and should be used in data processing to predictive modeling
