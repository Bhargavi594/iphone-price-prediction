import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
df = pd.read_csv("iphone_data.csv")
le = LabelEncoder()
df['condition'] = le.fit_transform(df['condition'])
X = df[['storage', 'ram', 'camera', 'battery', 'condition', 'year']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
sample = [[128, 4, 12, 3095, 1, 2021]]
predicted_price = model.predict(sample)
print("Predicted Price:", predicted_price[0])
