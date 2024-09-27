import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('my_flask_app/irrigation_ds_1.csv')
df_new = df.drop(['CropType','Irrigation'],axis=1)

X = df_new
y = df['Irrigation']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=43)

rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
pred = rfc.predict(X_test)

joblib.dump(rfc, 'irrigation_model.pkl')  # Replace 'your_model_file.pkl' with your desired filename
