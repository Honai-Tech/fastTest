import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import gradio as gr
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

url = 'https://raw.githubusercontent.com/Honai-Tech/fastTest/main/adverse-effects.csv'
df = pd.read_csv(url)

# df.head()

food_name = df['PRI_Reported Brand/Product Name']
idustry_name = df['PRI_FDA Industry Name']
age = df['CI_Age at Adverse Event']
gender = df['CI_Gender']
df.shape
df['PRI_FDA Industry Code'] = df['PRI_FDA Industry Code'].map({'Not Available': '00'}).fillna(df['PRI_FDA Industry Code'])
df.iloc[:,5].unique()

df = df[~df['CI_Age Unit'].isin(['Month(s)', 'Not Available', 'Week(s)', 'Day(s)', 'Decade(s)'])]

df = df.fillna(0)
df.isnull().sum()
df['CI_Age Unit'].unique()
df1 = df.iloc[:, 4:12]
df1.head()

food_name = df['PRI_Reported Brand/Product Name']
idustry_name = df['PRI_FDA Industry Name']
age = df['CI_Age at Adverse Event']
gender = df['CI_Gender']

df2 = pd.concat([food_name,idustry_name,age,gender],axis=1)
df2.head()

sym_exp = df["SYM_One Row Coded Symptoms"].str.split(pat=",", expand=True)
sym_exp = sym_exp.astype('string')

sym_exp.head(5)

sym_exp = sym_exp.fillna('None')
sym_enc = pd.DataFrame(columns=sym_exp.columns, data=LabelEncoder().fit_transform(sym_exp.values.flatten()).reshape(sym_exp.shape))
sym_enc = sym_enc.replace(4310, 0)
sym_enc.head()
sym = sym_enc.dropna()
sym.shape
df2['Gender_Encoded'] = LabelEncoder().fit_transform(df2['CI_Gender'])
df2['FoodName_Encoded'] = LabelEncoder().fit_transform(df2['PRI_Reported Brand/Product Name'])
df2['IndustryName_Encoded'] = LabelEncoder().fit_transform(df2['PRI_FDA Industry Name'])
df2.shape
X = pd.merge(df2, sym, left_index=True, right_index=True)
# master_df.rename({0:'Symptom', 'PRI_Reported Brand/Product Name':'FoodName', 'PRI_FDA Industry Name':'IndustryName', 'CI_Age at Adverse Event':'Age', 'CI_Gender':'Gender'}, axis=1, inplace=True)
X.drop(columns=[ "CI_Gender","PRI_Reported Brand/Product Name", "FoodName_Encoded","PRI_FDA Industry Name","CI_Age at Adverse Event","IndustryName_Encoded","Gender_Encoded"], axis=1, inplace=True)
# df2.drop(columns=[ "CI_Gender","PRI_Reported Brand/Product Name","PRI_FDA Industry Name"], axis=1, inplace=True)
df2.drop(columns=[ "CI_Gender","PRI_Reported Brand/Product Name"], axis=1, inplace=True)

x_col = X.columns
master_df = pd.merge(X, df2, left_index=True, right_index=True)
XX = master_df.drop(columns=["FoodName_Encoded","PRI_FDA Industry Name",'IndustryName_Encoded'], axis=1)
# yy = master_df['FoodName_Encoded']
yy = master_df['IndustryName_Encoded']



X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size = 0.1)
classifier = RandomForestClassifier()
# classifier = DecisionTreeClassifier()

classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred
y_pred.shape

foo = LabelEncoder().fit(df2['PRI_FDA Industry Name'])

prediction_decoded = foo.inverse_transform(y_pred)
print(prediction_decoded)


print("Accuracy of Model::",accuracy_score(y_test,y_pred))

###############################Gradio###############################
s = sym_exp.iloc[:,0].dropna()
s.columns = ['Symptom']
# choices = list(s.head(50))

le = preprocessing.LabelEncoder()

sym_exp = sym_exp.fillna('None')
sym_single = pd.unique(sym_exp.values.ravel('K'))
sym_single_data = pd.DataFrame(sym_single)

sym_single_enc = LabelEncoder().fit_transform(sym_single_data)

le.fit(['Male','Female'])

# le2 = LabelEncoder().fit(a)
le2 = LabelEncoder().fit(sym_single_data)
sym_single.sort()
choices = list(sym_single)


def food(age,gender,symptoms,symptoms2,symptoms3):

  # di = pd.DataFrame(np.zeros((1, 55)))
  # di[0] = 4677
  # di[53] = age
  # di[54] = 0
  
  # do = classifier.predict(di)
  # do_decoded = foo.inverse_transform(do)  
  # food = do_decoded



  # INPUT
  # gender = ['Male']
  # symps = ['BLOOD PRESSURE INCREASED','DEATH']

  g_enc = le.transform([gender])
  s_enc = le2.transform([symptoms])
  s_enc2 = le2.transform([symptoms2]) 
  s_enc3 = le2.transform([symptoms3]) 

    
  di = pd.DataFrame(np.zeros((1, 55)))


  di[0] = s_enc
  di[1] = s_enc2
  di[2] = s_enc3
  di[53] = age
  di[54] = g_enc
    
  do = classifier.predict(di)
  do_decoded = foo.inverse_transform(do)  
  
  do2 = classifier.predict(di)
  do_decoded2 = foo.inverse_transform(do2) 

  
  return do_decoded


age = gr.inputs.Textbox("text")
gender = gr.inputs.Dropdown(['Male','Female'])
symptoms = gr.inputs.Dropdown(choices)
symptoms2 = gr.inputs.Dropdown(choices)
symptoms3 = gr.inputs.Dropdown(choices)
out = gr.outputs.Textbox(label='Food to be avoided')

# out2 = gr.outputs.Textbox(label='Food to be avoided 2')


gr.Interface(food, inputs=[age, gender, symptoms, symptoms2, symptoms3], outputs = [out]).launch(debug=False)