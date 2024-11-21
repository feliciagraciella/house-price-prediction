from flask import Flask, request, render_template, url_for
import pickle
import numpy
import scipy
import sklearn
# import pandas as pd
from sklearn.preprocessing import StandardScaler

# from sklearn.impute import SimpleImputer

# url = 'https://raw.githubusercontent.com/twiradinata/datasets/main/property_surabaya.csv'
# df = pd.read_csv(url, delimiter=';')

# df = df.dropna(subset=['pricing_category'], how='all')

# imputer = SimpleImputer(strategy='most_frequent')
# df.iloc[:,:] = imputer.fit_transform(df)

# from scipy import stats

# def remove_outlier(df_in, col_name):
#     q1 = df_in[col_name].quantile(0.25)
#     q3 = df_in[col_name].quantile(0.75)
#     iqr = q3-q1 #Interquartile range
#     fence_low  = q1-1.5*iqr
#     fence_high = q3+1.5*iqr
#     df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
#     return df_out

# df = remove_outlier(df, 'price')

# from sklearn.preprocessing import LabelEncoder

# # For nominal scale simply use LabelEncoder
# le = LabelEncoder()
# df['facing'] = le.fit_transform(df['facing'])
# df['house_position'] = le.fit_transform(df['house_position'])
# df['urgent'] = le.fit_transform(df['urgent'])
# df['ready_to_use'] = le.fit_transform(df['ready_to_use'])
# df['furnished'] = le.fit_transform(df['furnished'])

# # For ordinal scale need to use map
# ownership_status_map = {'Surat Hijau':1, 'Pengikatan Perjanjian Jual Beli (PPJB)':2, 'Hak Guna Bangunan (HGB)':3, 'Hak Milik (SHM)':4}
# df['ownership_status'] = df['ownership_status'].map(ownership_status_map)

# road_width_map = {'< 1 Mobil':1, '1-2 Mobil':2, '> 2 Mobil':3}
# df['road_width'] = df['road_width'].map(road_width_map)

# building_age_map = {'1 - 4 Tahun':1, '5 - 10 Tahun':2, '> 10 Tahun':3}
# df['building_age'] = df['building_age'].map(building_age_map)

# area_category_map = {'Standard': 1, 'Premium': 2, 'Sangat Premium': 3}
# df['category'] = df['category'].map(area_category_map)

# pricing_category_map = {'Under Priced': 1, 'Normal Price': 2, 'Over Priced': 3}
# df['pricing_category'] = df['pricing_category'].map(pricing_category_map)

# df.drop(columns=['pricing_category'], inplace=True)
# df.drop(columns=['community_price','category'], inplace=True)

# # Predictor Variabels
# X = df.iloc[:, 1:-1].values

# # Target Variabel: price
# y = df.iloc[:, -1].values
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)

# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

model_file = open('lin_regressor.pkl', 'rb')
model = pickle.load(model_file, encoding='bytes')

sc = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    surfaceArea, buildingArea, bedrooms, bathrooms, storey, buildingAge, facing, roadWidth, ownershipStatus, housePosition, readyToUse, furnished, urgent = [x for x in request.form.values()]

    buildingAgeText = "default"
    facingText = "default"
    roadWidthText = "default"
    ownershipStatusText = "default"
    housePositionText = "default"
    readyToUseText = "default"
    furnishedText = "default"
    urgentText = "default"

    data = []

    data.append(float(surfaceArea))
    data.append(float(buildingArea))
    data.append(int(bedrooms))
    data.append(int(bathrooms))
    data.append(float(storey))
    data.append(int(buildingAge))
    data.append(int(facing))
    data.append(int(roadWidth))
    data.append(int(ownershipStatus))
    data.append(int(housePosition))
    data.append(int(readyToUse))
    data.append(int(furnished))
    data.append(int(urgent))
    
    transformedData = sc.transform([data])

    prediction = model.predict(transformedData)
    output = int(prediction[0])
    str_output = f'{output:,}'.replace('.',',')

    if int(buildingAge) == 1:
        buildingAgeText = "1-4 year(s)"
    elif int(buildingAge) == 2:
        buildingAgeText = "5-10 years"
    elif int(buildingAge) == 3:
        buildingAgeText = ">10 years"
    else :
        buildingAgeText = "error"

    if int(facing) == 0 :
        facingText = "West"
    elif int(facing) == 1 :
        facingText = "South"
    elif int(facing) == 2 :
        facingText = "East"
    elif int(facing) == 3 :
        facingText = "North"
    else :
        facingText = "error"
    
    if int(roadWidth) == 1 :
        roadWidthText = "<1 Cars"
    elif int(roadWidth) == 2 :
        roadWidthText = "1-2 Cars"
    elif int(roadWidth) == 3:
        roadWidthText = ">2 Cars"
    else :
        roadWidthText = "error"

    if int(ownershipStatus) == 1:
        ownershipStatusText = "Surat Hijau"
    elif int(ownershipStatus) == 2:
        ownershipStatusText = "Pengikatan Perjanjian Jual Beli (PPJB)"
    elif int(ownershipStatus) == 3 :
        ownershipStatusText = "Hak Guna Bangunan (HGB)"
    elif int(ownershipStatus) == 4:
        ownershipStatusText = "Hak Milik (SHM)"
    else :
        ownershipStatusText = "error"
    
    if int(housePosition) == 0:
        housePositionText = "Cul De Sac"
    elif int(housePosition) == 1 :
        housePositionText = "Kantong Belakang"
    elif int(housePosition) == 2 :
        housePositionText = "Standard"
    elif int(housePosition) == 3 :
        housePositionText = "Hook"
    elif int(housePosition) == 4:
        housePositionText = "Tusuk Sate"
    else :
        housePositionText == "error"

    if int(readyToUse) == 0:
        readyToUseText = "No"
    elif int(readyToUse) == 1:
        readyToUseText = "Yes"
    else :
        readyToUseText = "error"

    if int(furnished) == 0:
        furnishedText = "No"
    elif int(furnished) == 1 :
        furnishedText = "Yes"
    else :
        furnishedText = "error"

    if int(urgent) == 0 :
        urgentText = "No"
    elif int(urgent) == 1 :
        urgentText = "Yes"
    else :
        urgentText = "error"

        

    return render_template('index.html', prediction=str_output, surfaceArea=surfaceArea, buildingArea=buildingArea, bedrooms=bedrooms, bathrooms=bathrooms, storey=storey, buildingAgeText=buildingAgeText, facingText=facingText, roadWidthText=roadWidthText, ownershipStatusText=ownershipStatusText, housePositionText=housePositionText, readyToUseText=readyToUseText, furnishedText=furnishedText, urgentText=urgentText)

app.run(debug=True)