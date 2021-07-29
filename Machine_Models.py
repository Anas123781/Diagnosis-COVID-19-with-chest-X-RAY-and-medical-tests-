import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense,Dropout,Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
import pickle
import pyrebase
from firebase import firebase
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.svm import SVC

firebaseConfig = {'apiKey': "AIzaSyA3Wp9gbMjQyyMUug_mzQDxgPky-JE3Oos",
                  'authDomain': "last-55450.firebaseapp.com",
                  'databaseURL': "https://last-55450-default-rtdb.firebaseio.com",
                  'projectId': "last-55450",
                  'storageBucket':  "last-55450.appspot.com",
                  'messagingSenderId': "328587195662",
                  'appId': "1:328587195662:web:7f03ccffaf47b9fa80935f",
                  'serviceAccount': "C:\\Users\\Anas\\Downloads\\serviceaccount.json",

                  }


firebas = pyrebase.initialize_app(firebaseConfig)
fireba = firebase.FirebaseApplication('https://last-55450-default-rtdb.firebaseio.com/')

#firebase realtime
db = firebas.database()
#firebase storge
storage=firebas.storage()





# medical_tests Model
# ___________________________________________

path1 = 'E:\\data\\dataset.txt'

dataset = pd.read_csv(path1, header=1,
                      names=['Sex', 'ALT', 'LDH', 'WBC', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'PLT1', 'NE', 'LY',
                             'MO', 'EO', 'BA', 'NET'
                          , 'LYT', 'MOT', 'EOT', 'BAT', 'Result'], delimiter="\t")



X = dataset.iloc[:, :-1] #all data without Result
y = dataset.iloc[:, -1]  #Result of data

#print(X)
#print(y)






# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, random_state=0, shuffle=True)


#print( "X_train=",X_train)
#print("X_test=",X_test)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
SVCModel = SVC(kernel='rbf', max_iter=200, C=1
               , gamma='scale', random_state=1)
SVCModel.fit(X_train, y_train)

# Accuracy
#print('SVCModel Train Score is : ' , SVCModel.score(X_train, y_train))
#print('SVCModel Test Score is : ' , SVCModel.score(X_test, y_test))
 #print('----------------------------------------------------')

# Calculating Prediction
y_pred = SVCModel.predict(X_test)
#print('Predicted Value for SVCModel is : ' , list(y_pred[:10]))
#print('Predicted Value for SVCModel is : ' , list(y_test[:10]))
# ----------------------------------------------------
# Calculating Confusion Matrix
#CM = confusion_matrix(y_test, y_pred)
# drawing confusion matrix
#sns.heatmap(CM, center=True)
#plt.show()


# Image_Model
model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(200,200,3)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss=keras.losses.binary_crossentropy,optimizer='adam',metrics=['accuracy'])

#///////////////////////////////////////////////////////////////////////////////////

result = db.child("patient").get()

for i in result.each():
    x = i.key()
    alt = (db.child('patient').child(x).child('alt').get().val())
    Heamoglobin = (db.child('patient').child(x).child('heamoglobin').get().val())
    Red_Cells_Count = (db.child('patient').child(x).child('red_Cells_Count').get().val())
    Haematocrit = (db.child('patient').child(x).child('haematocrit').get().val())
    MCV = (db.child('patient').child(x).child('mcv').get().val())
    MCH = (db.child('patient').child(x).child('mch').get().val())
    MCHC = (db.child('patient').child(x).child('mchc').get().val())
    RDW = (db.child('patient').child(x).child('rdw').get().val())
    Platelets_Count = (db.child('patient').child(x).child('platelets_Count').get().val())
    Neutrophils = (db.child('patient').child(x).child('neutrophils').get().val())
    Lymphocytes = (db.child('patient').child(x).child('lymphocytes').get().val())
    Monocytes = (db.child('patient').child(x).child('monocytes').get().val())
    Eosinophils = (db.child('patient').child(x).child('eosinophils').get().val())
    Basophils = (db.child('patient').child(x).child('basophils').get().val())
    Neutrophils_absolute_count = (db.child('patient').child(x).child('neutrophils_absolute_count').get().val())
    Lymphocytes_absolute_count = (db.child('patient').child(x).child('lymphocytes_absolute_count').get().val())
    Monocytes_absolute_count = (db.child('patient').child(x).child('monocytes_absolute_count').get().val())
    Eosinophils_absolute_count = (db.child('patient').child(x).child('eosinophils_absolute_count').get().val())
    Basophils_absolute_count = (db.child('patient').child(x).child('basophils_absolute_count').get().val())
    CRP = (db.child('patient').child(x).child('crp').get().val())
    D_Dimmer = (db.child('patient').child(x).child('D-Dimmer').get().val())
    LDH = (db.child('patient').child(x).child('ldh').get().val())
    Serum_Ferritin = (db.child('patient').child(x).child('serum_Ferritin').get().val())
    sex = (db.child('patient').child(x).child('sex').get().val())
    Total_Leucocytic_Count = (db.child('patient').child(x).child('total_Leucocytic_Count').get().val())
    Name = (db.child('patient').child(x).child('name').get().val())
    Phone = (db.child('patient').child(x).child('phone').get().val())
    ID = (db.child('patient').child(x).child('id').get().val())
    Age = (db.child('patient').child(x).child('age').get().val())
    Period = (db.child('patient').child(x).child('period').get().val())
    chest_pain = (db.child('patient').child(x).child('chest_pain').get().val())
    difficulty_of_breathing = (db.child('patient').child(x).child('difficulty_of_breathing').get().val())
    dry_cough = (db.child('patient').child(x).child('dry_cough').get().val())
    fever = (db.child('patient').child(x).child('fever').get().val())
    heahache = (db.child('patient').child(x).child('heahache').get().val())
    loss_of_speech_movement = (db.child('patient').child(x).child('loss_of_speech_movement').get().val())
    loss_of_taste_or_smell = (db.child('patient').child(x).child('loss_of_taste_or_smell').get().val())
    rash_on_skin = (db.child('patient').child(x).child('rash_on_skin').get().val())
    sore_throat = (db.child('patient').child(x).child('sore_throat').get().val())
    tirdness = (db.child('patient').child(x).child('tirdness').get().val())
    conjunctivitis = (db.child('patient').child(x).child('conjunctivitis').get().val())
    Result = (db.child('patient').child(x).child('result').get().val())
    Image_Path=(db.child('patient').child(x).child('image_view').get().val())
    x_new = [
        [sex, alt, LDH, RDW, Red_Cells_Count, Heamoglobin, Haematocrit, MCV, MCH, MCHC, Platelets_Count, Neutrophils,
         Lymphocytes, Monocytes, Eosinophils, Basophils, Neutrophils_absolute_count, Lymphocytes_absolute_count,
         Monocytes_absolute_count, Eosinophils_absolute_count, Basophils_absolute_count
         ]]
    with open('saved_model.pickle', 'wb')as f:
        pickle.dump(SVCModel, f)
    with open('saved_model.pickle', 'rb')as f:
        SVCModel = pickle.load(f)
    result_tests_Model = SVCModel.predict(x_new)
    #download Image
    storage.child(Image_Path).download("images\\"+ID+".jpeg")
    model.load_weights("weights.best.hdf5")

    dir_path = "images\\"+ID+".jpeg"
    img = image.load_img(dir_path, target_size=(200, 200))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    result_image = model.predict(images)




    if (alt <= 35) & (CRP >= 0 & CRP <= 5) & (Serum_Ferritin >= 10 & Serum_Ferritin <= 120) & (
            LDH >= 0 & LDH <= 247) & (D_Dimmer >= 1.6 & D_Dimmer <= 4.9 & Period>=14 ) :
        if ( result_image == 0):
            Result = db.child('patient').child(x).update({"result":"Negative"})

        if (result_image == 1):
            Result = db.child('patient').child(x).update({"result": "Suspected"})
    else:

        if (result_tests_Model == 1 & result_image == 1):
        # update Result of this patient to (Positive) in database
            Result = db.child('patient').child(x).update({"result": "Positive"})
        if(result_tests_Model == 0 & result_image == 0 &Period>=14):
            Result = db.child('patient').child(x).update({"result": "Negative"})
        if(result_tests_Model == 0 & result_image == 0 &Period<14):
            Result = db.child('patient').child(x).update({"result": "wait"})
        if ((result_tests_Model == 1 & result_image == 0) | (result_tests_Model == 0 & result_image == 1)):
            Result = db.child('patient').child(x).update({"result": "Suspected"})

