import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

cred = credentials.Certificate("firebaseServiceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL': '',
    'storageBucket': ''
})

folderPath = 'images'
pathList = os.listdir(folderPath)

imgList = []
personId = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    personId.append(path.split('.')[0])

    #Envia as imagens para o database
    fileName = f'{folderPath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)

print(personId)


def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
print('Encoding Started...')
encodeListKnown = findEncodings(imgList)
encodeListKnownWithId = [encodeListKnown, personId]
print('Encoding Completed')

file = open("EncodeFile.p", "wb")
pickle.dump(encodeListKnownWithId, file)
file.close()
print('Encoding File Saved')