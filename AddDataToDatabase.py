import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("firebaseServiceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL': '',
    'storageBucket': ''
})

ref = db.reference('Users')

data = {
    '00001':
        {
            'Name': 'Teste Teste',
            'Age': 22,
            'Last_record': '2024-03-30 10:00:00'
        }

}

for key, value in data.items():
    ref.child(key).set(value)
    print('Data added to database')