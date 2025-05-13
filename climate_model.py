import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import joblib


data = pd.read_csv('climate_plant_dataset.csv')

X = data[['Temperature', 'Humidity']].values
y = data['Plant'].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

model = Sequential([
    Dense(256, input_dim=2, activation='relu'),  
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dense(len(label_encoder.classes_), activation='softmax')
])  

optimizer = Adam(learning_rate=0.00005)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#checkpoint = ModelCheckpoint('best_climate_plant_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.5,verbose=1)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

model.save('climate_plant_model.keras')
model.save('climate_plant_model.h5')

joblib.dump(label_encoder,'label_encoder.pkl')

print("Successfully Model Trained!")
