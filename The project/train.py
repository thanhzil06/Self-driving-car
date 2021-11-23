
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten, Activation, MaxPooling2D
from keras.regularizers import l2
from utils import INPUT_SHAPE, batch_generator

# Thư mục để dữ liệu
data_dir = 'traindata'  
# Đọc file driving_log.csv với các cột tương ứng
data_df = pd.read_csv(os.path.join(os.getcwd(), data_dir, 'driving_log.csv'),
                      names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

# Lấy đường dẫn đến ảnh ở camera giữa, trái, phải
X = data_df[['center', 'left', 'right']].values
# Lấy góc lái của ô tô
y = data_df['steering'].values
#plt.hist(y)
#plt.show()

# Loại bỏ và chỉ lấy 1000 dữ liệu có góc lái ở 0
pos_zero = np.array(np.where(y==0)).reshape(-1, 1)
pos_none_zero = np.array(np.where(y!=0)).reshape(-1, 1)
np.random.shuffle(pos_zero)
pos_zero = pos_zero[:1000] 

pos_combined = np.vstack((pos_zero, pos_none_zero))
pos_combined = list(pos_combined)

y = y[pos_combined].reshape(len(pos_combined))
X = X[pos_combined, :].reshape((len(pos_combined), 3))

# After process
#plt.hist(y)
#plt.show()

# Chia ra traing set và validation set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

# Xây dựng model
model = Sequential()

import tensorflow as tf
# model architecture
model = tf.keras.Sequential()
model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
print(INPUT_SHAPE)


# input shape (66,200,3), convolution 1
model.add(tf.keras.layers.Conv2D(filters=24, kernel_size=5, padding='same', activation='relu', input_shape=INPUT_SHAPE)) 
# max pooling 1
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
# convolution 2
model.add(tf.keras.layers.Conv2D(filters=36, kernel_size=5, activation='relu'))
# max pooling 2
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
# convolution 3

model.add(tf.keras.layers.Conv2D(filters=48, kernel_size=5, activation='relu'))
# max pooling 3
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
# convolution 4
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))

model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.add(Dense(1))
#Tong ket model da xay dung - model summary
model.summary()

nb_epoch = 10
samples_per_epoch = 500
batch_size = 32
save_best_only = True
learning_rate = 1e-4
# Checkpoint này để nói cho model lưu lại model nếu validation loss thấp nhất     
checkpoint = ModelCheckpoint('models/model-{epoch:03d}.h5',
                                 monitor='loss',
                                 verbose=0,
                                 save_best_only=save_best_only,
                                 mode='auto')
                            
# Dùng mean_squrared_error làm loss function
model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))
#model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(
                    batch_generator(data_dir, X_train, y_train, batch_size, True),
                    steps_per_epoch = samples_per_epoch, 
                    epochs = nb_epoch,
                    validation_data=batch_generator(data_dir, X_valid, y_valid, batch_size, False),
                    validation_steps =(len(X_valid)//batch_size),
                    callbacks=[checkpoint],
                    verbose=1)
print("Trained!")


