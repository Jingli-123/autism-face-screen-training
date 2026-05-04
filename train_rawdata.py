import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
from tensorflow import keras
from tensorflow.keras.layers import Input, InputLayer, Conv2D, Activation, LeakyReLU, Concatenate, Dense, Lambda , MaxPooling2D , UpSampling2D , Conv2DTranspose, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
#import matplotlib.pyplot as plt
from tensorflow.keras.losses import MSE
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, classification_report
from tensorflow.keras import models, layers, optimizers, losses
#from tensorflow.python.keras.saving import hdf5_format
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
import h5py, itertools, collections
import itertools
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from tensorflow.keras.regularizers import L2
from keras.utils import custom_object_scope
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization, Flatten ,Layer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def datagenerator(images, labels, batchsize, mode="train"):
    while True:
        start = 0
        end = batchsize

        while start  < len(images): 
            # load your images from numpy arrays or read from directory
            x = images[start:end] 
            y = labels[start:end]
            yield x, y

            start += batchsize
            end += batchsize

image_size = (224, 224)
'''
以下为可以微调的数据
'''
batch_size = 32
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)
epochs = 50
# early_stopping = EarlyStopping(
#     monitor='val_loss',   # 监控验证集loss
#     patience=6,           # 如果验证loss连续5个epoch没改善，就停止训练
#     restore_best_weights=True,  # 恢复到验证集表现最好的权重
#     verbose=1
# )
# checkpoint = ModelCheckpoint(
#     filepath='/home/jingli/my_ai_project/best_model_3.h5', 
#     monitor='val_loss', 
#     save_best_only=True,
#     verbose=1,
# )

train_image_generator = ImageDataGenerator(
    # rescale=1./255,    
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.8, 1.2],
    rotation_range=10,
    zoom_range=0.1,
    # shear_range=0.1,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # channel_shift_range=30, 
    # fill_mode='nearest',
)
'''
以上为可以微调的数据， 另外请调整ArcFace 的参数，还有 lr
'''

valid_test_generator = ImageDataGenerator()

train_dir = "/home/jingli/my_ai_project/raw_data/small" 
val_dir = "/home/jingli/my_ai_project/raw_data/valid"
test_dir = "/home/jingli/my_ai_project/raw_data/test" 


# 1. 训练数据迭代器：应用数据增强 (train_image_generator)
train_gen2 = train_image_generator.flow_from_directory(
    train_dir, 
    target_size=image_size, 
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True 
) 

# 2. 验证数据迭代器：不应用数据增强 (valid_test_generator)
val_gen2 = valid_test_generator.flow_from_directory(
    val_dir, 
    target_size=image_size, 
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False 
)

# 3. 测试数据迭代器：不应用数据增强 (valid_test_generator)
test_gen2 = valid_test_generator.flow_from_directory(
    test_dir, 
    target_size=image_size, 
    batch_size=200, 
    class_mode='categorical',
    shuffle=False 
)


def train_datagen(train_gen2):
    while True:
        x,y =next(train_gen2)
        y = y.astype('float32')
        yield(x,y),y
        # yield((x,y),y)


def val_datagen(test_gen2):
  while True:
    x,y =next(test_gen2)
    y = y.astype('float32')
    yield(x,y),y
    # yield ([x,y],y)

tf.keras.applications.mobilenet.preprocess_input 

def createmodel():
    weight_decay = 1e-4
    img_shape=(224,224,3)
    covn_base_input = Input(shape=img_shape, name="image_input")
    input_2 = Input(shape=(2,), name="label_input")
    # covn_base = tf.keras.applications.MobileNet(weights='imagenet',input_shape=img_shape,include_top=False)
    covn_base = tf.keras.applications.MobileNet(weights='imagenet',input_tensor=covn_base_input,include_top=False)
    covn_base.trainable = True
    # for layers in covn_base.layers[:-5]:
    #     layers.trainable = False

    #Build model    
    # model = tf.keras.Sequential()
    x = covn_base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = Dropout(rate=0.5,seed = 123)(x) 
    x=keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001 )(x)
    x = Dense(128,activation='relu')(x)
    x=keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001 )(x)
    x = Dense(16,activation='relu')(x)
    x=keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001 )(x)

    output = ArcFace(2, regularizer=tf.keras.regularizers.L2(weight_decay))([x, input_2])

    model=Model(inputs = [covn_base.input,input_2], outputs = output)
    return model

class ArcFace(Layer):
    def __init__(self, n_classes=10, s=64.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = tf.keras.regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(theta + self.m)
        # sin = tf.sqrt(1 - logits**2)
        # cos_m = tf.cos(logits)
        # sin_m = tf.sin(logits)
        # target_logits = logits * cos_m - sin * sin_m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "n_classes":self.n_classes,
            "s":self.s,
            "m":self.m,
            "regularizer":self.regularizer
        })
        return config
    
model = createmodel()
model.summary()

# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=0.001,
#     decay_steps=1000,
#     decay_rate=0.96,
#     staircase=False
# )
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  #Use adam optimizer, the learning rate is 0.0001
              loss= tf.keras.losses.CategoricalCrossentropy(), #Cross entropy loss function
              metrics=["accuracy"]) 

plot_data = model.fit(
    train_datagen(train_gen2),
    steps_per_epoch=train_gen2.n // batch_size,
    epochs=epochs,
    validation_data=val_datagen(val_gen2),
    validation_steps=val_gen2.n // batch_size,
    callbacks=[reduce_lr]
    )

history = plot_data.history 
print("Saving model")
model.save("/home/jingli/my_ai_project/model_10_13_2025.h5")
print("Saved")

from tensorflow.keras.models import load_model
print("Loading model")

model = createmodel()
with custom_object_scope({'ArcFace': ArcFace, 'L2': L2}):
    model = load_model("/home/jingli/my_ai_project/model_10_13_2025.h5")
print("Model loaded successfully")

from tensorflow.keras.utils import plot_model
# print("start plotting")
# plot_model(model,to_file="model.png",show_shapes=True,show_layer_names=False,rankdir="TB",expand_nested=False,dpi=96)
# print("plot done!") 
test_history = []

test_history.append(
    model.evaluate(val_datagen(test_gen2), steps=len(test_gen2))
)

# print("Loading best model")
# model = tf.keras.models.load_model(
#     "/home/jingli/my_ai_project/model_4.h5",
#     custom_objects={"ArcFace": ArcFace}
# )
# print("Model loaded successfully")
# test_history = []

# test_history.append(
#     model.evaluate(val_datagen(test_gen2), steps=len(test_gen2))
# )
print("history")
print(test_history)
print("history, done!")

def plot_training_metrics(history):
    """
    绘制模型训练和验证的损失与准确率曲线。
    """
    
    # 检查是否有准确率数据
    has_accuracy = 'accuracy' in history and 'val_accuracy' in history

    # 根据是否有准确率，确定子图数量
    num_plots = 2 if has_accuracy else 1
    plt.figure(figsize=(6 * num_plots, 5))

    # --- 1. 绘制损失 (Loss) 曲线 ---
    plt.subplot(1, num_plots, 1) 
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # --- 2. 绘制准确率 (Accuracy) 曲线 ---
    if has_accuracy:
        plt.subplot(1, num_plots, 2)
        plt.plot(history['accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout() 
    #plt.show() # 在本地环境中显示图像
    # 如果在远程服务器上运行，请用下面这行保存图像
    plt.savefig('training_curves.png')

# 调用函数开始绘图
plot_training_metrics(history)

