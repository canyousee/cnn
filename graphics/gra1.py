import matplotlib.pyplot as plt
import numpy as np
import keras
from keras import backend as BK
from keras.utils import np_utils
from keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Activation,Flatten,BatchNormalization
from keras.models import Sequential
import pickle
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
def unpickle(file):
    f=open(file,'rb')
    dict=pickle.load(f,encoding='bytes')
    return dict


train_files=['data_batch_'+str(i) for i in range(1,6)]
train_data_list=[]
train_label_list=[]

for f in train_files:
    fpath='/Users/mac/Desktop/graphics/cifar10/'+f
    # print(fpath)
    batch_dict=unpickle(fpath)
    batch_data=batch_dict[b'data']
    batch_labels=batch_dict[b'labels']
    train_data_list.append(batch_data)
    train_label_list.append(batch_labels)

x_train=np.concatenate(train_data_list,axis=0)
y_train=np.concatenate(train_label_list,axis=0)

test_batch=unpickle('/Users/mac/Desktop/graphics/cifar10/test_batch')
x_test=np.array(test_batch[b'data'])
y_test=np.array(test_batch[b'labels'])
label_names_batch=unpickle('/Users/mac/Desktop/graphics/cifar10/batches.meta')
label_names=label_names_batch[b'label_names']
label_names=[l.decode("utf-8") for l in label_names]
f = open("text.txt", 'w+')
print>>f,'训练集特征'
print>>f,x_train.shape
print('训练集label')
print(y_train.shape)
print('测试集特征')
print(y_train.shape)
print('测试集label')
print(y_test.shape)
print('类别名字：')
print(label_names)

num_classes=10

x_train=x_train.reshape(-1,3,32,32)
x_test=x_test.reshape(-1,3,32,32)

fig=plt.figure(figsize=(8,3))

for i in range(num_classes):
    ax=fig.add_subplot(2,5,1+i,xticks=[],yticks=[])
    idx=np.where(y_train[:]==i)[0]
    features_idx=x_train[idx,::]
    img_num=np.random.randint(features_idx.shape[0])
    im=np.transpose(features_idx[img_num,::],(1,2,0))
    ax.set_title(label_names[i])
    plt.imshow(im)
plt.show()

x_train=x_train.astype('float32')
x_test=x_test.astype('float32')

x_train/=255
x_test/=255

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)

keras.backend.set_image_data_format('channels_first')
#print('Backend:',keras.backend.image_data_format())
#print(x_train[1:].shape)
def plot_loss_and_accuracy(history):
    plt.figure(0)
    plt.plot(history.history['acc'],'r')
    plt.plot(history.history['val_acc'],'g')
    plt.xticks(np.arange(0,101,20))
    plt.rcParams['figure.figsize']=(10,8)
    plt.xlabel("Num of Epoches")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy vs Validation Accuracy")
    plt.legend(['train','validation'])

    plt.figure(1)
    plt.plot(history.history['loss'],'r')
    plt.plot(history.history['val_loss'],'g')
    plt.xticks(np.arange(0,101,20))
    plt.rcParams['figure.figsize']=(10,8)
    plt.xlabel("Num of Epoches")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Validation Loss")
    plt.legend(['train','validation'])

    plt.show()

def base_model(opt):
    model = Sequential()
    
    # 32个卷积核(feature maps),步长为1，特征图的大小不会改变（周边补充空白），
    model.add(Conv2D(32,(3,3), padding="same", input_shape=(3,32,32)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # channel是在前面 (theano后台)
    MaxPooling2D(pool_size=(2, 2), data_format="channels_first")
    model.add(Dropout(0.25))
    
    # 64个卷积核
    model.add(Conv2D(64,(3,3),padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    MaxPooling2D(pool_size=(2, 2), data_format="channels_first")
    model.add(Dropout(0.25))
    
    model.add(Flatten())   # Flatten layer
    model.add(Dense(512))  # fully connected layer with 512 units
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes)) # Fully connected output layer with 10 units
    model.add(Activation('softmax')) # softmax activation function
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy']) # 要优化的是准确率
    return model

opt1=keras.optimizers.rmsprop(lr=0.0001,decay=1e-6)
cnn2=base_model(opt1)
cnn2.summary()

history=cnn2.fit(x_train,y_train,epochs=100,batch_size=32,validation_data=(x_test,y_test),shuffle=True)
score2=cnn2.evaluate(x_test,y_test)
print>>f,"损失值为{0:.2f},准确率为{1:.2%}".format(score2[0],score2[1]))

plot_loss_and_accuracy(history)


from sklearn.metrics import classification_report, confusion_matrix
Y_pred = cnn2.predict(x_test, verbose=2)
y_pred = np.argmax(Y_pred, axis=1)
 
# for ix in range(10):
#     print(ix, confusion_matrix(np.argmax(y_test,axis=1),y_pred)[ix].sum())
cm = confusion_matrix(np.argmax(y_test,axis=1),y_pred)
print>>f,cm

# Visualizing of confusion matrix
import seaborn as sn
import pandas  as pd
 
 
df_cm = pd.DataFrame(cm, range(10),range(10))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, 
            annot=True,
            annot_kws={"size": 12})# font size
plt.show()
f.close