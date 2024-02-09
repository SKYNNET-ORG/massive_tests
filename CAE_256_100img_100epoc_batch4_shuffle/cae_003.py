# adaptado de https://medium.com/@patelharsh7458/demystifying-image-compression-with-autoencoders-a-light-hearted-journey-4b36334c9651

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

import sys
from random import random
import cv2
import math
#             PARAMETROS CONFIGURABLES
#=======================================================
F=int(sys.argv[1])
# el original era 2 pero 2 sale muy mal al deconstruir
# el mejor es 4, tanto para original como para decons
# he probado con 2,4,8,16 y 32

batch_size = 4# #16#4 #4 #5 #8 # original era 2 (50imgs=5*10)

# 2 sale mal con encoder deconstruido (0.1)
# 16 no converge (0.073) se atasca
# 32 se atasca
# 8  se queda clavado en 0.1 y train en 0.068
# 4 se clava en 0.065 - es el mejor
# shuffe=false se atasca

resolution=256
total_images=int(100/(F*F)) # numero total de imagenes con las que queremos entrenar
total_images=100
#ojo si no es dibisible entre batch_size, seran mas imagenes que las que digamos.
myshufle=True
my_split=0.2
epochs = 100 #500  #400 #45
print ("TOTAL images=", total_images)
m=total_images #*F*F
#=======================================================


datagen = ImageDataGenerator(rescale=1.0 / 255.0)
train_generator = datagen.flow_from_directory(
    #'pattern',
    "./train",
    target_size=(int(resolution), int(resolution)),
    batch_size=batch_size,
    class_mode='input',
    #class_mode=None,
    subset="training"
)

# Determine the number of steps (batches of samples) in the generator
num_steps = len(train_generator)
print ("num_steps:",num_steps)

# Initialize an empty list to store the batches
batches = []
#total_images=100 # numero total de imagenes con las que queremos entrenar
count=0
num_batches=0
# Iterate through the generator to collect all batches
for _ in range(num_steps):
    batch = train_generator.next()[0]
    batches.append(batch)
    count=count+batch_size
    num_batches=num_batches+1
    if (count>=total_images):
        break

# Concatenate the batches to create the full dataset

x_train = np.concatenate(batches, axis=0)

#ahora troceamos y cogemos uno de los trozos al azar
#x_train_orig = np.concatenate(batches, axis=0)

y_train=None
if (F==1):
    y_train=x_train
else:
    print (x_train.shape) # = (total_images,alto,ancho,3) =(100,256,256,3)
    print ("alto=", x_train.shape[1])

    #porcion=int(math.sqrt(x_train.shape[1]*x_train.shape[1]/F)) # area/F
    porcion=int(x_train.shape[1]/F)
    print ("porcion:", porcion)
    
    #F es el factor de reduccion de salida, no del ancho y el alto
    #por lo tanto
    Fv=resolution*resolution/(porcion*porcion)
    print ("el verdadero F es : ",Fv)

    #y_train=np.zeros(F*F*x_train.shape[0]*porcion*porcion*x_train.shape[3])
    #y_train.shape=(F*F*x_train.shape[0],porcion,porcion,x_train.shape[3])
    #mmax=int(total_images*F)
    mmax=int(total_images)
    y_train=np.zeros(mmax*porcion*porcion*x_train.shape[3])
    y_train.shape=(mmax,porcion,porcion,x_train.shape[3])
    
    print ("y shape =", y_train.shape)
    porciones=resolution / porcion
    print("porciones=",porciones)
    m=0
    
    for i in range (0,x_train.shape[0]):
        x=int(random()*porciones)
        y=int(random()*porciones)
        #x=0
        #y=0
        #print (i, "porcion=", porcion, "elegimos: ",x,y)
        #el azar no es buena idea. cojo el cuadro central
        #int(0.5+porciones/2)
        #int(0.5+porciones/2)
        #print (i, "porcion=", porcion, "elegimos: ",x,y)
        yini=y*porcion
        xini=x*porcion
        #print (" porcion= ", xini," hasta " , xini+porcion)
        #y_train[i]=x_train[i:i+1,yini:yini+porcion,xini:xini+porcion]
        for j in range (0,F):
            if (m==mmax):
                    break
            for k in range (0,F):
                if (m==mmax):
                    break
                yini=j*porcion
                xini=k*porcion

                # dependiendo de la porcion
                # escogida se obtienen resultados distintos
                # si cogemos todas las porciones, tendremos un loss medio
                # con F=4, de las 100 imagenes, en realidad solo estaremos entrenando con 25
                # ya que cada imagen contiene 4 porciones. esto daña la loss
                yini=(F//2)*porcion
                xini=(F//2)*porcion
                
                y_train[m]=x_train[i:i+1,yini:yini+porcion,xini:xini+porcion]
                m=m+1 
                #continue # me quedo solo con una porcion
                #print("i=", i, "  m=",m)
                
                # descomentar para mostrar imagenes
                mostrar=False
                if (mostrar):
                    a=y_train[m-1]*255
                    a =np.array(a, dtype=np.uint8)
                    img = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)                
                    cv2.imshow('portion', img)
                    cv2.waitKey(0);
                

                
    print("total images=",m)
#exit();

#x_train = np.concatenate(batches, axis=0, dtype=np.float32)

#encoding_dim = 512
#epochs = 100 #500  #400 #45

#creo que hay que coger la porcion, de lo contrario trata de convertir la imagen entera
#en la porcion y aunuqe logre converger en train, val se va de madre logicamente

# Define the autoencoder architecture
#CAMBIAMOS EL SHAPE DE ENTRADA Y POR TANTO CAMBIA AUTOMATICAMENTE EL DE LA SALIDA
# DEBIDO A LOS POOLING Y UPSAMPLING
input_shape = (int(resolution/F), int(resolution/F), 3)
Fe=F
Fd=F
if (F>1):
    Fe=1
    Fd=1#F#1
    
input_data = Input(shape=input_shape)
#capa inventada
#x = MaxPooling2D((F, F), padding='same')(input_data)

x = Conv2D(int(512/Fe), (3, 3), activation='relu', padding='same')(input_data)


x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(int(256/Fe), (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(int(128/Fe), (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(int(64/Fe), (3, 3), activation='relu', padding='same')(x) 
encoded = MaxPooling2D((2, 2), padding='same')(x)  # este es el "codigo" o "bottleneck"
print ("CODIGO:",encoded.shape)
# code= 64 filters of 16x16 = 64x16x16= 16384
# Decoding part
x = Conv2D(int(64/Fd), (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)

x = Conv2D(int(128/Fd), (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(int(256/Fd), (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(int(512/Fd), (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

#capa que no se puede quitar. son 3 filtros para generar 3 canales
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

#capa inventada
#decoded = MaxPooling2D((F, F), padding='same')(decoded)

autoencoder = Model(input_data, decoded)
print(autoencoder.summary())
print("")
#exit()
lr=0.0001
print("-------------------")
print("  >FACTOR: ", F)
#print("  >LR=", lr)
print("  >resolucion: ", resolution)
print("  >batch_size=", batch_size)
print("  >batches=",num_batches)
print("  >total images:", count)
#myshufle=False
print("  >shuffle= ", myshufle)
print("  >epocas= ", epochs)
#my_split=0.2
print("  >split=", my_split)
print("  >y shape =", y_train.shape)
print("  >codigo: ", encoded.shape)
print("total images=",m)
print("-------------------")

optimizer = tf.keras.optimizers.Adam() #lr=lr)
autoencoder.compile(loss='mse', optimizer=optimizer)

#print("-------MEMORY --------------------")
#print(tf.config.experimental.get_memory_info('CPU:0'))
#print("")
#autoencoder.compile(optimizer='adam', loss='mean_squared_error')#,
                    #metrics=['acc'])

#print(autoencoder.history)
#exit()
# Train the autoencoder using x_train
#history=
history2=autoencoder.fit(y_train,
                         y_train,
                         #train_generator,#x_train,
                         #train_generator,#x_train,
                         epochs=epochs,
                         batch_size=batch_size,
                         shuffle=myshufle,
                         #validation_data=(x_train,y_train)
                         validation_split=my_split
                         #validation_data=train_generator
                         )
# Save the trained autoencoder model to a file
name_model="pattern_"+ str(F)+".h5"
autoencoder.save(name_model)


# Visualize loss  and accuracy history
#--------------------------------------
cosa="CAE using Caltech101"

plt.title(cosa)
#xticks = np.arange(0, epocas, max(1,epocas/10))
#tics=max(1,epocas/10)
#plt.axes.xaxis.set_major_locator(MaxNLocator(ticks))
#xaxis.set_major_locator(ticker.MaxNLocator(4))
plt.plot(history2.history['loss'], 'r--') #en mlp no voy a pintar la loss
plt.plot(history2.history['val_loss'], 'g--') #en mlp no voy a pintar la loss

#plt.axes([1,2,3,4,5,6,7,8,9,10,11,12])
#plt.plot(autoencoder.history['accuracy'], 'b-')
#plt.plot(autoencoder.history['val_accuracy'], 'g-')
#plt.legend(['Training Loss', 'Training Accuracy'])
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')

#antes del show imprimo todo
print("-------SUMMARY ------------")
print("  >FACTOR: ", F)
print("  >resolucion: ", resolution)
print("  >batch_size=", batch_size)
print("  >batches=",num_batches)
print("  >total images:", count)
print("  >shuffle= ", myshufle)
print("  >epocas= ", epochs)
print("  >split=", my_split)
print("  >y shape =", y_train.shape)
print("  >codigo: ", encoded.shape)
print("total images=",m)
print("-------------------")
plt.show();





