import tensorflow_datasets as tfds
import tensorflow as tf, numpy as np
from tensorflow.keras import layers,models
import time
import sys # para coger parametros de entrada

import cv2 # opencv para el reescalado y crop de imagenes
import math
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

#ideas de
#https://www.tutorialspoint.com/how-to-find-discrete-cosine-transform-of-an-image-using-opencv-python
#https://riunet.upv.es/bitstream/handle/10251/123378/Agust%C3%AD%20-%20Un%20ejemplo%20visual%20del%20uso%20de%20la%20DCT%20en%20el%20est%C3%A1ndar%20JPEG%20mediante%20OpenCV.pdf?sequence=1
#             PARAMETROS CONFIGURABLES
#=======================================================
F=int(sys.argv[1])
G=int(sys.argv[2])
batch_size = 2
resolution=64
total_images=500
total_images=(total_images//batch_size)*batch_size # asi ajustamos a numero entero de batches
myshufle=True
my_split=0.2
epochs=25
#epochs = int(25/math.sqrt(F))
freq=32
print ("TOTAL images=", total_images)
#=======================================================

###########################################################################################################################
def prepareImages(num_freq):
    y_final=np.zeros(0)
    y_final2=np.zeros(0)
    #y_final.shape=(total_images, 64,64,1)
    contenido = os.listdir('./orig_images/')
    print (contenido)
    idx=0
    for img_name in contenido:
        idx=idx+1
        print(img_name)
        name='./orig_images/'+img_name
        img=cv2.imread(name)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img,(resolution,resolution)) # reduce a 64x64
        #la paso a BN
        name2='./train/all/'+img_name
        #cv2.imshow("orig", img)
        #cv2.waitKey(0);

        #cv2.imwrite(name2,img) # guarda version en BN y 64x64  --> quito la escritura
        
        
        #print("------float orig-----")
        #print (img)
        img=np.float32(img)# ahora ya es un decimal
        img=img/255.0 # ahora ya entre 0 y 1
        """
        print ("  img:")
        print ("el maximo es:",np.max(img))
        print ("el minimo es:",np.min(img))
        print ("el media es:",np.mean(img))
        """
        h,w=img.shape[:2]
        img_dct=cv2.dct(img)#,cv2.DCT_INVERSE)

        img_dct[16:64,0:64]=0
        img_dct[0:64,16:64]=0
        img_dct2=img_dct[0:16,0:16]
        y_final2=np.append(y_final2,img_dct2/100+0.5)
        y_final2.shape=(idx,16,16)
        
        
        #print ("img_dct shape=", img_dct.shape)
        y_final=np.append(y_final,img_dct/100+0.5)
        y_final.shape=(idx,64,64)
        
        #print ("yfinal shape=", y_final[-1].shape)

        #img_dct=img_dct/100.0 # ajuste
        """
        print ("  DCT:")
        print ("el maximo es:",np.max(img_dct)) #supongo max=64
        print ("el minimo es:",np.min(img_dct))#supongo min=-64
        print ("el media es:",np.mean(img_dct)) #supongo max=64
        """
        #elmax=np.max(img_dct)
        
        #k=128/np.max(img_dct)
        #hay que elegir un K, no vale escoger el mejor
        k=4 #4 #2048/128
        off=128
        #img_dct[img_dct>255]=255
        #img_dct[img_dct<0]=0
        
        img_dct=(img_dct*k+off)
        #control de limites
        img_dct[img_dct>255]=255
        img_dct[img_dct<0]=0
        #print("------float -----")
        #print (img_dct)
        #img_dct=img_dct*255
        img_dct = np.uint8(img_dct)
        #img_dct8 =np.array(img_dct, dtype=np.uint8)
        #print("------8 bit-----")
        #print (img_dct)
        

        #img_dct=cv2.dft(img)#,cv2.DCT_INVERSE)
        #img_dct_crop=img_dct[0:num_freq,0:num_freq]#filtro paso bajo
        
        #img_dct[num_freq:h,0:w]=0#filtro paso bajo
        #img_dct[0:h,num_freq:w]=0#filtro paso bajo
        #img_dct[num_freq:h,num_freq:w]=0#filtro paso bajo
        # convert to uint8 para guardar la DCT como imagen
        #img_dct=img_dct*255.0
        #img_dct[img_dct>255]=255
        #img_dct[img_dct<0]=0
        
        #img_dct8 = np.uint8(img_dct)
        #img_dct8 =np.array(img_dct, dtype=np.uint8)
        #img_dct_crop8 = np.uint8(img_dct_crop)
        #img_dct8crop=img_dct8[0:num_freq,0:num_freq]#crop =filtro paso bajo
        name_dct="./trainDCT/all/dct_"+img_name
        #cv2.imshow("DCT", img_dct)
        #cv2.waitKey(0);
        
        #cv2.imwrite(name_dct,img_dct)--> quito la escritura


        #reconstruccion
        img3= np.float32(img_dct)
        #img_dct32 =np.array(img_dct8, dtype=np.float32)
        #img_dct=img_dct/255.0
        img3=(img3-off)/k
        #img3=img3*100.0
        img_dct_inv=cv2.idct(img3)

        img_dct_inv=img_dct_inv *255
        #control de limites
        img_dct_inv[img_dct_inv>255]=255
        img_dct_inv[img_dct_inv<0]=0
        img_dct_inv = np.uint8(img_dct_inv)
        #img_dct_inv=cv2.resize(img_dct_inv,(256,256))
        #print (img_dct_crop8)

        a=(y_final[idx-1]-0.5)*100.0

        #begin con solo 16x16 frecuencias
        a=(y_final2[idx-1]-0.5)*100.0
        b=np.zeros(64*64)
        b.shape=(64,64)
        b[0:16,0:16]=a
        a=b
        #end con solo 16
        #cv2.imshow("dct", a)
        #cv2.waitKey(0)
        #print("a shape:",a.shape)
        a=cv2.idct(a)
        #a=a*255
        #cv2.imshow("inv", a)
        #cv2.waitKey(0)
        #cv2.imshow("inv", img_dct_inv)
        #cv2.waitKey(0);
        name_idct='./idct/idct_'+str(num_freq)+"_"+img_name
        #cv2.imwrite(name_idct,img_dct_inv)---> quito la escritura
    #return y_final
    return y_final2
###########################################################################################################################
def computex():
    x_final=np.zeros(0)
    contenido = os.listdir('./orig_images/')
    #print (contenido)
    idx=0
    for img_name in contenido:
        idx=idx+1
        print("loading ", img_name,"   ", end='\r')
        name='./orig_images/'+img_name
        img=cv2.imread(name)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img,(resolution,resolution)) # reduce a 64x64
        img=np.float32(img)# ahora ya es un decimal
        img=img/255.0 # ahora ya entre 0 y 1
        x_final=np.append(x_final,img)
        #cv2.imshow("orig", img)
        #cv2.waitKey(0)
    x_final.shape=(idx,resolution,resolution,1)
    print("", end="\n")
    return x_final
###########################################################################################################################

def computey16(x_train):
    y_final2=np.zeros(0)
    idx=0
    for img in x_train:
        idx=idx+1
        img.shape=(64,64)
        #cv2.imshow("orig", img)
        #cv2.waitKey(0)
        img_dct=cv2.dct(img)#,cv2.DCT_INVERSE)
        img_dct[16:64,0:64]=0
        img_dct[0:64,16:64]=0
        img_dct2=img_dct[0:16,0:16]
        y_final2=np.append(y_final2,img_dct2/100+0.5)
        y_final2.shape=(idx,16,16)
        #cv2.imshow("orig", img)
        #cv2.waitKey(0)
    return y_final2
###########################################################################################################################
def computey(x_train):
    y_final2=np.zeros(0)
    idx=0
    for img in x_train:
        idx=idx+1
        img.shape=(64,64)
        #prueba=cv2.resize(img,(256,256)) 
        #cv2.imshow("orig", prueba)
        #cv2.waitKey(0)
        img_dct=cv2.dct(img)#,cv2.DCT_INVERSE)
        #prueba=cv2.resize(img_dct,(256,256)) 
        #cv2.imshow("dct", prueba)
        #cv2.waitKey(0)
        img_dct[freq:64,0:64]=0
        img_dct[0:64,freq:64]=0
        #prueba=cv2.resize(img_dct,(256,256)) 
        #cv2.imshow("filter", prueba)
        #cv2.waitKey(0)
        img_dct2=img_dct[0:freq,0:freq]
        y_final2=np.append(y_final2,img_dct2/100+0.5)
        y_final2.shape=(idx,freq,freq)
        #cv2.imshow("orig", img)
        #cv2.waitKey(0)
    return y_final2

###########################################################################################################################
print ("**************************************")
print ("*     REGRESION NO LOCAL MLP         *")
print ("*   programa de test                 *")
print ("*                                    *")
print ("* usage:                             *")
print ("* py regre.py <F>  <G>               *")
print ("*   F= ratio                         *")
print ("*   G= subred  [0..F-1]              *")
print ("*                                    *")
print ("**************************************")

"""
#las imagenes las genero y guardo en /train/todas
#ya reescaladas y en gris
# y lo mismo debo hacer con las imagenes DCT
num_freq=64

y_final=prepareImages(num_freq) # solo una vez
y_final[y_final>=1.0]=0.999999
#y_final=y_final/100.0 # la dct da coeficientes >0, incluso he visto 50
#exit()


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
print ("x_train shape=", x_train.shape)
x_train=x_train[:count,:resolution,:resolution,:1]
print ("x_train shape=", x_train.shape)
print("x_train shape[0]",x_train[0].shape)
"""
x_train=computex()
countx=(total_images//batch_size)*batch_size
x_train=x_train[:countx,:resolution,:resolution,:1]

#ahora vamos a crear el Y dataset

#debe corresponderse exactamente con lo que sea X
y_final2=computey(x_train)
y_final2=np.reshape(y_final2,(countx,freq*freq,1)) #aplanamos la salida

#reduccion de la Y para entrenar
salida=int((freq*freq)/F)
print("salida:",salida)
#y_final3=y_final2[:countx,0:salida,:1]#primeras freq
fini=int((freq*freq/2)-salida/2)#int(freq*freq/2) # inicio a la mitad (freq medias) para sacar "la media" de las loss (mas o menos)
#fini=int(salida/2) # mas realista?
fini=G*salida #parametro G
#fini=freq*4
ffin=fini+salida#freq*freq
print ("fini=", fini, "  ffin=", ffin)
y_final3=y_final2[:countx,fini:ffin,:1]# freq escogidas

#filtro, nueva tecnica. NO funciona. parece que converge peor si son frecuencias salteadas
"""
y_final3=np.zeros(0)
items=0
for i in range(0,freq*freq):
    if (i%F==0):
        items=items+1
        print (i,",", end='')
        y_final3=np.append(y_final3,y_final2[:countx,i,:1])
print("", end="\n")
print("añadidos " , items, " coeficientes a la salida")
y_final3=np.reshape(y_final3,(countx,salida,1))
"""
""" esto tampoco va bien
y_final3=np.zeros(0)
items=0
for i in range(0,freq*freq):
    x=i%freq
    y=i/freq
    if (x<math.sqrt(salida) and y<math.sqrt(salida)): #primer cuadradito
        items=items+1
        print (i,",", end='')
        y_final3=np.append(y_final3,y_final2[:countx,i,:1])
print("", end="\n")
print("añadidos " , items, " coeficientes a la salida")
y_final3=np.reshape(y_final3,(countx,salida,1))
"""


y_final2=y_final3
print("y final2 shape=",y_final2.shape)

"""

#---------------------------------------------------------------

resolutiony=num_freq
#datageny = ImageDataGenerator(rescale=1.0 / 255.0)
datageny = ImageDataGenerator()#rescale=1.0 / 255.0)
train_generatory = datageny.flow_from_directory(
    #'pattern',
    "./trainDCT",
    target_size=(int(resolutiony), int(resolutiony)),
    batch_size=batch_size,
    class_mode='input',
    #class_mode=None,
    subset="training"
)

num_stepsy = len(train_generatory)
print ("num_stepsy:",num_stepsy)
# Initialize an empty list to store the batches
batchesy = []
#total_images=100 # numero total de imagenes con las que queremos entrenar
county=0
num_batchesy=0
# Iterate through the generator to collect all batches
for _ in range(num_steps):
    batchy = train_generatory.next()[0]
    batchesy.append(batch)
    county=county+batch_size
    num_batchesy=num_batchesy+1
    if (county>=total_images):
        break

# Concatenate the batches to create the full dataset
y_train = np.concatenate(batchesy, axis=0)
print("y train shape=",y_train.shape)
y_train = y_train[:count,:num_freq,:num_freq,:1]
print("y train shape=",y_train.shape)
y_train=np.reshape(y_train,(count,num_freq*num_freq,1))
#y_train.shape=(count,num_freq*num_freq,1)
print("y train shape=",y_train.shape)

#y_final=y_final[0:count]
#y_final=np.reshape(y_final,(count,num_freq*num_freq,1))#aplanamos la salida
#print("y final shape=",y_final.shape)
"""
#y_final2=y_final[0:count]
#y_final2=np.reshape(y_final2,(count,16*16,1)) #aplanamos la salida
#print("y final2 shape=",y_final2.shape)


#y_train=(y_train -128)/4
"""
#prueba de concepto
#---------------------
prueba=y_train[0]
print ("prueba shape=", prueba.shape)
prueba.shape=(64,64)
print ("prueba shape=", prueba.shape)

prueba2 = prueba*255
prueba2 = cv2.idct(prueba2)
prueba2 = np.uint8(prueba2)
prueba2=cv2.resize(prueba2,(256,256))
cv2.imshow("idct(y)", prueba2)

cv2.waitKey(0)
#ahora la x
#-----------
pruebax= x_train[0]
#print (pruebax)
pruebax=pruebax*255
pruebax = np.uint8(pruebax)
#pruebax=cv2.resize(pruebax,(256,256))
cv2.imshow("x", pruebax)
cv2.waitKey(0)

#ahora de x-->y-->x
#-------------------
pruebax= x_train[0]
#pruebax=pruebax/255.0
pruebay=cv2.dct(pruebax)
pruebay=pruebay*255
pruebay = np.uint8(pruebay)
cv2.imshow("dctx", pruebay)
cv2.waitKey(0)
pruebayx=np.float32(pruebay)
pruebayx=pruebayx/255.0
pruebayx = cv2.idct(pruebayx)
pruebayx=pruebayx*255
pruebayx = np.uint8(pruebayx)
cv2.imshow("idct(yx)", pruebayx)

cv2.waitKey(0)
exit()
"""
#ahora la red
input_shape = (int(resolution), int(resolution), 1)
input_data = tf.keras.layers.Input(shape=input_shape)
n=freq*freq

Fr=F
#Fr=1 # todos los recursos de la red original
#x = tf.keras.layers.Conv2D(int(512/F), (8, 8), activation='relu', padding='same')(input_data)
#x = tf.keras.layers.MaxPooling2D((8,8 ), padding='same')(x)
x = tf.keras.layers.Conv2D(int(64/F), (16, 16), activation='relu', padding='same')(input_data)
x = tf.keras.layers.MaxPooling2D((2,2 ), padding='same')(x)

x = tf.keras.layers.Conv2D(int(64/F), (4, 4), activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D((2,2 ), padding='same')(x)

x=tf.keras.layers.Flatten()(x) 
x=tf.keras.layers.Dense(int(4*n/F),activation='relu')(x)
output=tf.keras.layers.Dense(int(n/F),activation='sigmoid')(x)
regresor = tf.keras.models.Model(input_data, output)

print(regresor.summary())
print("PARAM:")
print("  F=", F)
print("  G=", G)
print("  batch:",batch_size )
print("  images:",countx )
print("  shuffle:",myshufle )
print("  split:", my_split)
print("  epoc:", epochs)
print("  freq", freq)
print("  original output:", freq*freq)
print("  subnet output:", salida)
print("")
optimizer = tf.keras.optimizers.Adam() #lr=lr)
regresor.compile(loss='mse', optimizer=optimizer)


#print ("el maximo es:",np.max(y_final)) #supongo max=64
#print ("el minimo es:",np.min(y_final))#supongo min=-64
#print(" y final:")

#print (y_final[0])
#print(" x train:")
#print (x_train[0]*255)

history2=regresor.fit(x_train,
                         #y_train,
                         #y_final,
                         y_final2,
                         epochs=epochs,
                         batch_size=batch_size,
                         shuffle=myshufle,
                         validation_split=my_split
                         )

if (F>1):
    exit()
name_model="regresorDCT03.h5"#+ str(F)+".h5"
regresor.save(name_model)

cosa="Regressor using Caltech101"

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
plt.show()

#AHORA VEMOS UNAS MUESTRAS

for i in range(1, countx):
    print("input shape:",x_train[i-1].shape)
    # ESTA ES LA ORIGINAL
    orig=cv2.resize(x_train[i-1],(256,256)) 
    cv2.imshow("orig", orig) 
    cv2.waitKey(0);
    prueba=regresor.predict(x_train[i-1:i]) #ejecucion del modelo



    #print ("el maximo es:",np.max(prueba)) 
    #print ("el minimo es:",np.min(prueba))
    #print (prueba)

    # ESTA ES LA RECONSTRUIDA
    prueba=(prueba-0.5)*100 # la dct estaba comprimida
    prueba=prueba[0] # primer elemento de la lista
    
    #en caso de final2
    #prueba=(y_final2[i-1]-0.5)*100 # fake
    prueba2=np.zeros(resolution*resolution)
    prueba2.shape=(resolution,resolution)
    prueba.shape=(freq,freq)
    prueba2[0:freq,0:freq]=prueba[0:freq,0:freq]
    prueba=prueba2


    #prueba=y_train[0]
    #print ("prueba shape=", prueba.shape)
    prueba.shape=(resolution,resolution)
    prueba2 = cv2.idct(prueba)
    #prueba2 = np.uint8(prueba2)
    prueba2=cv2.resize(prueba2,(256,256)) 
    cv2.imshow("idct from Model", prueba2) # si prueba 2 es float , es valido rango (0..1)
    cv2.waitKey(0);

    #ESTA ES LA Y
    prueba=(y_final2[i-1]-0.5)*100 # fake
    prueba2=np.zeros(resolution*resolution)
    prueba2.shape=(resolution,resolution)
    prueba.shape=(freq,freq)
    prueba2[0:freq,0:freq]=prueba[0:freq,0:freq]
    prueba=prueba2
    prueba.shape=(resolution,resolution)
    prueba2 = cv2.idct(prueba)
    #prueba2 = np.uint8(prueba2)
    prueba2=cv2.resize(prueba2,(256,256)) 
    cv2.imshow("idct perfect", prueba2) # si prueba 2 es float , es valido rango (0..1)
    cv2.waitKey(0);
    
