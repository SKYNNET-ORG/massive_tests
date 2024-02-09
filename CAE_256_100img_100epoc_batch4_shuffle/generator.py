import tensorflow_datasets as tfds
import tensorflow as tf, numpy as np
from tensorflow.keras import layers,models
import time
import sys # para coger parametros de entrada

import cv2 # opencv para el reescalado y crop de imagenes
import math
import matplotlib.pyplot as plt

def adaptImages(x_train,a):
    j=0
    for i in range(0,samples):
        h,w, channels=x_train[i].shape
        #print("h=",h, " w=",w, "channels=", channels)
        img=x_train[i]
        #print ("dtype=",img.dtype)
        # dos estrategias: cortar o meter franjas negras para llegar a cuadrado
        # si cortamos aprende pero no aprende bien porque muchas imagenes se corrompen
        estrategia=2 # 1 =crop, 2 = franjas, 3 =escalado con deformacion

        if (estrategia==1): 
            # crop
            #---------------------
            #img=cv2.resize(img, dsize=(w,h))
            # diferenciamos segun relacion de aspecto w>=h o w<h
            if (w>=h):
                # cortamos el lado mayor
                margen=int((w-h)/2)
                img2=img[0:h, margen:margen+h]
            
            else:
                margen=int((h-w)/2)
                img2=img[margen:margen+w, 0:w]
                
        elif (estrategia==2): 
            # aumentar con franjas negras
            #------------------------
            if (w>=h):#imagen horizontal w>h
                img2= np.zeros(w*w*channels,dtype=data_type)
                #img2[img2==0]=128 #probamos 128
                img2.shape=(w,w,channels)
                ini=int((w-h)/2)
                fin=int(h+ini)
                
                #print ("ini, fin, dif =", ini, fin, (fin-ini))
                #img3=img2[ini:fin,0:w,0:3];
                #print ("shape ",x_train[i].shape, img.shape, img2.shape)
                img2[ini:fin,0:w,0:3]=img[0:h, 0:w,0:3]
                #img2 = np.float32(img2)

                #franjas de repeticion
                #img2[0:ini,0:w,0:3]=img[0:1, 0:w,0:3]
                #img2[fin:w,0:w,0:3]=img[h-1:h, 0:w,0:3]
                
            else: # imagen vertical h>w
                img2= np.zeros(h*h*channels,dtype=data_type)
                #img2[img2==0]=128 #probamos 128
                img2.shape=(h,h,channels)
                
                ini=int((h-w)/2)
                fin=int(w+ini)
                
                img2[0:h,ini:fin,0:3]=img[0:h, 0:w,0:3]
                #img2 = np.float32(img2)

                #franjas de repeticion
                #img2[0:h,0:ini,0:3]=img[0:h, 0:1,0:3]
                #img2[0:h,fin:h,0:3]=img[0:h, w-1:w,0:3]
                
        elif (estrategia==3):
            # escalado con deformacion
            #------------------------
            maxl=max(w,h)
            minl=min(w,h)
            if (maxl/minl>4/3):
                continue # cribamos
            img2=img
             
        # creamos la imagen nueva reescalada
        if (w!=w2 or h!=h2): # optimizacion
            img2=cv2.resize(img2, dsize=(w2,h2), interpolation=cv2.INTER_LINEAR) # INTER_NEAREST INTER_LINEAR, INTER_CUBIC , INTER_LANCZOS4
        
        
        #la guardamos
        
        if (channels2==1):
            #print("cambio shape")
            if (channels==3):
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) # convertimos a bn
            #b,g,r = cv2.split(img2)
            #img2=b #.shape=(h2,w2,1)
            img2.shape=(h2,w2,1)
            #printf("b shape =",b.shape)
            
        
        #print(f" item {i} xtrain shape: {x_train[i].shape}")
        a[j]=img2 # como  A es float, se copia desde uint8 a float. Es decir, a[i] es float
        j=j+1
        #print(f" item {i} a shape: {a[i].shape}")
        # las mostramos
        if (i>99 and i<500):
            img_orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # las magenes llegan en BGR y para hacer show necesitamos RGB
            #cv2.imshow('orig', img_orig)
            #print("img2.shape=", img2.shape)
            print ("saving ", i)
            img3 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) # las magenes llegan en BGR y para hacer show necesitamos RGB
            #name='./pattern/todas/'+str(i)+".jpg"
            name='./generated_256/'+str(i)+".png"
            cv2.imwrite(name,img3)
            
            #cv2.imshow('Image', img3)
            #img3 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
            #cv2.imshow('Image', img3)
            
            cv2.waitKey(0);
    
    return j




###########################################################################################################################
print ("**************************************")
print ("*     image creation from DS         *")
print ("*                                    *")
print ("**************************************")
ds_name="caltech101"
ds_dir="../"+ds_name
lote_size=1 #00


#cosa="model accuracy"+",DS="+ ds_name
#print (cosa)
#exit()

#(ds_train, ds_test), info= tfds.load(ds_name,
ds_train,  info= tfds.load(ds_name,
                     data_dir=ds_dir, #directorio de descarga
                     #en el campo splits del datasetinfo vemos los splits que contiene y aqui cogemos uno
                     # si usamos split, ya no retorna info y hay que sacarla
                     #split=['train','test'],
                     #split=["train[:10%]","test[:10%]"],
                     #split=["train[:10%]","train[:5%]"],
                     #split=["train","test"], #para caltech101
                     split="all", #para caltech101. es train + test +...
                     #split="train[:99%]", #imagenet
                     #split="train[:10%]", #imagenet
                     #supervised: retorna una estructura con dos tuplas input, label according to builder.info.supervised_keys
                     # si false, el dataset "tendra un diccionario con todas las features"
                     as_supervised=True,
                     shuffle_files=False, #True, # desordenar. lo quito para que funcione siempre igual y no dependa de ejecucion
                     #  if "batch_size" isset, add a batch dimension to examples
                     #batch_size=lote_size, #por ejemplo en lugar de (28,28,3) sera (10,28,28,3)
                     with_info=True # descarga info (tfds.core.DatasetInfo)
                     )


# Extract informative features
print("informative features")
print("--------------------")
class_names = info.features["label"].names
n_classes = info.features["label"].num_classes
print("  class names:", class_names) 
print("  num clases:", n_classes)
print("")

#tamanos de datasets
print("datasets sizes. IF batch_size is used THEN this is the number of batches")
print("-------------------------------------------------------------------------")
print("  Train set size: ", len(ds_train), " batches of ",lote_size, " elements") # Train set size
#print("  Test set size: ", len(ds_test)," batches of ",lote_size, " elements")   # Test set size
print()

#contenido dataset
print("ds_train contents (", ds_name,")")
print("--------------------------------")
print(ds_train)
print()
print("dataset to numpy conversion (", ds_name,")")
print("------------------------------------------")
ds_train_npy=tfds.as_numpy(ds_train)
#ds_test_npy=tfds.as_numpy(ds_test)

print("  convirtiendo DS en arrays numpy...(", ds_name,")")
x_train = np.array(list(map(lambda x: x[0], ds_train_npy)))
y_train = np.array(list(map(lambda x: x[1], ds_train_npy)))

#x_test = np.array(list(map(lambda x: x[0], ds_test_npy)))
#y_test = np.array(list(map(lambda x: x[1], ds_test_npy)))

print("  conversion OK")
print("   x_train numpy shape:",x_train.shape)
print("   y_train numpy shape:",y_train.shape)
#print("   x_test numpy shape:",x_test.shape)
#print("   y_test numpy shape:",y_test.shape)
print("")

print()

#crop y resscalado de imagenes
#------------------------------
print ("Aadaptando imagenes...(", ds_name,")")
print("------------------------------------")
if (ds_name=="caltech101"):
    w2=256 #128 #crop #32
    h2=256 #128
    
print ("ds_name=", ds_name, "-->",  "h2=", h2, "w2=", w2)
print("")
channels=x_train[0].shape[2] # todos los elementos tienen mismo num canales
samples=x_train.shape[0]

data_type= x_train[0].dtype
channels2=channels # es por si queremos pasar a bn, basta con poner channels2=1
print ("channels: ", channels2)
# creamos un nuevo array de tipo float para almacenar la imagen resultante del procesamiento
# la hacemos de tipo float para poder pasar despues a 0..1 dividiendo entre 255
a = np.zeros(h2*w2*channels2*samples, dtype=np.float32) #data_type)
#a = np.zeros(h2*w2*channels2*samples, dtype=np.float16) #data_type) # asi ocupa menos memoria
a.shape=(samples,h2,w2,channels2)
print(" nuevo array: samples:", samples, " h2=",h2," w2=",w2," channels2=",channels2)
#samples_test=x_test.shape[0]
#b= np.zeros(h2*w2*channels2*samples_test, dtype=np.float32) #data_type)
#b.shape=(samples_test,h2,w2,channels2)

#print("Adaptando imagenes... (", ds_name,")")

valid_samples=adaptImages(x_train,a)
