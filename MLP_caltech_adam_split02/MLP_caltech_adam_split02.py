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
        if (i<0):
            img_orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # las magenes llegan en BGR y para hacer show necesitamos RGB
            cv2.imshow('orig', img_orig)
            print("img2.shape=", img2.shape)
            img3 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) # las magenes llegan en BGR y para hacer show necesitamos RGB
            cv2.imshow('Image', img3)
            #img3 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
            #cv2.imshow('Image', img3)
            
            cv2.waitKey(0);
    
    return j




###########################################################################################################################
print ("**************************************")
print ("*     MASSIVE MLP                    *")
print ("*   programa de test                 *")
print ("*                                    *")
print ("* usage:                             *")
print ("* py massive_MLP.py <RF>             *")
print ("*    RF: reduction factor to N/RF    *")
reduccion=float(sys.argv[1])

print ("* input params:                      *")
print ("*    - RF", reduccion)
print ("*                                    *")
print ("**************************************")

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # solo CPU


# los datasets disponibles para el "builder" de datasets de TF se encuentran en
# https://www.tensorflow.org/datasets/catalog/overview?hl=en
#
# se crea el builder indicando el dataset y el directorio local
# al descargar se crea un directorio, no hace falta crearlo antes
# ojo: si el dataset ocupa gigas se descargaran gigas
# para procesar lo que esta en el disco podemos hacer
# datasets pequenos, usando builder.as_dataset(size) o con el api de splits, que es mejor
# y procesamos poco a poco cada dataset
# otra forma es usar tfds que hace los mismos pasos aunque el "as_dataset" no lo se.
#
# EL PROBLEMA DEL TAMANO
#------------------------
# para crear splits pequeños se puede usar el api splits o bien el campo lote_size.
# https://stackabuse.com/split-train-test-and-validation-sets-with-tensorflow-datasets-tfds/
# quizas sea mejor usar el api split, de modo que los datasets que cargamos con load sean pequenos

# EL DATASET
#------------------------
# cars196 esta "roto" porque el enlace no funciona desde tfds.load ni tampoco con el builder
# en su lugar tenemos caltech_birds2011 que ocupa 1 GB y tiene 200 categorias (casi 12000 imagenes).
# en este paper hablan de la accuracy, entre el 75% (RCNN) y el 90% (LSTM), lo cual lo hace muy adecuado
# la cnn parece que anda en torno al 85%.
# https://towardsdatascience.com/adventures-in-pytorch-image-classification-with-caltech-birds-200-part-1-the-dataset-6e5433e9897c
# PROBLEMA: no aprende nuestra RGA. es debido a que el dataset viene con boxes. Para evitar las boxes vamos a usar otro
# entre los candidatos posibles hay dos buenos: caltech101 y cifar100, ambos de 100 categorias
# el caltech101 requiere crop, lo cual puede castigar ciertas imagenes y por lo tanto ciertas categorias. por ejemplo guitarras
# el cifar100 es de 32x32 y no requiere crop
# al final lo que aprende bien es no usar crop sino bandas negras. Tampoco aprende bien con escalado deforme. de este modo
# podemos usar cifar100 como sistema de tamano intermedio (6-7M) y caltech101 como sistema para modelo gigante (80M)

# el tamano variable de las imagenes
# ----------------------------------
# crop -> resize

# HAY QUE ELEGIR EL DATASET Y EL CROP QUE VAMOS A HACER. si en origen son de 32x32 no tiene sentido poner mas.
#--------------------------------------------------------------------------------------------------------------
#crop=32 #128 #32
#ds_name="caltech_birds2010"  # 6k images 200 cat. train nok (box)
#ds_name="caltech_birds2011" # 12k images 200 cat. train nok (box)
#ds_name="mnist"  # train ok
#ds_name="imagenet_resized/32x32" # 1.2M images 32x32 1000 cat - 2.9 GB train NOK
ds_name="caltech101" #9k images  variable sizes around 200x300 , 101 cat - 130 MB. train OK <---buen candidato aunque pocas imagenes pero grandes
# caltech101 aprende bien con un MLP simple (no CNN) 40 epoch acc=0.4 con 40 millones de params

#ds_name="cifar10" # 60k 32x32 images, 10 cat (50k train, 10k test) - 130M train ok
#ds_name="cifar100"
#ds_name="plant_village"
#ds_name= "oxford_flowers102" # validation no sube de 30% con 64x64
#ds_name="omniglot"
#ds_name="siscore/rotation"
#ds_name="geirhos_conflict_stimuli"
#ds_name="visual_domain_decathlon/cifar100" problema load
#ds_name="cifar100" #60k 32x32 images, 100 cat  (50k train, 10k test) - 130M train ok <--- buen candidato. muchas imagenes pero pequenas
#ds_name="cars196" # 16k different sizes, 196 cat (8k train 8k test). NOK load
#ds_name="domainnet" # subconjunto "real" 172K images. (120k train, 52k test). images , different sizes around 640x480, 345 cat - 5.7Gb train NOK
#ds_name="celeb_a" # celebrities , no descarga
# ds_name="food101" # 101k images 101 cat (750k train, 250k test) different sizes around 512x512 - 4.6Gb. NO ME FUNCIONA EL COMANDO LOAD
#ds_dir="caltech_birds2011"
#ds_dir="mnist"
#ds_dir="caltech_birds2010"
#ds_dir="cars196"
ds_dir=ds_name
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

#deconstruccion
#------------------
print("deconstruccion")
print("----------------")
#n=n_classes/1 # numero de salidas (no numero de subredes)
#n= int(n_classes/int(sys.argv[1])) # parametro de entrada al script

#reduccion=int(sys.argv[1]) # 2 significa reducir a la mitad YA LO HEMOS COGIDO AL PPIO PROGRAMA


#lo dejamos en 1000
#--------------------------
#train_filter = np.where((y_train <1000))
#x_train, y_train = x_train[train_filter], y_train[train_filter]
#n_classes=1000



#reducciones
#------------------

n=int(n_classes/reduccion) # numero final de categorias


print(" n_clases: ", n_classes, "   reduction to ", n)
#calculo del numero de maquinas
n=int(n_classes/reduccion)
numerador=math.factorial(int(n_classes/(n/2)))
denominador=2*math.factorial(int(n_classes/(n/2) -2))
machines=numerador/denominador
print(" NUM MAQUINAS float=", machines)
machines=math.ceil(numerador/denominador)
print(" NUM MAQUINAS=", machines)
print("")



print ("Filter data (deconstruction) ")
print ("---------------------------")
print ("  number of examples:")
train_filter = np.where((y_train <n))
x_train, y_train = x_train[train_filter], y_train[train_filter]


#test_filter = np.where((y_test <n))
#x_test, y_test = x_test[test_filter], y_test[test_filter]

print('   x_train: ' + str(x_train.shape))
print('   y_train: ' + str(y_train.shape))
#print('   x_test: ' + str(x_test.shape))
#print('   y_test: ' + str(y_test.shape))
print("")

#crop y resscalado de imagenes
#------------------------------
print ("Aadaptando imagenes...(", ds_name,")")
print("------------------------------------")
w2=128
h2=128
if (ds_name=="celeb_a"):
    w2=178 #crop #32
    h2=218 #w2
if (ds_name=="cifar100" or ds_name=="cifar10" ):
    w2=32 #crop #32
    h2=w2
if (ds_name=="caltech101"):
    w2=128 #128 #crop #32
    h2=128 #128
if (ds_name=="omniglot"):
    w2=227 #105 #64 #32 #crop #32
    h2=227 #105 #64 #32

if (ds_name=="caltech_birds2010"):
    w2=64#105 #64 #32 #crop #32
    h2=64#105 #64 #32



if (ds_name=="visual_domain_decathlon/cifar100"):
    w2=72 #64 #32 #crop #32
    h2=72 #64 #32
if (ds_name=="oxford_flowers102"):
    w2=64 #64 #32 #crop #32
    h2=64 #64 #32
if (ds_name=="imagenet_resized/32x32" ):
    w2=32 #crop #32
    h2=w2
if (ds_name=="siscore" ):
    w2=500 #crop #32
    h2=500
if (ds_name=="geirhos_conflict_stimuli"):
    w2=128 #128 #crop #32
    h2=128 #128

if (ds_name=="mnist"):
    w2=227
    h2=227 
print ("ds_name=", ds_name, "-->",  "h2=", h2, "w2=", w2)
print("")
channels=x_train[0].shape[2] # todos los elementos tienen mismo num canales
samples=x_train.shape[0]

data_type= x_train[0].dtype
channels2=channels # es por si queremos pasar a bn, basta con poner channels2=1


# para omniglot, 1 solo canal
if (ds_name=="omniglot"):
    channels2=1

if (ds_name=="mnist"):
    channels2=1
    
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
if (valid_samples<samples):
    a=a[0:valid_samples]
    print("VALID SAMPLES= ", valid_samples)
#adaptImages(x_test,b)
print("")

#x_train=cv2

# esto funciona bien si train_ds = tfds.load() pero no si train_ds=datasets['train'] y no se por que
# hay que sacar los campos viendo el print(train_ds) que nos da info de dimensiones y nombres de campos

#Normalize the pixel values by deviding each pixel by 255
print("Normalizando...", ds_name)
print("-------------------------")
#x_train, x_test = x_train / 255.0, x_test / 255.0
print("array A --> dtype=", a.dtype)
print("array train --> dtype=",data_type)
#if (a.dtype==np.uint8): # solo si es 8 bit, que lo logico es que lo sea
if (data_type==np.uint8): # solo si es 8 bit, que lo logico es que lo sea
    a=a / 255.0
    # b=b / 255.0
else:
    print ("WARNING no es 8 bit")
    exit()
#y_train=y_train/n_classes


"""
print("items info using bucle of 5 items:")
print("-----------------------------------")
for i in range(0,3):
    print ("item ", i," x =",type(a[i]), " y=", type(y_train[i]))
    print(f" item {i} a shape: {a[i].shape}")
    print ("x=",a[i])
    print ("y=",y_train[i])
    print("***************")
    #print ("item ", i," x shape=",x_train[i].shape, " y shape=", y_train[i].shape)
"""

#Noramalize the pixel values by deviding each pixel by 255
#x_train, x_test = x_train / 255.0, x_test / 255.0
#x_train2=x_train2 / 255.0
#y_train=y_train/n_classes
#(x_train, y_train)= tfds.as_numpy(train_dataset)
#ds_numpy=tfds.as_numpy(train_dataset)
#(x_train, y_train)=ds_numpy
#(x_test, y_test) =test_dataset #.load_data()
"""

for ex in ds_numpy:
  # `{'image': np.array(shape=(28, 28, 1)), 'labels': np.array(shape=())}`
  print(ex)



  
"""
#a= np.zeros(3)
#print(ds_numpy.shape)
#train_dataset = train_dataset.map(
#    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
print("datos cargados en arrays")




#el problema es que cada imagen tiene un tamano diferente

# Build the CNN-MLP  neural network . esto es un VGG-10 por analogia con vgg-16



###alexnet
"""
model = tf.keras.models.Sequential([
  #1
  layers.Conv2D(filters=96, kernel_size=(3, 3), activation='relu', input_shape=(h2, w2,channels2)),
  

  #2
  layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
  layers.MaxPooling2D((2, 2)),
  
  #3
  layers.Conv2D(filters=384, kernel_size=(3, 3), activation='relu'),
  layers.MaxPooling2D((2, 2)),
  
  #4
  #layers.Conv2D(filters=384, kernel_size=(3, 3), activation='relu'),
  #layers.MaxPooling2D((2, 2)),
  #5
  #layers.Conv2D(filters=384, kernel_size=(3, 3), activation='relu'),
  #layers.MaxPooling2D((2, 2)),

  
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(int(4*n), activation='relu'), 
  tf.keras.layers.Dense(n, activation='softmax')
])
"""
"""
###   MOLDELO CNN
# ----------------------
# se puede decir que es similar a alexnet, con menos capas conv para acotar el tiempo de computo

#filters=256# 128 #para imagenet

filters=384 #para omniglot?
densas=4096
#filters=128 #para cifar100

#if (reduccion!=1):
#    filters=max(16,filters/n)

#filters=int(filters/reduccion) # numero final de filtros
#filters=max(2, filters)

# estilo alexnet con kernels cada vez menores y con mas filtros por capa salvo
# las ultimas convolucionales que se repiten con igual numero de filtros

#1a capa =(filters/4)*w2*h2->/4 ->*filters/2->/4

#esta red es alexnet, solo que adaptada con menos filtros. alexnet es 384,256,96 y esta es 32,21,14 (guarda proporciones)
# al reducir el numero de filtros, tambien debemos reducir la capa densa porque quedan 7200 neuronas
#tras las convoluciones y lo logico es poner una densa de 3200 antes de la capa final de 1600
# es decir es alexnet con tres  adaptaciones:
#   1) resolucion 105 en lugar de 224 (se hizo para imagenet)-->con strides 2 en lugar de 4 se compensa
#   2) mismo num filtros
#   3) capa final de 1600 en lugar de 1000 y capa densa adaptada a ello y a la capa anterior que es algo menor que en el original
# https://datapeaker.com/big-data/alexnet-architecture-introduccion-a-la-arquitectura-de-alexnet/
model = tf.keras.models.Sequential([
  # 105 debe ser multiplo de strides. no puedo usar 4
  #layers.Conv2D(filters=int((filters/1.5)/1.5), kernel_size=(11, 11),strides=2, activation='relu', input_shape=(h2, w2,channels2)),

  # original
  #layers.Conv2D(filters=int(filters/4), kernel_size=(11, 11),strides=2, activation='relu', input_shape=(h2, w2,channels2)),

  layers.Conv2D(filters=int(filters/4), kernel_size=(11, 11),strides=4, activation='relu', input_shape=(h2, w2,channels2)),

  #quizas 11 en 224 tiene el tamano de 5 en 105 
  #layers.Conv2D(filters=int(filters/4), kernel_size=(5, 5),strides=2, activation='relu', input_shape=(h2, w2,channels2)),

  #layers.MaxPooling2D((2, 2)),
  layers.MaxPooling2D((3, 3), strides=2),
  
  layers.Conv2D(filters=int(filters/1.5), kernel_size=(5, 5),strides=1, padding="same",activation='relu'),
  #layers.MaxPooling2D((2, 2)),

  layers.MaxPooling2D((3, 3), strides=2),
  
  layers.Conv2D(filters=int(filters), kernel_size=(3, 3), strides=1,padding="same",activation='relu'),
  layers.Conv2D(filters=int(filters), kernel_size=(3, 3), strides=1,padding="same",activation='relu'),

  # si meto esta ultima capa, ya no converge. se estanca en 0.0016, es decir, aleatorio
  layers.Conv2D(filters=int(filters/1.5), kernel_size=(3, 3), strides=1,padding="same",activation='relu'), # si metemos esta, estamos ante alexnet
  #layers.MaxPooling2D((2, 2)),
  layers.MaxPooling2D((3, 3), strides=2),
  
  tf.keras.layers.Flatten(),
  #tf.keras.layers.Dense(int(n*2.5), activation='relu'),
  tf.keras.layers.Dense(int(densas), activation='relu'),
  tf.keras.layers.Dropout(rate=0.5),
  #tf.keras.layers.Dense(int(n*2.5), activation='relu'),
  tf.keras.layers.Dense(int(densas), activation='relu'),
  tf.keras.layers.Dropout(rate=0.5),
  tf.keras.layers.Dense(n, activation='softmax')
])

"""


# MODELO PURO MLP
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(h2, w2,channels2)),
 
  tf.keras.layers.Dense(int(n*8), activation='relu'),
  tf.keras.layers.Dense(int(n*4), activation='relu'),
  tf.keras.layers.Dense(int(n*2), activation='relu'),
  tf.keras.layers.Dense(n, activation='softmax')
])





"""
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(h2, w2,channels2)),
  #tf.keras.layers.BatchNormalization(),
  #tf.keras.layers.Dense(int(n*64), activation='relu'), 
  #tf.keras.layers.Dense(int(n*32), activation='relu'),
  #tf.keras.layers.BatchNormalization(), # probar si a lo mejor mejora
  #tf.keras.layers.Dense(int(n*16), activation='relu'),
  #tf.keras.layers.BatchNormalization(), # probar si a lo mejor mejora
  #tf.keras.layers.Dropout(rate=0.01),

  
  tf.keras.layers.Dense(int(n*8), activation='relu'),
  #tf.keras.layers.Dropout(rate=0.05),
  #tf.keras.layers.BatchNormalization(), # probar si a lo mejor mejora
  #tf.keras.layers.Dropout(rate=0.01),
  
  #tf.keras.layers.BatchNormalization(), # probar si a lo mejor mejora..... estaba aqui
  
  #tf.keras.layers.Dropout(rate=0.01),
  tf.keras.layers.Dense(int(n*4), activation='relu'),
  #tf.keras.layers.BatchNormalization(), # probar si a lo mejor mejora
  #tf.keras.layers.Dropout(rate=0.05),
  #tf.keras.layers.Dropout(rate=0.01),
  
  tf.keras.layers.Dense(int(n*2), activation='relu'),
  #tf.keras.layers.Dropout(rate=0.05),
  #tf.keras.layers.BatchNormalization(), # probar si a lo mejor mejora
  tf.keras.layers.Dense(n, activation='softmax')
])
"""
optimizador= "adam"
#optimizador= "SGD001"
#optimizador= "SGD0001"
print ("OPTIMIZER=", optimizador)
print ("---------------------------")
if (optimizador=="SGD001"):
    optimizador= tf.optimizers.SGD(lr=0.01)
elif (optimizador=="SGD0001"):
    optimizador= tf.optimizers.SGD(lr=0.001)

print(model.summary())
print("")
print("DS:", ds_name, " n_clases: ", n_classes, "factor=",reduccion," reduction to ", n, " categories using ",machines ," machines")
print("")
# Compile the model and set optimizer,loss function and metrics
model.compile(optimizer=optimizador, #optimizer='adam',
              loss='sparse_categorical_crossentropy', #sparse es cuando hay muchas categorias
              #loss='categorical_crossentropy', #cuando hay pocas
              #optimizer=tf.optimizers.SGD(lr=0.001), # lo he sacado de https://valueml.com/alexnet-implementation-in-tensorflow-using-python/
              metrics=['accuracy'])

# Finally, train or fit the model
start=time.time()
"""
epocas_orig=150#300
minimo=150 #40 # al menos siempre dos epocas

if (reduccion<5):
    minimo=int(epocas_orig/reduccion) *2

epocas=max(int(epocas_orig/reduccion), minimo) # calculo 10 y asi es facil saber cuanto cuesta cada una
"""
epocas=50
my_split=0.2 #imagenet, caltech
if (ds_name=="omniglot"):
    my_split=0.2

batch_tamano=32
if (ds_name=="omniglot"):
    batch_tamano=32

print ("epocas finales:", epocas, ", split =", my_split, "batch=", batch_tamano)
print("")

#cada invocacion a fit tiene un history, de modo que si partimos el dataset en partes
#tendremos un problema para pintar la historia de aprendizaje. Es preferible
#dejar todo el dataset junto

trained_model = model.fit(a, # si se usa batch_size en el load, habria que escribir x_train[batch_number] 
                          y_train, # si se usa batch_size en el load, habria que escribir y_train[batch_number] 
                          validation_split=my_split,
                          #validation_data=(b,y_test),
                          #batch_size default es 32
                          batch_size=batch_tamano, #16, #8, #8, # con 4 va deprisa pero es erratico, con 8 es estable y rapido. con 16 da mismos bandazos que con 8
                          #batch_size=2, # prueba bestia
                          epochs=epocas
                          )
end=time.time()
print (" tiempo transcurrido (segundos) =", (end-start))

# Visualize loss  and accuracy history
#--------------------------------------
cosa="Acc. using"+" DS="+ ds_name+", cat="+str(n_classes)+" size:"+ str(h2)+"x"+str(w2)
if (reduccion!=1):
    cosa=cosa+"\n Factor="+str(reduccion)+" --> cat="+str(n)+ ", machines="+str(machines)

plt.title(cosa)
#xticks = np.arange(0, epocas, max(1,epocas/10))
#tics=max(1,epocas/10)
#plt.axes.xaxis.set_major_locator(MaxNLocator(ticks))
#xaxis.set_major_locator(ticker.MaxNLocator(4))
#plt.plot(trained_model.history['loss'], 'r--') en mlp no voy a pintar la loss
#plt.axes([1,2,3,4,5,6,7,8,9,10,11,12])
plt.plot(trained_model.history['accuracy'], 'b-')
plt.plot(trained_model.history['val_accuracy'], 'g-')
#plt.legend(['Training Loss', 'Training Accuracy'])
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('Epoch')
plt.ylabel('Percent')
plt.show();
