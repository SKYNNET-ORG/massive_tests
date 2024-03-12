
# codigo base tomado de https://github.com/tonywu71/vision-transformer
# Tensorflow implementation of Image Classification with Vision Transformer
# VIT architecture is following the one from An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, Dosovitskiy, 2021.

# dependencias:
# para instalar algo (ejemplo): py -m pip install tensorflow==2.12
# para desinstalar algo (ejemplo): py -m pip uninstall tensorflow  

# tensorflow 2.12   (ya trae numpy 1.25)
# numpy 1.25
# tensorflow-datasets
# tfds-nightly
# seaborn NO HACE FALTA
# matplotlib 3.8.3
# opencv 4.6.0 (al menos, aunque la 4.9 tambien vale)


import argparse
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from dataloader.dataloader import load_mnist_dataset
#from model.hparams import read_config
#from model.vision_transformer import create_vit_classifier
#from plot.learning_curve import plot_learning_curve
from typing import List


import time
import sys # para coger parametros de entrada

import cv2 # opencv para el reescalado y crop de imagenes
import math
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf, numpy as np
import os
#######################################################################################
def adaptImages(x_train,samples,data_type,a, h2,w2, channels2):
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
            img_orig = img #cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # las magenes llegan en BGR y para hacer show necesitamos RGB
            cv2.imshow('orig', img_orig)
            print("img2.shape=", img2.shape)
            img3 = img2 #cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) # las magenes llegan en BGR y para hacer show necesitamos RGB
            cv2.imshow('Image', img3)
            #img3 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
            #cv2.imshow('Image', img3)
            
            cv2.waitKey(0);
    
    return j




##################################################################################
class Patches(layers.Layer):
    """Create a a set of image patches from input. The patches all have
    a size of patch_size * patch_size.
    """

    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    """The `PatchEncoder` layer will linearly transform a patch by projecting it into a
    vector of size `projection_dim`. In addition, it adds a learnable position
    embedding to the projected vector.
    """
    
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
############################################################################################
def get_data_augmentation_layer(image_size: int, normalization: bool=True) -> keras.layers.Layer:
    list_layers = []
    
    if normalization:
        list_layers.append(layers.Normalization()) # TODO: Implement adapt in `create_vit_classifier`
    
    list_layers.extend([
            layers.Resizing(image_size, image_size),
            # layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ])
    
    data_augmentation = keras.Sequential(
        list_layers,
        name="data_augmentation",
    )
    return data_augmentation

############################################################################################
def mlp(x: tf.Tensor, hidden_units: List[int], dropout_rate: float) -> tf.Tensor:
    """Multi-Layer Perceptron

    Args:
        x (tf.Tensor): Input
        hidden_units (List[int])
        dropout_rate (float)

    Returns:
        tf.Tensor: Output
    """
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

    
##################################################################################
def create_vit_classifier(input_shape,
                          num_classes: int,
                          image_size: int,
                          patch_size: int,
                          num_patches: int,
                          projection_dim: int,
                          dropout: float,
                          n_transformer_layers: int,
                          num_heads: int,
                          transformer_units: List[int],
                          mlp_head_units: List[int],
                          normalization: bool=False):
    inputs = layers.Input(shape=input_shape)
    print ("num patches= ", num_patches)
    # Augment data.
    data_augmentation = get_data_augmentation_layer(image_size=image_size, normalization=normalization)
    augmented = data_augmentation(inputs)
    
    # Create patches.
    patches = Patches(patch_size)(augmented)
    
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(n_transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(dropout)(representation)
    
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=dropout)
    
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    
    return model
##################################################################################
#def run_experiment(model, ds_train, ds_test, num_epocas, my_split) -> tf.keras.callbacks.History:
def run_experiment(model, a,  y_train, num_epocas, my_split, batch_size,lr) -> tf.keras.callbacks.History:
    # --- Read config ---
    #config = read_config()
    #lr=1e-3 #1e-2 #1e-4
    #optimizer = tf.optimizers.Adam(learning_rate=config["learning_rate"])
    optimizer = tf.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )


    # --- CHECKPOINTS ---
    """
    checkpoint_filepath = "./checkpoints/mnist"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
    
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        patience=3,
        monitor="val_loss",
        mode="min",
        restore_best_weights=True
    )
    
    log_dir = f'logs/mnist/' + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    """
    
    # --- TRAINING ---
    history = model.fit(
        #ds_train,
        a,
        y_train,
        #epochs=config["num_epochs"],
        epochs=num_epocas,
        #validation_data=ds_test,
        batch_size=batch_size,
        shuffle=True,
        validation_split=my_split,
        #callbacks=[checkpoint_callback, early_stopping_callback, tensorboard_callback],
    )


    # --- EVALUATION ---
    """
    model.load_weights(checkpoint_filepath)
    _, accuracy = model.evaluate(ds_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    S"""
    return history
##################################################################################
def loadDataset(tipo, ds_name, directorio):
    # tipo :
    #   "local"  si es un directorio con subdirectorios de imagenes por ejemplo si directorio == "./disney/
    #        ./disney/mickey/ contiene imagenes de mickey ( es decir, en todos ellos Y_train será el nombre del subdirectorio "mickey" )
    #        ./disney/donald/ contiene imagenes de donald ( es decir, en todos ellos Y_train será "donald")
    #        ./disney/pluto/ contiene imagenes de pluto ( es decir, en todos ellos Y_train será "pluto")
    #
    #   "standar" si es un dataset de los que se pueden cargar con el modulo tdfs de tensorflow
    #
    #  esta funcion retorna x_train, y_train
    ds_dir=ds_name #directorio de descarga
    lote_size=1 
    if (tipo=="standar"):
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
                     #split="train[:10%]", #prueba inicial
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
        print()
        
        #contenido dataset
        print("ds_train contents (", ds_name,")")
        print("--------------------------------")
        print(ds_train)
        print()
        print("dataset to numpy conversion (", ds_name,")")
        print("------------------------------------------")
        ds_train_npy=tfds.as_numpy(ds_train)

        print("  convirtiendo DS en arrays numpy...(", ds_name,")")
        x_train = np.array(list(map(lambda x: x[0], ds_train_npy)))
        y_train = np.array(list(map(lambda x: x[1], ds_train_npy)))


        print("  conversion OK")
        print("   x_train numpy shape:",x_train.shape)
        print("   y_train numpy shape:",y_train.shape)
        print("")

        print()

        return n_classes, x_train, y_train
    
    else : # tipo local
        print(" dataset local")
        #resolution=1000 # resolucion igual para todos pero en numero total de pixeles (al cuadrado)
        x_train=[] #np.zeros(0,dtype=np.uint8)
        y_train=[]#np.zeros(0,dtype=np.float32)
        carpetas = os.listdir(directorio)
        idx=0
        n_classes=0
        for subdir in carpetas:
            n_classes=n_classes+1
            categoria=n_classes-1
            imagenes = os.listdir(directorio+"/"+subdir)
            for img_name in imagenes:
                idx=idx+1
                #print("loading ", img_name,"     cat=",categoria, end='\r')
                print("loading ", img_name,"     cat=",categoria)
                name=directorio+"/"+subdir+"/"+img_name
                img=cv2.imread(name)
                h_orig,w_orig, channels_orig=img.shape
               
                #print("shape orig=",img.shape)
                #img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #factor=math.sqrt((resolution*resolution)/(h_orig*w_orig))
                #h_fin=int(h_orig*factor)
                #w_fin=int(w_orig*factor)
                #img=cv2.resize(img,(h_fin,w_fin)) # reduce a resolicion fija para todas las imagenes

                #img=np.float32(img)# ahora ya es un decimal
                #img=img/255.0 # ahora ya entre 0 y 1

                #x_train=np.append(x_train,img)
                #y_train=np.append(y_train,categoria)

                x_train.append(img)
                y_train.append(categoria)

                
                #cv2.imshow("orig", img)
                #cv2.waitKey(0)
        x_train=np.array(x_train)
        y_train=np.array(y_train)
        print("x_train.shape", x_train.shape)
        #x_train.shape=(idx,resolution,resolution,3)

        print("", end="\n")
        return n_classes, x_train, y_train
        
##################################################################################
def main():

    # ESTA PARTE HAY QUE TOCARLA PARA CARGAR EL DATASET DESEADO, YA SEA ESTANDAR O LOCAL
    #==================================================================================
    ds_name="caltech101" #9k images  variable sizes around 200x300 , 101 cat - 130 MB. train OK <---buen candidato aunque pocas imagenes pero grandes
    #ds_name="omniglot"
    #ds_name="mnist"
    #ds_name="disney"
    
    # si quieres cargar imagenes de dataset standar de tensorflow para hacer tu dataset , descomenta esto:
    n_classes,x_train,y_train = loadDataset("standar", ds_name, "")
    
    # si quieres cargar imagenes locales para hacer tu dataset , descomenta esto:
    #n_classes,x_train,y_train = loadDataset("local", ds_name, "./disney/")
    #==================================================================================
   
    #deconstruccion
    #------------------
    print("deconstruccion")
    print("----------------")

    n=int(n_classes/F) # numero final de categorias
    print("  n_clases: ", n_classes, "   reduction to ", n)

    #calculo del numero de maquinas
    n=int(n_classes/F)
    numerador=math.factorial(int(n_classes/(n/2)))
    denominador=2*math.factorial(int(n_classes/(n/2) -2))
    machines=numerador/denominador
    machines=math.ceil(numerador/denominador)
    print("  NUM MAQUINAS=", machines)
    print("")



    print ("Filtering input data for training a subnet... ")
    print ("----------------------------------------------")
   
    cat_min=int(n*G)
    cat_max=int(n*G+n)
    print (" subred ", G, " cat min=",cat_min,"  cat_max=", cat_max)
    print("")
    train_filter = np.where((y_train >=cat_min) & (y_train<cat_max)) # para poder usar G

    x_train, y_train = x_train[train_filter], y_train[train_filter]

    #falta alterar y_train para poder entrenar y lo hacemos asi:
    y_train=y_train-cat_min
    
    print ("  number of examples after filtering:")
    print('   x_train: ' + str(x_train.shape))
    print('   y_train: ' + str(y_train.shape))
    print("")

    #crop y resscalado de imagenes
    #------------------------------
    print ("Adaptando imagenes...(", ds_name,")")
    print("------------------------------------")
    w2=128
    h2=128
    if (ds_name=="caltech101"):
        w2=60 
        h2=60 #32#128 
    if (ds_name=="omniglot"):
        w2=105 #64 #32 #crop #32
        h2=105 #64 #32

    if (ds_name=="mnist"):
        w2=28
        h2=28 
    if (ds_name=="disney"): 
        w2=36
        h2=36
        
    print ("ds_name=", ds_name, "-->",  "h2=", h2, "w2=", w2)
    print("")
    channels=x_train[0].shape[2] # todos los elementos tienen mismo num canales
    samples=x_train.shape[0]

    data_type= x_train[0].dtype
    channels2=channels # es por si queremos pasar a bn, basta con poner channels2=1


    # para omniglot y mnist, usaremos 1 solo canal porque son casi grises
    if (ds_name=="omniglot"):
        channels2=1

    if (ds_name=="mnist"):
        channels2=1
        
    print ("channels: ", channels2)
    # creamos un nuevo array de tipo float para almacenar la imagen resultante del procesamiento
    # la hacemos de tipo float para poder pasar despues a 0..1 dividiendo entre 255
    a = np.zeros(h2*w2*channels2*samples, dtype=np.float32) #data_type)
    a.shape=(samples,h2,w2,channels2)
    print("nuevo array: samples:", samples, " h2=",h2," w2=",w2," channels2=",channels2)
    print("")
    print("Adaptando imagenes... (", ds_name,")")

    valid_samples=adaptImages(x_train,samples,data_type, a, h2,w2, channels2)
    if (valid_samples<samples):
        a=a[0:valid_samples]
        print("VALID SAMPLES= ", valid_samples)
    print("")


    #Normalize the pixel values by deviding each pixel by 255
    print("Normalizando...", ds_name)
    print("-------------------------")
    #x_train, x_test = x_train / 255.0, x_test / 255.0
    print("array A --> dtype=", a.dtype)
    print("array train --> dtype=",data_type)
    if (data_type==np.uint8): # solo si es 8 bit, que lo logico es que lo sea
        a=a / 255.0
    else:
        print ("WARNING no es 8 bit")
        exit()

    print("datos cargados en arrays")






    
    # --- CONFIGURACION : MANIPULAR PARA ALTERAR EL TRANSFORMER Y SUS PARAMETROS DE TRAINING ---
    #============================================================================================
    batch_size= 32 #8 #32# 32#64 para disney, usar 8 porque hay pocas imagenes. para caltech usar 32
    input_shape=[h2, w2, channels2] # alto final, ancho final, canales.
    image_size=w2 # lado del tamaño final de las imagenes
    embeddings_dim=int(32/F) # bytes de cada vector
    transformer_layers=3 # 3 encoders
    num_heads=8 
    transformer_units =[512, embeddings_dim] # Size of the transformer layers
    mlp_head_units=[256]  # Size of the dense layers of the final classifier. 
    num_epocas=50 #10
    my_split=0.2 # un 20% del conjunto de train lo usamos para validar
    lr=1e-3 #1e-2 #1e-4 # learning rate  (default de keras es 0.001 es decir 1e-3)
    num_patches_lado=6 #6 # los patches que caben por lado
    num_patches=num_patches_lado*num_patches_lado # numero total de patches
    patch_size=w2//num_patches_lado # lado de un patch
    if (patch_size!=w2/num_patches_lado):
        print(" imagen no puede dividirse en numero entero de patches")
        exit()
    #============================================================================================
   
        
    # --- Get model ---
    
    vit_classifier = create_vit_classifier(input_shape=input_shape, #config["input_shape"],
                                           num_classes=n, #config["num_classes"],
                                           image_size=image_size, #config["image_size"],
                                           patch_size=patch_size,#config["patch_size"],
                                           num_patches=num_patches, #config["num_patches"],
                                           projection_dim=embeddings_dim, #config["projection_dim"],
                                           dropout=0.2,#config["dropout"],
                                           n_transformer_layers=transformer_layers, #config["n_transformer_layers"],
                                           num_heads=num_heads,#config["num_heads"],
                                           transformer_units=transformer_units,#config["transformer_units"],
                                           mlp_head_units=mlp_head_units, #config["mlp_head_units"])
                                           )

    
   
    print(vit_classifier.summary())
    print("")
    print (" ---- configuacion del VIT -----")
    print ("  > dataset =",ds_name)
    print ("  > F=", F)
    print ("  > subred G=", G)
    print ("  > categorias originales =", n_classes)
    print ("  > categorias de cada subred =",n)
    print ("  > subred ", G, " categoria min=",cat_min,"  categoria max=", cat_max) 
    print ("  > num subredes (=maquinas) =",machines)
    print ("  > batch_size=", batch_size)
    print ("  > image size (cuadrada) =", w2)
    print ("  > canales =", channels2) 
    print ("  > dim embeddings=",embeddings_dim)
    print ("  > num transformer layers =", transformer_layers)
    print ("  > num heads=", num_heads)
    print ("  > size of transformer layers =", transformer_units)
    print ("  > size MLP layers  =",  mlp_head_units)
    print ("  > epocas:", num_epocas)
    print ("  > split=", my_split)
    print ("  > learning rate (adam) =", lr)
    print ("  > num patches/lado =", num_patches_lado)
    print ("  > num total patches =", num_patches)
    print ("  > patch size (lado) =", patch_size)
    print("")
    # --- Training ---
    #history = run_experiment(vit_classifier, ds_train, ds_test, num_epocas,my_split)
    history = run_experiment(vit_classifier, a, y_train,num_epocas,my_split, batch_size,lr)

    #filename="learning_curve_VIT_"+ds_name+".png"
    #plot_learning_curve(history=history, filepath="figs/learning_curve_mnist-VIT.png")
    #plot_learning_curve(history=history, filepath=filename)

    titulo="learning_curve_VIT_"+ds_name
    plt.title(titulo)
    plt.plot(history.history['accuracy'], 'b-')
    plt.plot(history.history['val_accuracy'], 'g-')
    plt.legend(['train', 'val'], loc='upper left')
    plt.xlabel('Epoch')
    plt.ylabel('Percent')
    plt.show();
    return


if __name__ == "__main__":
    print ("**************************************")
    print ("* TEST OF VISION TRANSFORMER (VIT    *")
    print ("*   programa de test                 *")
    print ("*                                    *")
    print ("* usage:                             *")
    print ("* py VIT003.py <F> <G>               *")
    print ("*    F: reduction factor to n=N/F    *")
    print ("*    G: subnet id  [0..num machines] *")
    print ("*                                    *")
    print ("**************************************")
    F=float(sys.argv[1])
    G=int(sys.argv[2])
    parser = argparse.ArgumentParser(description="Train Vision Transformer Classifier")
    main()
