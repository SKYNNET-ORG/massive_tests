from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
# Load the saved autoencoder model
loaded_model = load_model('./pattern_1.h5')
print(loaded_model.summary())
resolution=256

# Load and preprocess custom images
total=99
#custom_images = np.zeros(total)
for i in range(0,total-1):
    print ("ejemplo ", i, " de ", total)
    #name="./train/all/"+str(i)+".png"
    name="./generated_256/"+str(i)+".png"
    print("testing image:", name)
    #name="./quitadas/"+str(i+20)+".jpg"
    custom_image = load_img(name, target_size=(resolution, resolution))
    custom_image = img_to_array(custom_image) / 255.0
    #custom_images[i]=custom_image
    print (custom_image.shape)
    img1 = cv2.cvtColor(custom_image, cv2.COLOR_BGR2RGB) 
    cv2.imshow('orig', img1)
    #cv2.waitKey(0)
    custom_images = np.array([custom_image]) #, custom_image)
    reconstructed_custom_images=loaded_model.predict(custom_images)
    img2 = cv2.cvtColor(reconstructed_custom_images[0], cv2.COLOR_BGR2RGB) 
    cv2.imshow('rebuilt', img2)
    cv2.waitKey(0);
"""
custom_image1 = load_img("first.png", target_size=(resolution, resolution))
custom_image1 = img_to_array(custom_image1) / 255.0

custom_image2 = load_img("second.png", target_size=(resolution, resolution))
custom_image2 = img_to_array(custom_image2) / 255.0

custom_image3 = load_img("third.png", target_size=(resolution, resolution))
custom_image3 = img_to_array(custom_image2) / 255.0
"""
# Combine the custom images into an array
#custom_images = np.array([custom_image1, custom_image2 , custom_image3])

# Use your autoencoder model to predict and calculate the reconstruction errors for custom images

"""
reconstructed_custom_images = loaded_model.predict(custom_images)
mse_custom = np.mean(np.square(custom_images - reconstructed_custom_images), axis=(1, 2, 3))
print('mse_custom', mse_custom)

for i in (0,total-1):
  cv2.imshow('orig', reconstructed_custom_images[i])
  
"""
