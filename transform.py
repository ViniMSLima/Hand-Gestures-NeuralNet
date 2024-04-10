import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os

directory = r'C:/Users/disrct/Desktop/VC3'

os.chdir(directory)

# Transformada de Fourier
def fft(img):
    img = np.fft.fft2(img)
    img = np.fft.fftshift(img)
    return img

# Inversa (retorna para imagem original)
def ifft(fimg):
    fimg = np.fft.ifftshift(fimg)
    fimg = np.fft.ifft2(fimg)
    return fimg

# Obtém a magnitude da imagem
def mag(img):
    absvalue = np.abs(img)
    magnitude = 20 * np.log(absvalue)
    return magnitude

# Normaliza a imagem entre 0 e 255
def norm(img):
    img = cv.normalize(
        img, None, 0, 255,
        cv.NORM_MINMAX
    )
    return img

# Melhor para ver imagens da transformada e imagens pequenas em geral.
def show(img):
    plt.imshow(img, cmap = 'gray')
    plt.show()
    return img

def read_images_from_directory(directory):
    images = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".png"):
                img_path = os.path.join(subdir, file)
                img = cv.imread(img_path, cv.COLOR_BGRA2GRAY)
                images.append(img)
    return images


for j in range (6):
    directories = [f"{directory}/ds2/{j}"]
    images = []
    for dir in directories:
        images.extend(read_images_from_directory(dir))

    for i, img in enumerate(images):
        img = fft(img)
        img = mag(img)
        
        cv.imwrite(os.path.join(f"{directory}/ds3/{j}", f"{i}.png"), img)