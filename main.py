import math
import tkinter as tk
from doctest import master
from math import cos, sqrt, sin, ceil, floor

import cv2
from PIL import Image as IMG
from PIL import ImageTk as IMGtk


import scipy
from skimage import io, data
import numpy as np
import pybresenham
import matplotlib.pyplot as plt

from skimage.transform import radon, iradon

from pydicom.dataset import Dataset, FileDataset
from pydicom.data import get_testdata_files
import pydicom
import datetime

detectors = 180  # number of detectors
deltaAlfa = 1  # degrees between (n) and (n+1) emitter
width = 180  # degrees between first and the last detector


def load_image(filename):  # Load image
    if filename.endswith(".dcm"):
        image=pydicom.dcmread(filename).pixel_array
        startimage=IMG.fromarray(np.uint8(image))
        startimage.save('output.png')
    else:
        im = IMG.open(filename)
        sqrWidth = np.ceil(np.sqrt(im.size[0] * im.size[1])).astype(int)
        im_resize = im.resize((sqrWidth, sqrWidth))
        im_resize.save('output.png')

        image = cv2.imread('output.png', 0)

    return image

def get_positions(alfa,radius):
    emitterX = radius * cos(alfa)
    emitterY = radius * sin(alfa)
    detectorsX = []
    detectorsY = []
    for j in range(detectors):
        detectorsX.append(radius * cos(
            alfa + math.pi - (width / 2) * math.pi / 180 + (j * (width / (detectors - 1))) * math.pi / 180))
        detectorsY.append(radius * sin(
            alfa + math.pi - (width / 2) * math.pi / 180 + (j * (width / (detectors - 1))) * math.pi / 180))
    return emitterX,emitterY, detectorsX, detectorsY


def bresenham(emitterX,emitterY,detectorX,detectorY):
    indexes = list(pybresenham.line(emitterX, emitterY, detectorX, detectorY))

    return indexes


def radon_t(image):  # Radon Transformation
    steps = floor(360 / deltaAlfa)
    side, _ = image.shape
    radius = floor(side / 2)
    sinogram = np.zeros((steps, detectors), dtype=float)
    for i in range(steps):
        print(i)
        # calculate positions of emitter and detectors
        alfa = i * deltaAlfa * math.pi / 180
        emitterX, emitterY, detectorsX, detectorsY = get_positions(alfa,radius)
        # Bresenham method to find pixels which are crossed by line
        for j in range(detectors):
            index = bresenham(emitterX, emitterY, detectorsX[j], detectorsY[j])

            for idx, k in enumerate(index):
                sinogram[i][j] += image[index[idx][0]+radius-1][index[idx][1]+radius-1]
        if (anime.get() == 1):
            smaximum = sinogram.max()
            ss=np.divide(sinogram,smaximum)
            ss=np.multiply(ss,255)
            sinogram_to_array = IMG.fromarray(np.array(ss))
            sinogram_photo = IMGtk.PhotoImage(sinogram_to_array)
            sinogram1.create_image(0, 0, image=sinogram_photo, anchor='nw')
            root.update()

    #sinogram = filter_sinogram(sinogram)

    maximum = sinogram.max()
    sinogram = np.divide(sinogram,maximum)
    sinogram = np.multiply(sinogram,255)
    if(filtr.get()==1):
        sinogram = filter_sinogram(sinogram)
        sinogram=sinogram*8
    return sinogram


def filter_sinogram(sinogram):

    for i in range(len(sinogram)):
        sinogram[i]=np.real(scipy.fft.ifft(scipy.fft.fft(sinogram[i])*np.hamming(len(sinogram[i]))))

    return sinogram

def i_radon_t(sinogram,side,inp):  # Inverse Radon Transformation
    steps = floor(360 / deltaAlfa)
    radius = floor(side / 2)
    image = np.zeros((side,side), dtype=float)

    #sinogram = filter_sinogram(sinogram)

    for i in range(steps):
        print(i)
        # calculate positions of emitter and detectors
        alfa = i * deltaAlfa * math.pi / 180
        emitterX, emitterY, detectorsX,detectorsY = get_positions(alfa,radius)

        for j in range(detectors):
            index = bresenham(emitterX, emitterY, detectorsX[j], detectorsY[j])

            for idx, k in enumerate(index):
                image[index[idx][0]+radius-1][index[idx][1]+radius-1] += sinogram[i][j]
        if(anime.get()==1):
            ii = np.divide(image,255)
            output_to_array = IMG.fromarray(np.array(ii))
            output_photo = IMGtk.PhotoImage(output_to_array)
            output.create_image(0, 0, image=output_photo, anchor='nw')
            root.update()
        out=np.divide(image,255)
        MSE_val.set("RMSE:" + str(process_MSE(inp, out)))
    image = np.divide(image,255)

    return image

def animations(sinogram):
    for i in range(360//deltaAlfa):
        sinogram_to_array = IMG.fromarray(np.array(sinogram[:i]))
        sinogram_photo = IMGtk.PhotoImage(sinogram_to_array)
        sinogram1.create_image(0, 0, image=sinogram_photo, anchor='nw')
        root.update()
def set_options():
    global detectors
    global deltaAlfa
    global width

    if len(detectors_input.get()) == 0:
        detectors = 180
    else:
        detectors = int(detectors_input.get())

    if len(alfa_input.get()) == 0:
        deltaAlfa = 1
    else:
        deltaAlfa = int(alfa_input.get())

    if len(width_input.get()) == 0:
        width = 180
    else:
        width = int(width_input.get())


def Computing():
    #Clear old images
    input.delete('all')
    sinogram1.delete('all')
    output.delete('all')
    set_options()

    #Get and load image
    filename = pic_input.get()
    img = load_image(filename)

    # SHOW INPUT IMAGE
    photo = tk.PhotoImage(file='output.png')
    input.create_image(0, 0, image=photo, anchor='nw')
    root.update()

    # SHOW SINOGRAM
    sinogram = radon_t(img)

    sinogram_to_array = IMG.fromarray(np.array(sinogram))
    sinogram_photo = IMGtk.PhotoImage(sinogram_to_array)
    sinogram1.create_image(0, 0, image=sinogram_photo, anchor='nw')
    root.update()
    #else:
     #   animation(sinogram)

    # SHOW OUTPUT
    reconstruction = i_radon_t(sinogram, img.shape[0],img)
    output_to_array = IMG.fromarray(np.array(reconstruction))
    output_photo = IMGtk.PhotoImage(output_to_array)
    output.create_image(0, 0, image=output_photo, anchor='nw')
    dicomF(reconstruction, name_input.get(), description_input.get(), sex_input.get(),birth_date_input.get())
    MSE_val.set("RMSE:" + str(process_MSE(img, reconstruction)))
    print("RSME="+str(process_MSE(img,reconstruction)))
    root.update()
    root.wait_window()

def dicomF(image, name, description, sex, birthDate):
    filename = get_testdata_files("CT_small.dcm")[0]
    ds = pydicom.dcmread(filename)

    image2 = np.asarray(image, dtype=np.uint16)
    ds.Rows = image2.shape[1]
    ds.Columns = image2.shape[0]
    ds.PixelData = image2.tostring()

    ds.PatientName = name
    ds.PatientSex = sex
    ds.PatientBirthDate = birthDate
    dt = datetime.datetime.now()
    ds.StudyDate = dt.strftime('%Y%m%d')
    timeStr = dt.strftime('%H%M%S.%d')
    ds.StudyTime = timeStr
    ds.InstitutionName = 'PP'
    ds.StudyDescription = description

    ds.save_as("Tomograf2_DICOM.dcm")

def process_MSE(input_image, current_image):
    return round(np.sqrt(((input_image - current_image) ** 2).mean()), 2)


if __name__ == '__main__':
    directory = "D:\Studia\IwM\Tomograf\pictures/"
    root = tk.Tk()
    root.resizable(0, 0)
    root.title("Tomograph")

    name = tk.StringVar()

    input = tk.Canvas(root, width=400, height=400)
    input.grid(row=0, column=0)

    sinogram1 = tk.Canvas(root, width=400, height=400)
    sinogram1.grid(row=0, column=1)

    output = tk.Canvas(root, width=1024, height=1024)
    output.grid(row=0, column=2)

    inputs = tk.Canvas(root, width=400, height=400)
    inputs.grid(row=0, column=3)

    # Patient name
    name = tk.Label(inputs, text='Name', fg="black")
    name.grid(row=0, column=0)

    name_input = tk.Entry(inputs)
    name_input.grid(row=0, column=1)

    # Patient birth date
    birth_date = tk.Label(inputs, text='Sex', fg="black")
    birth_date.grid(row=1, column=0)

    birth_date_input = tk.Entry(inputs)
    birth_date_input.grid(row=1, column=1)


    # Patient sex
    sex = tk.Label(inputs, text='Sex', fg="black")
    sex.grid(row=1, column=0)

    sex_input = tk.Entry(inputs)
    sex_input.grid(row=1, column=1)

    # Description
    description = tk.Label(inputs, text='Description', fg="black")
    description.grid(row=2, column=0)

    description_input = tk.Entry(inputs)
    description_input.grid(row=2, column=1)

    # Picture
    pic_name = tk.Label(inputs, text='Picture')
    pic_name.grid(row=3, column=0)

    pic_input = tk.Entry(inputs)
    pic_input.grid(row=3, column=1)

    # Options
    detectors_name = tk.Label(inputs, text='Detectors')
    detectors_name.grid(row=4, column=0)

    detectors_input= tk.Entry(inputs, width=5)
    detectors_input.grid(row=4, column=1)

    alfa_name = tk.Label(inputs, text='Delta alfa')
    alfa_name.grid(row=5, column=0)

    alfa_input = tk.Entry(inputs, width=5)
    alfa_input.grid(row=5, column=1)

    width_name = tk.Label(inputs, text='Width')
    width_name.grid(row=6, column=0)

    width_input = tk.Entry(inputs, width=5)
    width_input.grid(row=6, column=1)

    # CheckBoxes
     # Animation
    anime = tk.IntVar()
    animation = tk.Checkbutton(inputs, text='Animation', variable=anime, onvalue=1, offvalue=0)
    animation.grid(row=7, column=0)

    #
    filtr= tk.IntVar()
    filter = tk.Checkbutton(inputs, text='Filtr', variable=filtr, onvalue=1, offvalue=0)
    filter.grid(row=8, column=0)

    # BUTTONS
     # quit
    quit = tk.Button(inputs, text='Exit', command=root.destroy)
    quit.grid(row=9, column=0)
     # start computing
    start = tk.Button(inputs, text='Start', command=Computing)
    start.grid(row=9, column=1)

    MSE_val = tk.DoubleVar()
    MSE_val.set("RMSE:")
    RMSE_label = tk.Label(inputs, textvariable=MSE_val)
    RMSE_label.grid(row=10, column=0)

    root.mainloop()
    root.wait_window()
