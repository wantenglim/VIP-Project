from pywebio.exceptions import SessionClosedException
from pywebio.exceptions import SessionClosedException
from pywebio.input import *
from pywebio.output import *
import numpy as np
import cv2
import math
import os
from matplotlib import pyplot as plt
import time
import json
from pywebio import start_server
from imgaug import augmenters as iaa
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
import pywebio
import sys
import asyncio

def demo():
    logo = open('VIP-Project/logo-beige.png', 'rb').read()  
    put_row([None, None, put_image(logo, width='110px'), put_markdown('# Pixtono'), None, None],size=50)
    put_markdown('### A Non-Photorealistic Rendering and Pixelation of Face Images Application')
    put_markdown('#### TDS3651 Visual Information Processing - Project')
    put_markdown('##### Group 5: Lee Min Xuan (1181302793) Lim Wan Teng (1181100769) Tan Jia Qi (1191301879) Vickey Tan (1181101852)')
    
    userinput()
    #put example images for preprocess, style transform one by one
    #put_image(logo, width='px') enhance
    #put_image(logo, width='px') filter
    #put_image(logo, width='px') segment
    #put_image(logo, width='px') cartoon
    #put_image(logo, width='px') oil paint
    #put_image(logo, width='px') pencil sketch
    #put_image(logo, width='px') watercolour
    #put_image(logo, width='px') pixelization

def userinput():
    # upload image and direct to image preprocessing (enhancement, filtering, fg-bg segmentation)
    put_text('Welcome to Pixtono! You may now upload your image for editing.')
    put_text('>> Preferably a full face photo to generate the best results.')
    img = file_upload("Select a image:", accept="image/*") 
    
    put_text('Image pre-procesing includes the functions of image enhancement, filtering and background changing.')
    put_text('Image style transformation includes the effects of cartoon, oil painting, pencil sketch, watercolour painting, and pixelation.')
    put_text('You may choose to do image pre-processing before image style transformation or proceed to image style transformation directly.')
    operation = radio("Choose",options = ['Image Pre-Processing','Image Style Transformation'])
    if operation == "Image Pre-Processing":
        img_preprocess(img)
    elif operation == "Image Style Transformation":
        img_styletransform(img)

def img_preprocess(img):
    put_text('Please choose the function preferred, you can continue with other function after one.')
    operation = radio("Choose",options = ['Image Enhancement','Image Filtering','Image Background Changing'])
    if operation == "Image Enhancement":
        img_enhance(img)
    elif operation == "Image Filtering":
        img_filter(img)
    elif operation == "Image Background Changing":
        img_background(img)
        
def img_enhance(img):
    #codes
    
    operation = radio("Choose",options = ['Image Filtering','Image Background Changing','Image Style Transformation'])
    if operation == "Image Filtering":
        img_filter(img)
    elif operation == "Image Background Changing":
        img_background(img)
    elif operation == "Image Style Transformation":
        img_styletransform(img)
        
def img_filter(img):
    # filtering codes
    #sepia effect
    def sepia(img):
        img_sepia = np.array(img, dtype=np.float64) # converting to float to prevent loss
        img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                    [0.349, 0.686, 0.168],
                                    [0.393, 0.769, 0.189]])) # multipying image with special sepia matrix
        img_sepia[np.where(img_sepia > 255)] = 255 # normalizing values greater than 255 to 255
        img_sepia = np.array(img_sepia, dtype=np.uint8)
        return img_sepia
        
    def getBGR(img, lj_map, i, j):
        b, g, r = img[i][j]
        x = int(g/4 + int(b/32) * 63)
        y = int(r/4 + int((b%32) / 4) * 63)
        return lj_map[x][y]

    def filter_sepia(img):
        #img = cv2.imread(img)
        result = img['content']
        result = np.frombuffer(result, np.uint8)
        result = cv2.imdecode(result, cv2.IMREAD_COLOR)
        #result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        result = sepia(result)
        #print(result)
        is_success, im_buf_arr = cv2.imencode(".jpg", result)
        byte_im = im_buf_arr.tobytes()
        put_markdown('## **Result**')
        put_row([put_text("Before: "), None, put_text("After: ")])
        put_row([put_image(img['content']), None, put_image(byte_im)])
        img['content'] = byte_im
        put_file(label="Download",name='filter_'+ img['filename'], content=img['content'])
        #put_button("Retry", onclick=img_filter, color='primary', outline=True)
        put_html('<hr>')
        #put_markdown("**If you wish to proceed, click the next button.**")
        #put_button("Next", onclick=filter_lighting(img), color='primary', outline=True)
        return img['content']
        
    def filter_lighting(img):
        image = img['content']
        image = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        rows, cols = image.shape[:2]
        centerX = rows / 2 - 20
        centerY = cols / 2 + 20
        radius = min(centerX, centerY)
        strength = 100
        dst = np.zeros((rows, cols, 3), dtype="uint8")
        for i in range(rows):
            for j in range(cols):
                distance = math.pow((centerY-j), 2) + math.pow((centerX-i), 2)
                B = image[i,j][0]
                G = image[i,j][1]
                R = image[i,j][2]
                if (distance < radius * radius):
                    val = (int)(strength*( 1.0 - math.sqrt(distance) / radius ))
                    B = image[i,j][0] + val
                    G = image[i,j][1] + val
                    R = image[i,j][2] + val
                    B = min(255, max(0, B))
                    G = min(255, max(0, G))
                    R = min(255, max(0, R))
                    dst[i,j] = np.uint8((B, G, R))
                    result = dst
                else:
                    dst[i,j] = np.uint8((B, G, R))
                    result = dst
        is_success, im_buf_arr = cv2.imencode(".jpg", result)
        byte_im = im_buf_arr.tobytes()
        put_markdown('## **Result**')
        put_row([put_text("Before: "), None, put_text("After: ")])
        put_row([put_image(img['content']), None, put_image(byte_im)])
        img['content'] = byte_im
        put_file(label="Download",name='filter_'+ img['filename'], content=img['content'])
        #put_button("Retry", onclick=img_filter, color='primary', outline=True)
        return img['content']

    def filter_clarendon(img,lj_map):
        image = img['content']
        image = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        lj_map = np.frombuffer(lj_map, np.uint8)
        lj_map = cv2.imdecode(lj_map, cv2.IMREAD_COLOR)
        rows, cols = image.shape[:2]
        dst = np.zeros((rows, cols, 3), dtype="uint8")
        for i in range(rows):
            for j in range(cols):
                dst[i][j] = getBGR(image, lj_map, i, j)
        result = dst
        is_success, im_buf_arr = cv2.imencode(".jpg", result)
        byte_im = im_buf_arr.tobytes()
        put_markdown('## **Result**')
        put_row([put_text("Before: "), None, put_text("After: ")])
        put_row([put_image(img['content']), None, put_image(byte_im)])
        img['content'] = byte_im
        put_file(label="Download",name='filter_'+ img['filename'], content=img['content'])
        #put_button("Retry", onclick=img_filter, color='primary', outline=True)
        return img['content']
       
    lj_map = open('VIP-Project/lj_map.png', 'rb').read()
    operation = radio("Choose",options = ['filter_sepia','filter_lighting','filter_clarendon'])
    if operation == "filter_sepia":
        filter_sepia(img)
    elif operation == "filter_lighting":
        filter_lighting(img)
    elif operation == "filter_clarendon":
        filter_clarendon(img,lj_map)
    
    # continue to img preprocess/style transform
    operation = radio("Choose",options = ['Image Enhancement','Image Background Changing','Image Style Transformation'])
    if operation == "Image Enhancement":
        img_enhance(img)
    elif operation == "Image Background Changing":
        img_background(img)
    elif operation == "Image Style Transformation":
        img_styletransform(img)
        
def img_background(img):
    #codes
    
    operation = radio("Choose",options = ['Image Enhancement','Image Filtering','Image Background Changing'])
    if operation == "Image Enhancement":
        img_enhance(img)
    elif operation == "Image Filtering":
        img_filter(img)
    elif operation == "Image Background Changing":
        img_background(img)

def img_styletransform(img):
    put_text('Please choose the effect preferred, you can only choose one effect then choose to continue with pixelization.')
    operation = radio("Choose",options = ['Cartoon','Oil Painting','Pencil Sktech','Watercolour Painting'])
    if operation == "Cartoon":
        img_cartoon(img)
    elif operation == "Oil Painting":
        img_oilpaint(img)
    elif operation == "Pencil Sktech":
        img_pencilsketch(img)
    elif operation == "Watercolour Painting":
        img_watercolour(img)
        
if __name__ == "__main__":
    try:
        demo()
    except SessionClosedException:
        print("The session was closed unexpectedly")