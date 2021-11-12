from pywebio.input import *
from pywebio.output import *
import numpy as np
import cv2
import math
import os
from matplotlib import pyplot as plt
import time
import json
from pywebio.input import *
from pywebio.output import *
from pywebio import start_server
from imgaug import augmenters as iaa
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
import pywebio
import sys
import asyncio

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def start():
    img = file_upload("Select a image:", accept="image/*")
    lj_map = open('lj_map.png', 'rb').read()
    operation = radio("Choose",options = ['filter_sepia','filter_lighting','filter_clarendon'])
    if operation == "filter_sepia":
        filter_sepia(img)
    elif operation == "filter_lighting":
        filter_lighting(img)
    elif operation == "filter_clarendon":
        filter_clarendon(img,lj_map)
        
    continue_choice = radio("Proceed",options = ['Another_filter','Cartoonization','Sketch','Water_colour'])
    #give user stack filter once more
    if continue_choice == "Another_filter":
        operation = radio("Choose",options = ['filter_sepia','filter_lighting','filter_clarendon'])
        if operation == "filter_sepia":
            filter_sepia(img)
        elif operation == "filter_lighting":
            filter_lighting(img)
        elif operation == "filter_clarendon":
            filter_clarendon(img,lj_map)
        #put_button("Next", onclick=cartoon(img), color='primary', outline=True)
    elif continue_choice == "Cartoonization":
        cartoon_choice = radio("Proceed",options = ['Comics','Twilight','Classic'])
        if cartoon_choice == "Comics" :
            cartoon_comics(img)
        elif cartoon_choice == "Twilight":
            cartoon_twilight(img)
        elif cartoon_choice == "Classic":
            cartoon_classic(img)
    elif continue_choice == "Sketch":
        sketch()
    elif continue_choice == "Water_Colour":
        watercolor()  
        
# ---------------------------------FILTERING--------------------------------

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
    remove(scope="scope_filter_sepia")
    remove(scope="scope_filter_lighting")
    remove(scope = "scope_filter_clarendon")
    remove(scope = "scope_cartoon_comics")
    remove(scope = "scope_cartoon_twilight")
    remove(scope = "scope_cartoon_classic")
    remove(scope="scope_sketch")
    remove(scope="scope_watercolor")
    with use_scope("scope_filter_sepia"):
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
        put_button("Retry", onclick=start, color='primary', outline=True)
        put_html('<hr>')
        #put_markdown("**If you wish to proceed, click the next button.**")
        #put_button("Next", onclick=filter_lighting(img), color='primary', outline=True)
        
def filter_lighting(img):
    remove(scope="scope_filter_sepia")
    remove(scope="scope_filter_lighting")
    remove(scope = "scope_filter_clarendon")
    remove(scope = "scope_cartoon_comics")
    remove(scope = "scope_cartoon_twilight")
    remove(scope = "scope_cartoon_classic")
    remove(scope="scope_sketch")
    remove(scope="scope_watercolor")
    with use_scope("scope_filter_lighting"):
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
        put_button("Retry", onclick=start, color='primary', outline=True)

def filter_clarendon(img,lj_map):
    remove(scope="scope_filter_sepia")
    remove(scope="scope_filter_lighting")
    remove(scope = "scope_filter_clarendon")
    remove(scope = "scope_cartoon_comics")
    remove(scope = "scope_cartoon_twilight")
    remove(scope = "scope_cartoon_classic")
    remove(scope="scope_sketch")
    remove(scope="scope_watercolor")
    with use_scope("scope_filter_clarendon"):
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
        put_button("Retry", onclick=start, color='primary', outline=True)
# ---------------------------------CARTOON--------------------------------
#Create Edge Mask
def edge_mask(img, line_size, blur_value):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray_blur = cv2.medianBlur(gray, blur_value)
  edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
  return edges

def ColourQuantization(image, K=9):
    Z = image.reshape((-1, 3)) 
    Z = np.float32(Z) 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    compactness, label, center = cv2.kmeans(Z, K, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))
    return res2

#to get countours
def Countours(image):
    contoured_image = image
    gray = cv2.cvtColor(contoured_image, cv2.COLOR_BGR2GRAY) 
    edged = cv2.Canny(gray, 200, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    cv2.drawContours(contoured_image, contours, contourIdx=-1, color=6, thickness=1)
    return contoured_image

def cartoon_comics(img):
    remove(scope="scope_filter_sepia")
    remove(scope="scope_filter_lighting")
    remove(scope = "scope_filter_clarendon")
    remove(scope = "scope_cartoon_comics")
    remove(scope = "scope_cartoon_twilight")
    remove(scope = "scope_cartoon_classic")
    remove(scope="scope_sketch")
    remove(scope="scope_watercolor")   
    with use_scope("scope_cartoon_comics"):
        image = img['content']
        image = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        plt.figure(figsize= (8,10))
        plt.contourf(np.flipud(image[:,:,0]),levels=4,cmap='inferno')
        plt.axis('off')
        plt.savefig('cartooned.png',bbox_inches='tight')
        plt.close()
        result = open('cartooned.png', 'rb').read()
        put_markdown('## **Result**')
        put_row([put_text("Before: "), None, put_text("After: ")])
        put_row([put_image(img['content']), None, put_image(result)])
        img['content'] = result
        put_file(label="Download",name='filter_'+ img['filename'], content=img['content'])
        os.remove("cartooned.png") 
        put_button("Retry", onclick=start, color='primary', outline=True)
        
def cartoon_twilight(img):
    remove(scope="scope_filter_sepia")
    remove(scope="scope_filter_lighting")
    remove(scope = "scope_filter_clarendon")
    remove(scope = "scope_cartoon_comics")
    remove(scope = "scope_cartoon_twilight")
    remove(scope = "scope_cartoon_classic")
    remove(scope="scope_sketch")
    remove(scope="scope_watercolor")
    with use_scope("scope_cartoon_twilight"):
        image = img['content']
        image = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        line_size = 7
        blur_value = 7
        edges = edge_mask(image, line_size, blur_value)
        #colour quantization
        #k value determines the number of colours in the image
        total_color = 8
        k=total_color
        data = np.float32(image).reshape((-1, 3))
        # Determine criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
        # Implementing K-Means
        ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        array = center[label.flatten()]
        array = array.reshape(image.shape)
        blurred = cv2.bilateralFilter(array, d=10, sigmaColor=250,sigmaSpace=250)
        #blurred and edges
        result = cv2.bitwise_and(blurred, blurred, mask=edges)
        is_success, im_buf_arr = cv2.imencode(".jpg", result)
        byte_im = im_buf_arr.tobytes()
        put_markdown('## **Result**')
        put_row([put_text("Before: "), None, put_text("After: ")])
        put_row([put_image(img['content']), None, put_image(byte_im)])
        img['content'] = byte_im
        put_file(label="Download",name='filter_'+ img['filename'], content=img['content'])
        put_button("Retry", onclick=start, color='primary', outline=True)
        
def cartoon_classic(img):
    remove(scope="scope_filter_sepia")
    remove(scope="scope_filter_lighting")
    remove(scope = "scope_filter_clarendon")
    remove(scope = "scope_cartoon_comics")
    remove(scope = "scope_cartoon_twilight")
    remove(scope = "scope_cartoon_classic")
    remove(scope="scope_sketch")
    remove(scope="scope_watercolor")
    with use_scope("scope_cartoon_classic"):
        image = img['content']
        image = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        coloured = ColourQuantization(image)
        result = Countours(coloured)
        is_success, im_buf_arr = cv2.imencode(".jpg", result)
        byte_im = im_buf_arr.tobytes()
        put_markdown('## **Result**')
        put_row([put_text("Before: "), None, put_text("After: ")])
        put_row([put_image(img['content']), None, put_image(byte_im)])
        img['content'] = byte_im
        put_file(label="Download",name='filter_'+ img['filename'], content=img['content'])
        put_button("Retry", onclick=start, color='primary', outline=True)

# ---------------------------------WATERCOLOR--------------------------------
path = "./assets"

def check_sigma_s(sigma_s):
    if sigma_s < 0 or sigma_s > 200:
        return 'The range of sigma s should between 0 to 200.'


def check_sigma_r(sigma_r):
    if sigma_r < 0 or sigma_r > 1:
        return 'The range of sigma r should between 0 to 1.'
               
def watercolor():
    remove(scope="scope_sketch")
    remove(scope="scope_watercolor")
    remove(scope="scope_sketch")
    remove(scope="scope_filter_sepia")
    remove(scope="scope_filter_lighting")
    remove(scope = "scope_filter_clarendon")
    remove(scope = "scope_cartoon_comics")
    remove(scope = "scope_cartoon_twilight")
    remove(scope = "scope_cartoon_classic")
    with use_scope("scope_watercolor"):
        #style(put_code("Only float input can be typed."), 'color:red')
        data = input_group("WATERCOLOR",[
        file_upload("Select an image:", accept="image/*", name="image", required=True),
        input("Adjust smoothening filter sigma s : ", name='sigma_s', type=FLOAT, validate=check_sigma_s, min = 0, max=200, placeholder= "Type sigma s between 0 to 200", required=True),
        input("Adjust smoothening filter sigma r : ", name='sigma_r', type=FLOAT, validate=check_sigma_s, min = 0, max=1, placeholder= "Type sigma r between 0 to 1", required=True),
        select("Choose a color tone: ", ['blue','brown','gray', 'green', 'pink', 'purple','yellow'], name="color_tone", required=True),
        input("Adjust Mean Shift Filtering spatial window radius: ", name='spatial_radius', type=FLOAT, required=True),
        input("Adjust Mean Shift Filtering color window radius: ", name='color_radius', type=FLOAT, required=True),
        input("Adjust Mean Shift Filtering maximum level of pyramid: ", name='max', type=NUMBER, min=0, required=True),
        input("Adjust segmentation scale: ", name='scale', type=FLOAT, required=True),
        input("Adjust segmentation sigma: ", name='sigma_seg', type=FLOAT, required=True),
        input("Adjust segmentation minimun component size: ", name='min',  type=NUMBER, min = 0, required=True),
        radio("Select a texture", options=['brick','paper'], name="texture", required=True)
        ])
        
        put_processbar('bar')
        for i in range(1, 11):
            set_processbar('bar', i / 10)
            time.sleep(0.1)
        
        result = data['image']['content']
        result = np.frombuffer(result, np.uint8)
        result = cv2.imdecode(result, cv2.IMREAD_COLOR)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        # 1. Edge Preserving Filter
        # sigma s, sigma r
        result = cv2.edgePreservingFilter(result, flags=1, sigma_s=data['sigma_s'], sigma_r=data['sigma_r'],)
        
        # 2. Color Adjustment
        # color tone image
        for image_name in os.listdir(path):
            input_path = os.path.join(path, image_name)
            if image_name== data['color_tone']+".jpg":
                src_img = open(input_path, 'rb').read()    
        
        #src_img = data['colorimg']['content']
        src_img = np.frombuffer(src_img, np.uint8)
        src_img = cv2.imdecode(src_img, cv2.IMREAD_COLOR)
        #src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        src_lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        src_mean, src_stddev = cv2.meanStdDev(src_lab)
        #put_text(src_mean)
        #put_text(src_stddev)

        result_lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB).astype(np.float32)
        result_mean, result_stddev = cv2.meanStdDev(result_lab)
        #put_text(result_mean)
        #put_text(result_stddev)


        result_lab -= result_mean.reshape((1, 1, 3))
        result_lab = np.multiply(result_lab, np.divide(src_stddev.flatten(), result_stddev.flatten()).reshape((1, 1, 3)))
        result_lab += src_mean.reshape((1, 1, 3))
        result_lab = np.clip(result_lab, 0, 255)
        result = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

        
        # 3. Mean Shift Filtering
        # spatial window radius, color window radius and the maximum level of the pyramid
        result = cv2.pyrMeanShiftFiltering(result, sp=data['spatial_radius'], sr=data['color_radius'], maxLevel=data['max'])


        
        # 4. Image Segmentation and Fill Mean Value
        segments = felzenszwalb(result, scale=data['scale'], sigma=data['sigma_seg'], min_size=data['min'])

        for i in range(np.max(segments)):
            logical_segment = segments == i
            segment_img = result[logical_segment]
            result[logical_segment] = np.mean(segment_img, axis=0)
          
        
       
        # 5. Merge Texture
        # texture image
        for image_name in os.listdir(path):
            input_path = os.path.join(path, image_name)
            if image_name== data['texture']+".jpg":
                texture = open(input_path, 'rb').read()    
        #texture = data['textureimg']['content']
        texture = np.frombuffer(texture, np.uint8)
        texture = cv2.imdecode(texture, cv2.IMREAD_COLOR)
        texture = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)
        if texture.shape[0] > result.shape[0]: # crop
            texture = texture[:result.shape[0], :]
        elif texture.shape[0] < result.shape[0]: # pad
            texture = np.pad(texture, ((0, result.shape[0] - texture.shape[0]), (0, 0)), mode='reflect')
        if texture.shape[1] > result.shape[1]: # crop
            texture = texture[:, :result.shape[1]]
        elif texture.shape[1] < result.shape[1]: # pad
            texture = np.pad(texture, ((0, 0), (0, result.shape[1] - texture.shape[1])), mode='wrap')


        texture = np.clip(texture, 210, 255)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        result = cv2.merge((result, texture))
        #cv2.imwrite("abc.jpg", result)
        #result = result[::-1]
        is_success, im_buf_arr = cv2.imencode(".jpg", result)
        byte_im = im_buf_arr.tobytes()
        put_markdown('## **Result**')
        put_row([put_text("Before: "), None, put_text("After: ")])
        put_row([put_image(data['image']['content']), None, put_image(byte_im)])
        #put_image(byte_im)
        
        data['image']['content'] = byte_im
        put_file(label="Download",name='watercolor_'+ data['image']['filename'], content=data['image']['content']).onclick(lambda: toast('Your image is downloaded.'))
        put_button("Try Again", onclick=watercolor, color='primary', outline=True)
        put_html('<hr>')
        put_markdown("**If you wish to proceed to pixelation, click the next button.**")
        put_button("Next", onclick=lambda: toast("Link to pixelation"), color='primary', outline=True)

# ---------------------------------SKETCH--------------------------------

def check_kernel_odd(ksize):
    if ksize % 2 == 0:
        return 'Kernel size must be an odd number.'
    elif ksize < 3 or ksize > 200:
        return 'The suggested kernel size is in range 3 to 199.'

def color_change(bw_sketch_gray,color):
    bw_sketch_rgb = cv2.cvtColor(bw_sketch_gray, cv2.COLOR_GRAY2RGB)
    color_dict = {'red':[255,0,0], 'orange':[255,128,0],'yellow':[255,255,0],'green':[0,255,0],'blue':[0,0,255],'purple':[127,0,255]}
    for i in range (bw_sketch_rgb.shape[0]):
        for j in range (bw_sketch_rgb.shape[1]):
            if (bw_sketch_rgb[i,j] != [255,255,255]).any():
                #put_text(color_dict[color])
                bw_sketch_rgb[i,j] = color_dict[color]
                
    return bw_sketch_rgb

def sketch():
    remove(scope="scope_sketch")
    remove(scope="scope_filter_sepia")
    remove(scope="scope_filter_lighting")
    remove(scope = "scope_filter_clarendon")
    remove(scope = "scope_cartoon_comics")
    remove(scope = "scope_cartoon_twilight")
    remove(scope = "scope_cartoon_classic")
    with use_scope("scope_sketch"):
        data = input_group("SKETCH",[
        file_upload("Select an image:", accept="image/*", name="image", required=True),
        select("Choose a sketch color: ", ['red', 'orange','yellow','green','blue','purple', 'black'], name="color", required=True),
        input('Adjust kernel size / blurriness (n x n): ', name='ksize', type=NUMBER,  min = 3, max=199, step=2, validate=check_kernel_odd,  placeholder= "Type an odd number kernel size between 3 to 199", required=True)
        ])

        
        put_processbar('bar')
        for i in range(1, 11):
            set_processbar('bar', i / 10)
            time.sleep(0.1)
        img = data['image']['content'] #bytes
        #img = cv2.imread("photo.jpg")
        #put_image(img)
        img = np.frombuffer(img, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Invert Image
        invert_img = cv2.bitwise_not(grey_img)
        #invert_img=255-grey_img

        # Blur image
        blur_img = cv2.GaussianBlur(invert_img, (data['ksize'],data['ksize']),0)

        # Invert Blurred Image
        invblur_img = cv2.bitwise_not(blur_img)
        #invblur_img=255-blur_img

        # Sketch Image
        sketch_img = cv2.divide(grey_img,invblur_img, scale=256.0)
        if data['color'] != 'black':
            sketch_img = cv2.cvtColor(color_change(sketch_img,data['color']), cv2.COLOR_BGR2RGB)
        is_success, im_buf_arr = cv2.imencode(".jpg", sketch_img)
        byte_im = im_buf_arr.tobytes()
        put_markdown('## **Result**')
        put_row([put_text("Before: "), None, put_text("After: ")])
        put_row([put_image(data['image']['content']), None, put_image(byte_im)])
        data['image']['content'] = byte_im
        put_file(label="Download",name='sketch_'+ data['image']['filename'], content=data['image']['content']).onclick(lambda: toast('Your image is downloaded.'))
        put_button("Try Again", onclick=sketch, color='primary', outline=True)
        put_html('<hr>')
        put_markdown("**If you wish to proceed to pixelation, click the next button.**")
        put_button("Next", onclick=watercolor, color='primary', outline=True)
    
if __name__ == '__main__':
    pywebio.start_server(start, port=80)