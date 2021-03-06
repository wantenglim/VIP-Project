import pywebio
from pywebio.input import *
from pywebio.output import *
from pywebio import start_server
from pywebio.exceptions import SessionClosedException
import sys
import asyncio
import cv2
import random
from PIL import Image
from PIL import ImageFile
import io
from io import BytesIO
import os
import math
import numpy as np
from numpy import linalg as LA
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import time
import json
import skimage
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from rembg.bg import remove
from sklearn.cluster import KMeans
from imgaug import augmenters as iaa
from scipy import ndimage
import warnings
warnings.filterwarnings("ignore")

ImageFile.LOAD_TRUNCATED_IMAGES = True
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

path = "./assets"

def start():
    pywebio.session.set_env(title='Pixtono')
    #remove(scope="start_app")
    with use_scope("start_app", clear=True):
        #put_markdown('# **Multifunctional Face Image Processing**')
        startpage()
        img = file_upload("Upload an image:", accept="image/*", required=True)
        lj_map = open(path+'/lj_map.png', 'rb').read()
        #step 1: enhance image 
        # Image equalization
        image_enhance(img)
        #step 2: image filter 
        filter_image(img,lj_map)
        #step 3: background remove 
        img_background(img)
        #step 4: style tranformation
        main_function(img)
        #step5: pixelization
        pixelate(img)
        endpage()

def startpage():
    logo = open(path+'/logo-beige.png', 'rb').read()  
    put_row([None, None, put_image(logo, width='110px'), put_markdown('# Pixtono'), None, None],size=50)
    put_markdown('### A Non-Photorealistic Rendering and Pixelation of Face Images Application 📸')
    put_markdown('#### TDS3651 Visual Information Processing - Project 🎓')
    put_markdown('##### Group 5: Lee Min Xuan (1181302793) Lim Wan Teng (1181100769) Tan Jia Qi (1191301879) Vickey Tan (1181101852) 👩‍💻')

def endpage():
    put_text('Thank you for using this application. Kindly fill up the feedback form to help us improve better. 😃')
    put_link('Feedback Form', url='https://forms.gle/iodfkvqG8DmBUzWA9', new_window=True)

def progress_bar():
    put_processbar('bar',auto_close=True)
    for i in range(1, 11):
        set_processbar('bar', i / 10)
        time.sleep(0.1)
        
def main_function(img):
    main_function = radio("Choose a style effect",options = ['Cartoonization','Oil Paint','Pencil Sketch','Watercolour','No Effect'], required=True)
    # give user stack filter once more
    if main_function == "Cartoonization":
        cartoonization(img)
    elif main_function == "Oil Paint":
        oil_paint(img)
    elif main_function == "Pencil Sketch":
        sketch(img)
    elif main_function == "Watercolour":
        watercolor(img)
    elif main_function == "No effect":
        return img
# ---------------------------------ENHANCEMENT--------------------------------
#rotate effect
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
  return result

#adjust bright and contrast
def apply_brightness_contrast(image, brightness = 0, contrast = 0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        buf = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    else:
        buf = image.copy()
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf

def image_enhance(img):
    with use_scope("scope_enhance", clear=True):
        data = input_group("Image Enhancement",[
        select("Image Equalization: ", ["No", "Yes"], name='CLAHE', required=True),
        select("Denoise: ", ["No", "Yes"], name='denoise', required=True),
        input('Rotate Image: ', name='rotate', type=NUMBER, min = -90, max=90, placeholder= "0", value=0),
        input('Adjust brightness: ', name='brightness', type=NUMBER, min = -127, max=127, placeholder= "0", value=0),
        input('Adjust contrast: ', name='contrast', type=NUMBER, min = -127, max=127, placeholder= "0", value=0)
        ])
        progress_bar()

        result = img['content']
        result = np.frombuffer(result, np.uint8)
        result = cv2.imdecode(result, cv2.IMREAD_COLOR)
        # 1. image CLAHE (equalization)
        if data['CLAHE'] == 'Yes':
            src_lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(src_lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            result = cv2.merge((cl,a,b))
            result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
            result = np.array(result, dtype=np.uint8)
        # 2. image denoising
        if data['denoise'] == 'Yes':
            result = cv2.fastNlMeansDenoisingColored(result, None, 10, 10, 7, 21)
        # 3. rotate image
        result = rotate_image(result, data['rotate'])
        # 4. adjust brightness and contrast
        result = apply_brightness_contrast(result, data['brightness'], data['contrast'])

        if data['CLAHE'] == 'Yes' or data['denoise'] == 'Yes' or data['rotate'] != 0 or data['brightness'] != 0 or data['contrast'] != 0:
            is_success, im_buf_arr = cv2.imencode(".png", result)
            byte_im = im_buf_arr.tobytes()
            put_markdown('## **Image Enhancement Result**')
            put_row([put_text("Before: "), None, put_text("After: ")])
            put_row([put_image(img['content']), None, put_image(byte_im)])
            img['content'] = byte_im
            put_file(label="Download",name='enhance_'+ img['filename'], content=img['content']).onclick(lambda: toast('Your image is downloaded.'))
            put_button("Retry", onclick=start, color='primary', outline=True)
            put_html('<hr>')

# ---------------------------------FILTERING--------------------------------
def filter_image(img,lj_map):
    filter_preview = open(path+'/filter-preview.png', 'rb').read()  
    popup('Filter Preview', [put_image(filter_preview),put_buttons(['close'], onclick=lambda _: close_popup())])
    operation = radio("Image Filter",options = ['Filter Sepia','Filter Lighting','Filter Clarendon','No Filter'], required=True)
    if operation == "Filter Sepia":
        filter_sepia(img)
    elif operation == "Filter Lighting":
        filter_lighting(img)
    elif operation == "Filter Clarendon":
        filter_clarendon(img,lj_map)
    elif operation == "No Filter":
        return img
    
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
    with use_scope("scope_filter_sepia", clear=True):
        progress_bar()
        result = img['content']
        result = np.frombuffer(result, np.uint8)
        result = cv2.imdecode(result, cv2.IMREAD_COLOR)
        result = sepia(result)
        is_success, im_buf_arr = cv2.imencode(".png", result)
        byte_im = im_buf_arr.tobytes()
        put_markdown('## **Sepia Filter Result**')
        put_row([put_text("Before: "), None, put_text("After: ")])
        put_row([put_image(img['content']), None, put_image(byte_im)])
        img['content'] = byte_im
        put_file(label="Download",name='filter_'+ img['filename'], content=img['content']).onclick(lambda: toast('Your image is downloaded.'))
        put_button("Retry", onclick=start, color='primary', outline=True)
        put_html('<hr>')
        
def filter_lighting(img):
    with use_scope("scope_filter_lighting", clear=True):
        progress_bar()
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
        is_success, im_buf_arr = cv2.imencode(".png", result)
        byte_im = im_buf_arr.tobytes()
        put_markdown('## **Lighting Filter Result**')
        put_row([put_text("Before: "), None, put_text("After: ")])
        put_row([put_image(img['content']), None, put_image(byte_im)])
        img['content'] = byte_im
        put_file(label="Download",name='filter_'+ img['filename'], content=img['content']).onclick(lambda: toast('Your image is downloaded.'))
        put_button("Retry", onclick=start, color='primary', outline=True)

def filter_clarendon(img,lj_map):
    with use_scope("scope_filter_clarendon", clear=True):
        progress_bar()
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
        is_success, im_buf_arr = cv2.imencode(".png", result)
        byte_im = im_buf_arr.tobytes()
        put_markdown('## **Clarendon Filter Result**')
        put_row([put_text("Before: "), None, put_text("After: ")])
        put_row([put_image(img['content']), None, put_image(byte_im)])
        img['content'] = byte_im
        put_file(label="Download",name='filter_'+ img['filename'], content=img['content']).onclick(lambda: toast('Your image is downloaded.'))
        put_button("Retry", onclick=start, color='primary', outline=True)

# ---------------------------------BACKGROUND--------------------------------
def img_background(img):
    background_preview = open(path+'/background-preview.png', 'rb').read()  
    popup('Background Preview', [put_image(background_preview),put_buttons(['close'], onclick=lambda _: close_popup())])
    background_choice = radio("Background editing options",options = ['Transparent Background','Solid Color Background',
                                                    'Customize & Patterned Background', 'No Change'], required=True)
    if background_choice == "Transparent Background" :
        bg_remover(img)
    elif background_choice == "Solid Color Background":
        bg_solid(img)
    elif background_choice == "Customize & Patterned Background":
        bg_cuspat(img)
    elif background_choice == "No Change":
        return img

def bg_remover(img):
    with use_scope("scope_bg_remover", clear=True):
        progress_bar()
        image = img['content']
        result = remove(image)
        img_transpB = Image.open(io.BytesIO(result)).convert("RGBA")
        img_transpB.save('img_transpB.png', format='PNG')
        result = open('img_transpB.png','rb').read()
        put_markdown('## **Background Removing Result:**')
        put_row([put_text("Before: "), None, put_text("After: ")])
        put_row([put_image(img['content']), None, put_image(result)])
        img['content'] = result
        put_file(label="Download",name='transparentbg_'+ img['filename'], content=result).onclick(lambda: toast('Your image is downloaded.'))
        os.remove("img_transpB.png")
        put_button("Retry", onclick=start, color='primary', outline=True)

def bg_solid(img):
    with use_scope("scope_bg_solid", clear=True):
        html_colors = ['Aqua','Beige','Black','Brown','Coral',
                    'DarkGrey','Fuchsia','Green','HotPink','Indigo',
                    'LightBlue','Lime','MediumBlue','Orange','Pink',
                    'RebeccaPurple','Red','Teal','White','Yellow']    
        progress_bar()
        image = img['content']
        result = remove(image)
        img_transpB = Image.open(io.BytesIO(result)).convert("RGBA")
        img_transpB.save('img_transpB.png', format='PNG')
        foreground = Image.open('img_transpB.png').convert("RGBA")
        color_choice = input_group("Background Colour Picker",[select("Choose a background colour: ", name='colour', options=html_colors, required=True)])
        img_solidB = Image.new("RGBA", img_transpB.size, color_choice['colour']) 
        img_solidB.paste(foreground, mask=foreground)
        img_solidB.convert("RGB")
        img_solidB.save('img_solidB.png', format='PNG')
        result = open('img_solidB.png','rb').read()
        put_markdown('## **Solid Color Background Result:**')
        put_row([put_text("Before: "), None, put_text("After: ")])
        put_row([put_image(img['content']), None, put_image(result)])
        img['content'] = result
        put_file(label="Download",name='solidbg_'+ img['filename'], content=result).onclick(lambda: toast('Your image is downloaded.'))
        os.remove("img_transpB.png")
        os.remove("img_solidB.png")
        put_button("Retry", onclick=start, color='primary', outline=True)

def bg_cuspat(img):
    with use_scope("scope_bg_cuspat", clear=True):
        background_list = ['Customize','Horizontal-1','Horizontal-2','Horizontal-3','Horizontal-4','Horizontal-5',
                            'Square-1','Square-2','Square-3','Square-4','Square-5',
                            'Vertical-1','Vertical-2','Vertical-3','Vertical-4','Vertical-5']
        progress_bar()
        image = img['content']
        result = remove(image)
        img_transpB = Image.open(io.BytesIO(result)).convert("RGBA")
        img_transpB.save('img_transpB.png', format='PNG')
        foreground = Image.open('img_transpB.png').convert("RGBA")
        cuspat_choice = input_group("Background Changer",[select("Choose/Customize a background pattern: ", name='xuanze', options=background_list, required=True)])
        bg_path = path + '/background'
        if cuspat_choice['xuanze'] != 'Customize':
            for image_name in os.listdir(bg_path):
                input_path = os.path.join(bg_path, image_name)
                if image_name == cuspat_choice['xuanze']+".png":
                    background = Image.open(input_path).convert("RGBA")
        else:
            bgimg = file_upload("Select your background image:", accept="image/*")         
            background = bgimg['content']
            background = Image.open(io.BytesIO(background)).convert("RGBA")
            #background = background.convert("RGBA")
        
        width = (background.width - foreground.width) // 2
        height = (background.height - foreground.height) // 2
        background.paste(foreground, (width, height), foreground)
        background.convert("RGB")
        background.save('img_cuspatB.png', format='PNG')
        result = open('img_cuspatB.png','rb').read()
        put_markdown('## **Background Changing Result:**')
        put_row([put_text("Before: "), None, put_text("After: ")])
        put_row([put_image(img['content']), None, put_image(result)])
        img['content'] = result
        put_file(label="Download",name='newbg_'+ img['filename'], content=result).onclick(lambda: toast('Your image is downloaded.'))
        os.remove("img_transpB.png")
        os.remove("img_cuspatB.png")
        put_button("Retry", onclick=start, color='primary', outline=True)

# ---------------------------------CARTOON--------------------------------
def cartoonization(img):
    cartoon_preview = open(path+'/cartoon-preview.png', 'rb').read()  
    popup('Cartoonization Preview', [put_image(cartoon_preview),put_buttons(['close'], onclick=lambda _: close_popup())])
    cartoon_choice = radio("Cartoonization",options = ['Comics','Twilight','Classic'], required=True)
    if cartoon_choice == "Comics" :
        cartoon_comics(img)
    elif cartoon_choice == "Twilight":
        cartoon_twilight(img)
    elif cartoon_choice == "Classic":
        cartoon_classic(img)

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
    with use_scope("scope_cartoon_comics", clear=True):
        progress_bar()
        image = img['content']
        image = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        dpi = 100
        height, width, depth = image.shape
        figsize = width / float(dpi), height / float(dpi)
        plt.figure(figsize= figsize )
        plt.contourf(np.flipud(image[:,:,0]),levels=4,cmap='inferno')
        plt.axis('off')
        plt.savefig('cartooned.png',bbox_inches='tight')
        plt.close()
        result = open('cartooned.png', 'rb').read()
        put_markdown('## **Comics Cartoonization Result**')
        put_row([put_text("Before: "), None, put_text("After: ")])
        put_row([put_image(img['content']), None, put_image(result)])
        img['content'] = result
        put_file(label="Download",name='comics_'+ img['filename'], content=img['content']).onclick(lambda: toast('Your image is downloaded.'))
        os.remove("cartooned.png") 
        put_button("Retry", onclick=start, color='primary', outline=True)
        
def cartoon_twilight(img):
    with use_scope("scope_cartoon_twilight", clear=True):
        progress_bar()
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
        is_success, im_buf_arr = cv2.imencode(".png", result)
        byte_im = im_buf_arr.tobytes()
        put_markdown('## **Twilight Cartoonization Result**')
        put_row([put_text("Before: "), None, put_text("After: ")])
        put_row([put_image(img['content']), None, put_image(byte_im)])
        img['content'] = byte_im
        put_file(label="Download",name='twilight_'+ img['filename'], content=img['content']).onclick(lambda: toast('Your image is downloaded.'))
        put_button("Retry", onclick=start, color='primary', outline=True)
        
def cartoon_classic(img):
    with use_scope("scope_cartoon_classic", clear=True):
        progress_bar()
        image = img['content']
        image = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        coloured = ColourQuantization(image)
        result = Countours(coloured)
        is_success, im_buf_arr = cv2.imencode(".png", result)
        byte_im = im_buf_arr.tobytes()
        put_markdown('## **Classic Cartoonization Result**')
        put_row([put_text("Before: "), None, put_text("After: ")])
        put_row([put_image(img['content']), None, put_image(byte_im)])
        img['content'] = byte_im
        put_file(label="Download",name='classic_'+ img['filename'], content=img['content']).onclick(lambda: toast('Your image is downloaded.'))
        put_button("Retry", onclick=start, color='primary', outline=True)
        
# ---------------------------------OIL PAINT--------------------------------
def prewitt(img):
    img_gaussian = cv2.GaussianBlur(img,(3,3),0)
    kernelx = np.array( [[1, 1, 1],[0, 0, 0],[-1, -1, -1]] )
    kernely = np.array( [[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]] )
    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
    return img_prewittx // 15.36, img_prewitty // 15.36

def roberts(img):
    roberts_cross_v = np.array( [[ 0, 0, 0 ],
                                 [ 0, 1, 0 ],
                                 [ 0, 0,-1 ]] )
    roberts_cross_h = np.array( [[ 0, 0, 0 ],
                                 [ 0, 0, 1 ],
                                 [ 0,-1, 0 ]] )
    vertical = ndimage.convolve( img, roberts_cross_v )
    horizontal = ndimage.convolve( img, roberts_cross_h )
    return vertical // 50.0, horizontal // 50.0

# Different Edge Operator 
def get_gradient(img_o, ksize, gtype):
    if gtype == 'scharr':
        X = cv2.Scharr(img_o, cv2.CV_32F, 1, 0) / 50.0
        Y = cv2.Scharr(img_o, cv2.CV_32F, 0, 1) / 50.0
    elif gtype == 'prewitt':
        X, Y = prewitt(img_o)
    elif gtype == 'sobel':
        X = cv2.Sobel(img_o,cv2.CV_32F,1,0,ksize=5)  / 50.0
        Y = cv2.Sobel(img_o,cv2.CV_32F,0,1,ksize=5)  / 50.0
    elif gtype == 'roberts':
        X, Y = roberts(img_o)
    else:
        print('Not suppported type!')
        exit()

    # Blur the Gradient to smooth the edge
    X = cv2.GaussianBlur(X, ksize, 0)
    Y = cv2.GaussianBlur(Y, ksize, 0)
    return X, Y

def draw_order(h, w, scale):
    order = []
    for i in range(0, h, scale):
        for j in range(0, w, scale):
            y = random.randint(-scale // 2, scale // 2) + i
            x = random.randint(-scale // 2, scale // 2) + j
            order.append((y % h, x % w))
    return order

def oil_paint(img):
    oilpaint_preview = open(path+'/oilpaint-preview.png', 'rb').read()  
    popup('Oil Painting Preview', [put_image(oilpaint_preview),put_buttons(['close'], onclick=lambda _: close_popup())])
    with use_scope("scope_oilpaint", clear=True):
        data = input_group("Oil Paint",[
        input('Adjust brush: ', name='brush', type=NUMBER, min = 1, max=10, placeholder= "1", value=1, help_text="Large size of brush -> Thicker the stroke"),
        input('Adjust color: ', name='color', type=NUMBER, min = 1, max=30, placeholder= "1", value=1, help_text="More no. of colors -> More detailed stroke color"),
        select("Choose oil paint style: ", ['roberts','scharr','prewitt', 'sobel'], name="oil_style", required=True),
        ])

        progress_bar()
        
        result = img['content']
        result = np.frombuffer(result, np.uint8)
        result = cv2.imdecode(result, cv2.IMREAD_COLOR)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        r = 2 * int(result.shape[0] / 50) + 1
        Gx, Gy = get_gradient(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), (r, r), data["oil_style"])
        Gh = np.sqrt(np.sqrt(np.square(Gx) + np.square(Gy)))    # Length of the ellipse
        Ga = (np.arctan2(Gy, Gx) / np.pi) * 180 + 90            # Angle of the ellipse
        
        canvas = cv2.medianBlur(result, 11)    # Make the image artlistic
        order = draw_order(result.shape[0], result.shape[1], scale=data["brush"]*2)
        oil_img = []
        colors = np.array(result, dtype=np.float)
        for i, (y, x) in enumerate(order):
            length = int(round(data["brush"] + data["brush"] * Gh[y, x]))
            # Select color
            if data["color"] != 0: color = np.array([round(colors[y,x][0]/data["color"])*data["color"]+random.randint(-5,5), 
                round(colors[y,x][1]/data["color"])*data["color"]+random.randint(-5,5), round(colors[y,x][2]/data["color"])*data["color"]+random.randint(-5,5)], dtype=np.float)
            else: color = colors[y,x]
            cv2.ellipse(canvas, (x, y), (length, data["brush"]), Ga[y, x], 0, 360, color, -1, cv2.LINE_AA)
        oil_img.append(canvas)
        
        result = cv2.cvtColor(oil_img[0], cv2.COLOR_BGR2RGB)

        is_success, im_buf_arr = cv2.imencode(".png", result)
        byte_im = im_buf_arr.tobytes()

        put_markdown('## **Oil Painting Result**')
        put_row([put_text("Before: "), None, put_text("After: ")])
        put_row([put_image(img['content']), None, put_image(byte_im)])        
        img['content'] = byte_im
        put_file(label="Download",name='oilpaint_'+ img['filename'], content=img['content']).onclick(lambda: toast('Your image is downloaded.'))
        put_button("Retry", onclick=start, color='primary', outline=True)

# ---------------------------------WATERCOLOR--------------------------------

def check_sigma_s(sigma_s):
    if sigma_s < 0 or sigma_s > 200:
        return 'The range of sigma s should between 0 to 200.'


def check_sigma_r(sigma_r):
    if sigma_r < 0 or sigma_r > 1:
        return 'The range of sigma r should between 0 to 1.'
               
def watercolor(img):
    watercolor_preview = open(path+'/watercolor-preview.png', 'rb').read()  
    popup('Watercolor Preview', [put_image(watercolor_preview),put_buttons(['close'], onclick=lambda _: close_popup())])
    with use_scope("scope_watercolor", clear=True):
        data = input_group("Watercolor",[
        input("Adjust sigma s: ", name='sigma_s', type=FLOAT, validate=check_sigma_s, placeholder= "20", help_text="Control smoothening", required=True),
        input("Adjust sigma r: ", name='sigma_r', type=FLOAT, validate=check_sigma_s, placeholder= "0.4", help_text="Control smoothening", required=True),
        select("Choose a color tone: ", ['blue','brown','gray', 'green', 'pink', 'purple','yellow'], name="color_tone", required=True),
        input("Adjust segment scale: ", name='scale', type=FLOAT, placeholder= "40", help_text="Higher means larger watercolor-like segmentation", required=True)
        ])
        
        progress_bar()
        
        result = img['content']
        result = np.frombuffer(result, np.uint8)
        result = cv2.imdecode(result, cv2.IMREAD_COLOR)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        # 1. Edge Preserving Filter
        # sigma s, sigma r
        result = cv2.edgePreservingFilter(result, flags=1, sigma_s=data['sigma_s'], sigma_r=data['sigma_r'],)
        
        # 2. Color Adjustment
        # color tone image
        watercolor_path = path + '/watercolor tone'
        for image_name in os.listdir(watercolor_path):
            input_path = os.path.join(watercolor_path, image_name)
            if image_name== data['color_tone']+".jpg":
                src_img = open(input_path, 'rb').read()    
        
        src_img = np.frombuffer(src_img, np.uint8)
        src_img = cv2.imdecode(src_img, cv2.IMREAD_COLOR)
        src_lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        src_mean, src_stddev = cv2.meanStdDev(src_lab)

        result_lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB).astype(np.float32)
        result_mean, result_stddev = cv2.meanStdDev(result_lab)

        result_lab -= result_mean.reshape((1, 1, 3))
        result_lab = np.multiply(result_lab, np.divide(src_stddev.flatten(), result_stddev.flatten()).reshape((1, 1, 3)))
        result_lab += src_mean.reshape((1, 1, 3))
        result_lab = np.clip(result_lab, 0, 255)
        result = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
 
        # 3. Mean Shift Filtering
        '''
        sp – The spatial window radius.
        sr – The color window radius.
        maxLevel – Maximum level of the pyramid for the segmentation.
        '''
        result = cv2.pyrMeanShiftFiltering(result, sp=30, sr=30, maxLevel=3)

        # 4. Image Segmentation and Fill Mean Value
        '''
        scale(float): Free parameter. Higher means larger clusters.
        sigma(float): Width (standard deviation) of Gaussian kernel used in preprocessing.
        min_size(int): Minimum component size. Enforced using postprocessing.
        '''
        segments = felzenszwalb(result, scale=data['scale'], sigma=0.4, min_size=10)

        for i in range(np.max(segments)):
            logical_segment = segments == i
            segment_img = result[logical_segment]
            result[logical_segment] = np.mean(segment_img, axis=0)
        
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


        is_success, im_buf_arr = cv2.imencode(".png", result)
        byte_im = im_buf_arr.tobytes()

        put_markdown('## **Watercolour Painting Result**')
        put_row([put_text("Before: "), None, put_text("After: ")])
        put_row([put_image(img['content']), None, put_image(byte_im)])        
        img['content'] = byte_im
        put_file(label="Download",name='watercolor_'+ img['filename'], content=img['content']).onclick(lambda: toast('Your image is downloaded.'))
        #put_html('<hr>')
        #put_markdown("**If you wish to proceed to pixelation, click the next button.**")
        #put_button("Next", onclick=lambda: toast("Link to pixelation"), color='primary', outline=True)
        put_button("Retry", onclick=start, color='primary', outline=True)

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
                bw_sketch_rgb[i,j] = color_dict[color]
                
    return bw_sketch_rgb

def sketch(img):
    sketch_preview = open(path+'/sketch-preview.png', 'rb').read()  
    popup('Sketch Preview', [put_image(sketch_preview),put_buttons(['close'], onclick=lambda _: close_popup())])
    with use_scope("scope_sketch", clear=True):
        data = input_group("Sketch",[
        select("Choose a sketch color: ", ['red', 'orange','yellow','green','blue','purple', 'black'], name="color", required=True),
        input('Adjust kernel size: ', name='ksize', type=NUMBER,  min = 3, max=199, step=2, validate=check_kernel_odd,  placeholder= "111", help_text="Control thickness", required=True)
        ])

        progress_bar()

        result = img['content'] #bytes
        result = np.frombuffer(result, np.uint8)
        result = cv2.imdecode(result, cv2.IMREAD_COLOR)
        grey_img = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # Invert Image
        invert_img = cv2.bitwise_not(grey_img)

        # Blur image
        blur_img = cv2.GaussianBlur(invert_img, (data['ksize'],data['ksize']),0)

        # Invert Blurred Image
        invblur_img = cv2.bitwise_not(blur_img)

        # Sketch Image
        sketch = cv2.divide(grey_img,invblur_img, scale=256.0)
        sketch_img = cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB)
        if data['color'] != 'black':
            sketch_img = cv2.cvtColor(color_change(sketch,data['color']), cv2.COLOR_BGR2RGB)
        is_success, im_buf_arr = cv2.imencode(".png", sketch_img)
        byte_im = im_buf_arr.tobytes()

        put_markdown('## **Pencil Sketch Result**')
        put_row([put_text("Before: "), None, put_text("After: ")])
        put_row([put_image(img['content']), None, put_image(byte_im)])
        img['content'] = byte_im
        put_file(label="Download",name='sketch_'+ img['filename'], content=img['content']).onclick(lambda: toast('Your image is downloaded.'))
        #put_html('<hr>')
        #put_markdown("**If you wish to proceed to pixelation, click the next button.**")
        #put_button("Next", onclick=lambda: toast("Link to pixelation"), color='primary', outline=True)
        put_button("Retry", onclick=start, color='primary', outline=True)

# ---------------------------------PIXELIZATION--------------------------------

def colorClustering(idx, img, k):
    clusterValues = []
    for _ in range(0, k):
        clusterValues.append([])
    
    for r in range(0, idx.shape[0]):
        for c in range(0, idx.shape[1]):
            clusterValues[idx[r][c]].append(img[r][c])

    imgC = np.copy(img)

    clusterAverages = []
    for i in range(0, k):
        clusterAverages.append(np.average(clusterValues[i], axis=0))
    
    for r in range(0, idx.shape[0]):
        for c in range(0, idx.shape[1]):
            imgC[r][c] = clusterAverages[idx[r][c]]
            
    return imgC

def segmentImgClrRGB(img, k):
    
    imgC = np.copy(img)
    
    h = img.shape[0]
    w = img.shape[1]
    
    imgC.shape = (img.shape[0] * img.shape[1], 3)
    
    #Run k-means on the vectorized responses X to get a vector of labels (the clusters)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(imgC).labels_
    
    #Reshape the label results of k-means so that it has the same size as the input image
    #Return the label image which we call idx
    kmeans.shape = (h, w)

    return kmeans

def kMeansImage(image, k):
    idx = segmentImgClrRGB(image, k)
    return colorClustering(idx, image, k)

def pixelate(img):
    pixelate_preview = open(path+'/pixelization-preview.png', 'rb').read()  
    popup('Pixelization Preview', [put_image(pixelate_preview),put_buttons(['close'], onclick=lambda _: close_popup())])
    with use_scope("scope_pixelate", clear=True):
        progress_bar()
        def is_valid(data):
            if data <= 0:
                return 'Value cannot be negative!'
            elif data < 2:
                return 'Value must be bigger than 1!'
        
        pixel = input_group("Pixel Size",[
        input('Adjust pixel size: ', name='pixelsize', type=NUMBER,  min = 2, max=128, validate=is_valid,  placeholder= "8", 
              help_text="Lower pixel size -> Clearer & more detailed image", required=True),
        input('Adjust number of colors: ', name='pixelcolor', type=NUMBER,  min = 2, max=128, validate=is_valid,  placeholder= "6", 
              help_text="More no. of colors -> Clearer & more detailed image", required=True)
        ])
        
        pixel_size = pixel['pixelsize']
        num_colors = pixel['pixelcolor']
        image = img['content']
        image = Image.open(io.BytesIO(image))
        image.save('img_ori.png', format='PNG')
        img_pixelated = Image.open('img_ori.png')        
        img_pixelated = img_pixelated.resize((img_pixelated.size[0] // pixel_size, img_pixelated.size[1] // pixel_size),Image.NEAREST)
        img_pixelated = img_pixelated.resize((img_pixelated.size[0] * pixel_size, img_pixelated.size[1] * pixel_size),Image.NEAREST)
        img_pixelated = np.array(img_pixelated)
        img_pixelated = Image.fromarray(img_pixelated)
        img_pixelated.save('img_pixel.png', format='PNG')
        img_pixelated_color = cv2.imread('img_pixel.png') 
        img_pixelated_color = kMeansImage(img_pixelated_color, num_colors) #5 colors user need to choose by themselves
        img_pixelated_color = cv2.cvtColor(img_pixelated_color, cv2.COLOR_BGR2RGB)
        result = Image.fromarray(img_pixelated_color)
        buf = io.BytesIO()
        result.save(buf, format='PNG')
        byte_im = buf.getvalue()

        put_markdown('## **Pixelization Result**')
        put_row([put_text("Before: "), None, put_text("After: ")])
        put_row([put_image(img['content']), None, put_image(result)])
        put_file(label="Download",name='pixel_'+ img['filename'], content=byte_im).onclick(lambda: toast('Your image is downloaded.'))        
        os.remove("img_ori.png")
        os.remove("img_pixel.png")
        put_button("Retry", onclick=start, color='primary', outline=True)

if __name__ == '__main__':
    pywebio.start_server(start, port=80)
