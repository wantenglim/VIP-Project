import pywebio
from pywebio.input import *
from pywebio.output import *
from pywebio import start_server
import os
import time
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
import sys
import asyncio

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
  
# ---------------------------------WATERCOLOR--------------------------------
path = "./assets"

def check_sigma_s(sigma_s):
    if sigma_s < 0 or sigma_s > 200:
        return 'The range of sigma s should between 0 to 200.'


def check_sigma_r(sigma_r):
    if sigma_r < 0 or sigma_r > 1:
        return 'The range of sigma r should between 0 to 1.'
               
'''        
def check_spatial_radius(spatial_radius):
    if spatial_radius < 0 or spatial_radius > 200:
        return 'The range of spatial window radius should between 0 to 200.'

def check_color_radius(color_radius):
    if color_radius < 0 or color_radius > 200:
        return 'The range of color window radius should between 0 to 200.'

def check_max(max):
    if max < 0 or max > 200:
        return 'The range of maximum level of pyramid should between 0 to 200.'
          
def check_scale(scale):
    if scale < 0 or scale > 200:
        return 'The range of scale should between 0 to 200.'

def check_sigma_seg(sigma_seg):
    if sigma_seg < 0 or sigma_seg > 200:
        return 'The range of sigma_seg should between 0 to 200.'

def check_min(min):
    if min < 0 or min > 200:
        return 'The range of min-size should between 0 to 200.'
''' 
def watercolor():
    remove(scope="scope_sketch")
    remove(scope="scope_watercolor")
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
        
        #file_upload("Select an c image:", accept="image/*", name="colorimg"),
        #file_upload("Select an t image:", accept="image/*", name="textureimg"),
        
        '''
        input("Adjust spatial window radius: ", name='spatial_radius', type=FLOAT, validate=check_spatial_radius, required=True),
        input("Adjust color window radius: ", name='color_radius', type=FLOAT, validate=check_color_radius, required=True),
        input("Adjust maximum level of pyramid: ", name='max', type=NUMBER, validate=check_max, required=True),
        input("Adjust segment scale: ", name='scale', type=FLOAT, validate=check_scale, required=True),
        input("Adjust segment sigma: ", name='sigma_seg', type=FLOAT, validate=check_sigma_seg, required=True),
        input("Adjust segment min-size: ", name='min', type=NUMBER, validate=check_min, required=True),
        '''
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
        #result_mean, result_stddev = cv2.meanStdDev(result)
        #put_text(result_mean)
        #put_text(result_stddev)
        
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
        #result_mean, result_stddev = cv2.meanStdDev(result_lab)
        #put_text(result_mean)
        #put_text(result_stddev)
        result = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        #result_mean, result_stddev = cv2.meanStdDev(result)
        #put_text(result_mean)
        #put_text(result_stddev)
        
        
        # 3. Mean Shift Filtering
        # spatial window radius, color window radius and the maximum level of the pyramid
        result = cv2.pyrMeanShiftFiltering(result, sp=data['spatial_radius'], sr=data['color_radius'], maxLevel=data['max'])


        
        # 4. Image Segmentation and Fill Mean Value
        
        #scale-float
        #Free parameter. Higher means larger clusters.

        #sigma-float
        #Width (standard deviation) of Gaussian kernel used in preprocessing.

        #min_size-int
        #Minimum component size. Enforced using postprocessing.
       
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
        #result_mean, result_stddev = cv2.meanStdDev(texture)
        #put_text(result_mean)
        #put_text(result_stddev)
        #texture = cv2.cvtColor(texture, cv2.COLOR_GRAY2RGB)
        #texture = cv2.imread('./assets/texture.jpg', cv2.IMREAD_GRAYSCALE)


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
       
        #img1 = file_upload("Select a image:", accept="image/*") #dict
        #img1 = np.array(cv2.imread(img))
        #put_image(img1)
        #img = open('C:/VIP APP/bg-back.png', 'rb').read()  
        #put_image(img, width='50px')

        #put_image('https://www.python.org/static/img/python-logo.png')
        
        #put_text("123")
        #put_text("456")
        
        # Convert to Grey Image
        #img['content'] = '...'
        #img.pop('dataurl', None)
        #put_code(json.dumps(img, indent=4, ensure_ascii=False).replace('"..."', '...'), 'json')
        img = data['image']['content'] #bytes
        #img = open(img['filename'], 'rb').read()   # bytes
        #img=cv2.imread("photo.jpg")
        #put_image(img)

        
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
        #put_image(byte_im)
        #put_text(byte_im)
        #put_markdown('# **divide**')
        #put_text(img1['content']) 
        #put_file(byte_im,"download")
        data['image']['content'] = byte_im
        put_file(label="Download",name='sketch_'+ data['image']['filename'], content=data['image']['content']).onclick(lambda: toast('Your image is downloaded.'))
        put_button("Try Again", onclick=sketch, color='primary', outline=True)
        put_html('<hr>')
        put_markdown("**If you wish to proceed to pixelation, click the next button.**")
        put_button("Next", onclick=watercolor, color='primary', outline=True)
        
        
        '''
        height = input("Input your height(cm)：", type=FLOAT)
        weight = input("Input your weight(kg)：", type=FLOAT)

        BMI = weight / (height / 100) ** 2

        top_status = [(16, 'Severely underweight'), (18.5, 'Underweight'),
                      (25, 'Normal'), (30, 'Overweight'),
                      (35, 'Moderately obese'), (float('inf'), 'Severely obese')]

        for top, status in top_status:
            if BMI <= top:
                put_text('Your BMI: %.1f. Category: %s' % (BMI, status))
                break
        '''

    
    
if __name__ == '__main__':
    pywebio.start_server(sketch, port=80)