# VIP-Project
## A Multifunctional Face Image Processing Application

We have prepared three different sizes (width × height) of patterned background image to be used, which are square (720 × 720 px), vertical/portrait (720 × 1280 px), horizontal/landscape (1280 × 720 px). 

If you are intended to change the background of your image:
- To pure solid color background: you can keep your original image size.
- To your own background: you should make sure the background is bigger than your image.
- To our patterned background: you are required to make sure your image is smaller or equal to the size of the output desired in order to obtain the best results. You may resize your image before using the application.


## Application features
1. Image enhancement
2. Image filtering
3. Style conversion - cartoon , oil painting , pencil sketching , water color
4. Pixelation

## Get Started
### Conda (Recommended)

```bash
conda env create -f conda-env.yml
conda activate pixtono
``` 

### PIP 
```bash
pip install -r requirements.txt
``` 

## Running Application
```bash
python main.py
``` 