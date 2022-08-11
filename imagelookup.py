import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from PIL import Image

##Chakrabati  DataSet##
#dirName = "/Users/IZHov/Downloads/archive/brain_tumor_dataset/no"
#dirName = "/Users/IZHov/Downloads/archive/brain_tumor_dataset/yes"

##BTD 2020 DataSet##
#dirName = "/Users/IZHov/Downloads/DS2/no"
dirName = "/Users/IZHov/Downloads/DS2/yes"
count0 = 0
countim = 0 
imglarger = 0 
imgsmaller = 0

for filename in os.listdir(dirName):
    countim = countim +1
    fn = os.path.join(dirName, filename)
    im = Image.open(fn)
    pix = im.load()
    #Get the width and height of image
    width, height = im.size
    if(width> 224 or height> 224): 
        imglarger = imglarger +1
   
    if(width< 224 or height< 224): 
        imgsmaller = imgsmaller +1
    
    # Get the RGBA Value of the a pixel of an image
    rgb_im = im.convert('RGB')
    r, g, b = rgb_im.getpixel((1, 1))
    #print(r, g, b) 
    if(r == 0 and g==0 and b==0): 
        count0 = count0 +1

print("larger: ", imglarger)   
print("smaller: ", imgsmaller)  
print("color: ", countim, count0)  

