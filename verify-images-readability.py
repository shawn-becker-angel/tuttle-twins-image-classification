import os
import sys
import cv2

# python3 -m pip install --upgrade pip
# python3 -m pip install --upgrade Pillow
from PIL import Image

# pip install keras-utils
from keras.utils import load_img, img_to_array

# pip install opencv-python
from cv2 import imread, resize

SRC_IMAGES_DIR = "../src-images"

#
# Example usage:
#
# activate
# python verify-images-readability.py ../src-images
#

def verify_images_readable(dir):
    '''Verify that all images under dir can be read using cv2 and Image'''
    
    print("verifying readability of all jpg files under", directory)
    
    bad_list=[]
    good_list=[]
    cnt = 0
    good_exts=['jpg'] # make a list of acceptable image file types
    for f in os.listdir(dir) :  # iterate through the directory of all image files
        f_path=os.path.join(dir, f) # path to image files
        ext=f[f.rfind('.')+1:] # get the files extension
        if ext  not in good_exts:
                print(f'file {f_path}  has an invalid extension {ext}')
                bad_list.append(f_path)                    
        else:
            try:
                test_cv2_imread(f_path)
                test_Image_load_image(f_path)
                good_list.append(f_path)
                cnt += 1
                if (cnt % 1000) == 0:
                    print("image cnt:", cnt)
            except Exception as exp:
                print(f'file {f_path} raises {type(exp)} {str(exp)}')
                bad_list.append(f_path)

    return good_list, bad_list

def test_cv2_imread(f_path):
    img=cv2.imread(f_path)
    dim = [100,100]
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

def test_Image_load_image(f_path):
    dim = (230,230)
    img = load_img(f_path, target_size=dim)
    # convert the image pixels to a numpy array
    img = img_to_array(img)
    # reshape data for the model
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

if __name__ == "__main__":
    directory = SRC_IMAGES_DIR
    
    if len(sys.argv) > 1:
        directory = sys.argv[1]

    
    good_list, bad_list = verify_images_readable(dir=directory)

    if len(bad_list) > 0:
        print("bad_list:")
        for f_path in bad_list:
            print(f_path)

