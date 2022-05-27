import os
import cv2
# pip install opencv-python

from PIL import Image

def verify_images_readable(dir):
    '''Verify that all images under dir can be read using cv2.imread'''
    bad_list=[]
    good_list=[]
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
                test_PIL_load_image(f_path)
                good_list.append(f_path)
            except Exception as exp:
                print(f'file {f_path} raises {type(exp)} {str(exp)}')
                bad_list.append(f_path)

    return good_list, bad_list

def test_cv2_imread(f_path):
    img=cv2.imread(f_path)
    size=img.shape

def test_PIL_load_image(fpath):
    image = load_img(filename, target_size=(230,230))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

if __name__ == "__main__":

    SRC_IMAGES_DIR = "../src-images"
    good_list, bad_list = verify_images_readable(SRC_IMAGES_DIR)

    if len(bad_list) > 0:
        print("bad_list:")
        for f_path in bad_list:
            print(f_path)

