import os
import sys
import cv2
import shlex
from subprocess import PIPE, Popen
import time

# python3 -m pip install --upgrade pip
# python3 -m pip install --upgrade Pillow
from PIL import Image

# pip install keras-utils
from keras.utils import load_img, img_to_array

# pip install opencv-python
from cv2 import imread, resize

CSV_DATA_DIR = "../csv-data"
ALL_SRC_IMAGES_DIR = "../all-src-images"


def run_piped_commands(cmds):
    '''
    given an array of commands, pipes them, and returns a list of string
    from https://python-forum.io/thread-32163.html
    '''
    def run_pipes(cmds):
        '''Run commands in PIPE, return the last process in chain'''
        cmds = map(shlex.split, cmds)
        first_cmd, *rest_cmds = cmds
        procs = [Popen(first_cmd, stdout=PIPE)]
        for cmd in rest_cmds:
            last_stdout = procs[-1].stdout
            proc = Popen(cmd, stdin=last_stdout, stdout=PIPE)
            procs.append(proc)
        return procs[-1]

    lines = []
    last_proc = run_pipes(cmds)
    stdout = last_proc.stdout
    for line in stdout:
        line = line.decode()
        lines.append(line)
    return lines

def count_lines_in_file(path):
    '''return the number of lines in the given file or -1'''
    lines =  run_piped_commands([f"cat {path}","wc -l"])
    if len(lines) > 0:
        result = lines[0]   # only the first line is used
        size_str = result.strip()
        if size_str.isnumeric():
            return int(size_str)
    return -1

def test_cv2_imread(f_path):
    img=cv2.imread(f_path)
    dim = [100,100]
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

def test_PIL_Image_load_image(f_path):
    dim = (230,230)
    img = load_img(f_path, target_size=dim)
    # convert the image pixels to a numpy array
    img = img_to_array(img)
    # reshape data for the model
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

def is_invalid_image_file(f_path):
    '''Returns true if the given image file path cannot be decoded 
    by either cv2.imread or PIL.image_load'''
    try:
        test_cv2_imread(f_path)
        test_PIL_Image_load_image(f_path)
        return False
    except Exception as exp:
        return True
    

def find_missing_or_invalid_csv_data_jpg_images(csv_data_file_path, src_images_dir):
    '''
    Traverse the list of jpg image file names in the given 
    csv_data_file_path. Return a list of jpg image files under 
    all_src_images_dir that are missing or have an unreadable file format
    ''' 
    missing = []
    invalid = []
    ok = []
    assert os.path.isfile(csv_data_file_path), f"ERROR: {csv_data_file_path} is not a file"
    num_lines = count_lines_in_file(csv_data_file_path)
    num_digits = len(str(num_lines))
    if num_lines > 0:
        with open(csv_data_file_path,"r") as csv:
            cnt = 0
            start = time.perf_counter()
            while True:
                line = csv.readline()
                if not line:
                    break;
                line = line.strip()
                if len(line) > 0:  # skip blank lines
                    # TT_S01_E02_FRM-00-19-12-12.jpg,Common
                    filename = line.split(',')[0]
                    if filename.endswith(".jpg"):
                        cnt += 1
                        if (cnt % 1000) == 0:
                            millis_per_thousand = time.perf_counter() - start
                            thousands_remaining = num_lines - cnt
                            seconds_remaining = round( millis_per_thousand * thousands_remaining / 1000 )
                            cnt_str = str(cnt).rjust(num_digits) # right justified
                            print(cnt_str, "out of", num_lines, 
                                "- estimate", seconds_remaining, "seconds remaining")
                            start = time.perf_counter()
                        path = os.path.join(src_images_dir, filename)
                        if not os.path.isfile(path):
                            missing.append(path)
                        elif is_invalid_image_file(path):
                            invalid.append(path)
                        else:
                            ok.append(path)

    return missing, invalid


if __name__ == "__main__":
    
    # USAGE: 
    # activate
    # python test_image_files.py [csv-data-file-path] [src-images-dir]"
    
    # defaults
    csv_data_file_path = os.path.join(CSV_DATA_DIR,"S01E01-S01E02-data.csv")
    src_images_dir = ALL_SRC_IMAGES_DIR

    # optional overrides
    if len(sys.argv) > 1:
        csv_data_file_path = sys.argv[1]
    if len(sys.argv) > 2:
        src_images_dir = sys.argv[2]
    
    # here we go
    missing, invalid = find_missing_or_invalid_csv_data_jpg_images(
        csv_data_file_path, 
        src_images_dir)
    
    # output missing
    N = len(missing)
    print(f"num missing jpg images: {N}")
    for i in range(min(N,10)):
        print(missing[i])
        N = len(missing)
    
    # output invalid
    N = len(invalid)
    print(f"num invalid jpg images: {N}")
    for i in range(min(N,10)):
        print(invalid[i])

    print("done")