import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
import sys

import pdb

labels = '../data/splits.csv'
traindir = './train/'
testdir = './test/'

# Keys to csv mapping
case_code = {'Cmpgaze_V': 'Composite Gaze - With Vision',
             'DH_L' : 'Dix-Hallpike - Head Left - Vision Denied',
             'DH_R' : 'Dix-Hallpike - Head Right - Vision Denied',
             'GazeC_V' : 'Gaze - Center - With Vision',
             'GazeC_VD' : 'Gaze - Center - Vision Denied',
             'GazeD_V' : 'Gaze - Down - With Vision',
             'GazeD_VD' : 'Gaze - Down - Vision Denied',
             'GazeU_V' : 'Gaze - Up - With Vision',
             'GazeU_VD' : 'Gaze - Up - Vision Denied',
             'GazeL_V' : 'Gaze - Left - With Vision',
             'GazeL_VD' : 'Gaze - Left - Vision Denied',
             'GazeR_V' : 'Gaze - Right - With Vision',
             'GazeR_VD' : 'Gaze - Right - Vision Denied',
             'GazeReb_R_VD' : 'Gaze - Rebound Right - Vision Denied',
             'GazeReb_R_V' : 'Gaze - Rebound Right - With Vision',
             'GazeReb_L_VD' : 'Gaze - Rebound Left - Vision Denied',
             'GazeReb_L_V' : 'Gaze - Rebound Left - With Vision',
             'HSN' : 'Headshake - Head Straight - Vision Denied',
             'Saccade' : '-Laser Saccade',
             'Bow' : 'Lateropulsion-Bow-Lean - Chin to Chest - Vision Denied',
             'lean' : 'Lateropulsion-Bow-Lean - Head Extended Backward - Vision Denied',
             'Bow/Lean_V' : 'Lateropulsion-Bow-Lean - Sitting - With Vision',
             'Roll_L' : 'Roll - Leftward - Vision Denied',
             'Roll_R' : 'Roll - Rightward - Vision Denied'
}

class_code = {0: 'normal',
              1: 'nystagmus'
             }

beta = 0.5

# helper for reordering frames by timestamp
def ind(x):
    #return int(x.split('_')[-1][:-4])
    return int(x.split('_')[-1].split('.')[0])

# recursively filter batches of 60*x frames (x seconds), save final filtered image
def recursive_filter2(imgspath, imgsname, seconds=1):
    if os.path.exists(imgspath):
        frames = [imgspath + f for f in os.listdir(imgspath)]
        if len(frames) >= 600:

            # filter out non-image files
            frames = list(filter(lambda x: 'npy' not in x, frames))

            # reorder frames by timestamp
            frames = sorted(frames, key=ind)

            seconds_adjustment = int(60*seconds)
            for i in range(max(len(frames)-(seconds_adjustment-1), len(frames))):
                try:
                    save = True
                    im0 = cv2.imread(frames[i])
                    m0 = im0
                    mt = (((1-beta) * m0 + beta * im0)*255).astype(np.uint8)
                    for j in range(1, seconds_adjustment):
                        if(i+j >= len(frames)):
                            save = False
                            break
                        im1 = frames[i+j]
                        im1 = cv2.imread(im1)
                        ft = cv2.absdiff(im1, mt)
                        m0 = mt
                        mt = (((1-beta) * m0 + beta * im1)*255).astype(np.uint8)
                    if save:
                        # save
                        cv2.imwrite(imgsname+str(i)+'.jpg', ft)
                    else:
                        break
                except:
                    print("Error Batch: ", frames[i])

# resursively filter frames for one video to create filtered images
def recursive_filter(imgspath, imgsname,seconds=0):
    if os.path.exists(imgspath):
        frames = [imgspath + f for f in os.listdir(imgspath)]
        if len(frames) >= 600:

            # filter out non-image files
            frames = list(filter(lambda x: 'npy' not in x, frames))

            # reorder frames by timestamp
            frames = sorted(frames, key=ind)
            
            # REVERSAL ONLY
            frames.reverse()
            # END REVERSAL ONLY

            # seconds calculation
            if seconds > 0:
                seconds_adjustment = int(60*seconds)-1
            else:
                seconds_adjustment = 1

            im0 = cv2.imread(frames[0])
            m0 = im0
            mt = (((1-beta) * m0 + beta * im0)*255).astype(np.uint8)
            for i in range(1,len(frames)):
                try:
                    im1 = frames[i]
                    im1 = cv2.imread(im1)
                    ft = cv2.absdiff(im1, mt)
                    if i > seconds_adjustment: cv2.imwrite(imgsname+str(i)+'.jpg', ft)
                    m0 = mt
                    mt = (((1-beta) * m0 + beta * im1)*255).astype(np.uint8)
                except:
                    print("Error: ", frames[i])

rfmethods = {1: recursive_filter, 2: recursive_filter2}

def fix_key(k):
    if '/' in k:
        k = k.replace('/','')
    return k

def remake_keys(key_value):
    name_code =  {value:fix_key(key) for key, value in key_value.items()}
    return name_code

def name_images(code, df, datadir):
    if not os.path.exists(class_code[code]): os.mkdir(class_code[code])
    paths = df.loc[df['Label'] == code, 'Video']
    name_code = remake_keys(case_code)
    return name_code, paths

def recursive_filter(code, df, datadir, dest, rfmethod, seconds):
    name_code, paths = name_images(code, df, datadir)
    # REVERSING ONLY
    #paths = paths[:75]
    for path in paths:
        imgpath = datadir + path
        case = path.split('/')[0][-2:]
        vid = name_code[path.split('/')[1]]
        imgname = './'+class_code[code]+'/'+'r-'+case+'-'+vid+'-' # FOR REVERSING ADD IN 'r-
        rfmethods[rfmethod](imgpath, imgname, seconds=seconds)
    
def cleanup():
    try: shutil.rmtree('normal')
    except: pass
    try: shutil.rmtree('nystagmus')
    except: pass

def cleanup_reversed():
    try:
        for f in os.listdir('normal'):
            if 'r-' == f[:2]:
                os.remove('./normal/'+f)
        for f in os.listdir('nystagmus'):
            if 'r-' == f[:2]:
                os.remove('./nystagmus/'+f)
    except:
        pass

def preprocess_dir(datadir, dest, split_df, rfmethod, seconds):
    if not os.path.exists(dest): os.makedirs(dest)
    os.chdir(dest)
    # cleanup() # FOR REVERSING COMMENT OUT
    cleanup_reversed() # FOR REVERSING
    recursive_filter(0, split_df, '../../'+datadir, dest, rfmethod, seconds)
    recursive_filter(1, split_df, '../../'+datadir, dest, rfmethod, seconds)
    os.chdir('../..')

def preprocess(datadir, outdir, rfmethod=1, seconds=0):
    df = pd.read_csv(labels, index_col=0)
    
    split_df = df.loc[df['Split'] == 'Train']
    dest = traindir + outdir + '/'
    preprocess_dir(datadir, dest, split_df, rfmethod, seconds)
    
    # COMMENT OUT FOR REVERSING ONLY
    split_df = df.loc[df['Split'] == 'Test']
    dest = testdir + outdir + '/'
    preprocess_dir(datadir, dest, split_df, rfmethod, seconds)

if __name__=="__main__":
    # example - python preprocessing.py ../data/images/frames/ original 1 0
    scriptStart = datetime.now()
    datadir = str(sys.argv[1])
    outdir = str(sys.argv[2])
    rfmethod = int(str(sys.argv[3]))
    seconds = float(str(sys.argv[4]))
    preprocess(datadir, outdir, rfmethod, seconds)
    print('Total script time: ', datetime.now()-scriptStart)