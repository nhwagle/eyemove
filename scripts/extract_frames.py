import os
import cv2

# extract first frame from all videos
def extract_frame1(datadir):
    os.chdir(datadir)
    vids = [os.path.join(d, f) for d, dirs, files in os.walk(os.getcwd()) for f in files]

    for vid in vids:
        name = vid.split('\\')[-1][:-4]
        vidcap = cv2.VideoCapture(vid)
        success, image = vidcap.read()
        cv2.imwrite("../images/"+name+'.jpg', image)
        
# extract all frames from 1 video
def extract_frames_single_video(vidpath, vidname, outdir):
    vidcap = cv2.VideoCapture(vidpath)
    success, image = vidcap.read()
    count = 0
    outpath = outdir + '/' + vidname + '_'
    while success:
        cv2.imwrite(outpath+"%d.jpg" %count, image)
        success, image = vidcap.read()
        count += 1

# extract frames from all videos in case
def extract_frames_single_case(casepath, outdir):
    vidpaths = [casepath + '/' + vid for vid in os.listdir(casepath)]
    for vidpath in vidpaths:
        vid = vidpath.split('_')[-1][:-4]
        out = outdir + '/' + vid
        os.mkdir(out)
        extract_frames_single_video(vidpath, vid, out)

#extract frames from all videos in directory
def extract_frames(datadir, outdir):
    casepaths = [datadir + '/' + case for case in os.listdir(datadir)]
    for casepath in casepaths:
        case = casepath.split('/')[-1]
        out = outdir + '/' + case
        os.mkdir(out)
        extract_frames_single_case(casepath, out)