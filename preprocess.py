import numpy as np
import cv2
import os
import argparse
import itertools
import subprocess
import matplotlib.cm as cm

def padTo(img, w,h, value):
    ih,iw = img.shape[:2]
    left = (w-iw) / 2
    right = w-iw-left
    top = (h-ih) / 2
    bottom = h-ih-top
    return cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=value)

def smartResize(img, sidelen, value):
    ms = max(img.shape[:2])
    padded = padTo(img,ms,ms, value)
    if ms > sidelen:
        imethod = cv2.INTER_AREA
    else:
        imethod = cv2.INTER_LINEAR
    return cv2.resize(padded,(sidelen, sidelen), interpolation=imethod)

def preprocess(datadir, outfile, letters="abcdefghiklmnopqrstuvwxy", overfeat=None):

    color_data = []
    depth_data = []
    label_data = []
    person_data = []

    for pi, person in enumerate("ABCDE"):
        for letter in "abcdefghiklmnopqrstuvwxy":
            li = ord(letter)-ord("a")
            if letter not in letters:
                continue

            found = 0
            for i in xrange(2, 1000, 10):
                color_path = os.path.join(datadir, person, letter, "color_{}_{:04}.png".format(li, i))
                depth_path = os.path.join(datadir, person, letter, "depth_{}_{:04}.png".format(li, i))
                if not (os.path.isfile(color_path) and os.path.isfile(depth_path)):
                    continue

                found += 1

                img = cv2.imread(color_path,cv2.IMREAD_COLOR)
                bandless_img = img[:-1]/2 + img[1:]/2
                bandless_img = bandless_img.astype('float') / 255.0

                depth = cv2.imread(depth_path,cv2.IMREAD_UNCHANGED).astype('float')
                pc = np.percentile(depth[depth>0],.05)
                depthmask = (depth>0)&(depth<100+pc)
                fixdepth = (1 - (depth - np.min(depth[depth>0]))/100.0)*depthmask

                if overfeat is not None:
                    bandless_resize = smartResize(bandless_img, 231, (1,1,1))
                    bandless_resize = (bandless_resize*255).astype('uint8')
                    cv2.imwrite('tmp_img.png',bandless_resize)
                    output = subprocess.check_output([overfeat,'-f','tmp_img.png'], universal_newlines=True)
                    feats = np.fromstring(output.split('\n')[1], sep=" ")
                    color_data.append(feats)

                    depthcolor = (cm.get_cmap('gist_heat')(fixdepth)[:,:,:3]*255)[:,:,::-1]
                    depthcolor[~depthmask]=np.array([22,45,18])
                    depthcolor_resize = smartResize(depthcolor, 231, (22,45,18)).astype('uint8')
                    cv2.imwrite('tmp_img.png',depthcolor_resize)
                    output = subprocess.check_output([overfeat,'-f','tmp_img.png'], universal_newlines=True)
                    feats = np.fromstring(output.split('\n')[1], sep=" ")
                    depth_data.append(feats)

                else:
                    bandless_resize = smartResize(bandless_img, 128, (1,1,1))
                    color_data.append(bandless_resize.flatten())
                    fixdepth_resize = smartResize(fixdepth, 128, 0)
                    depth_data.append(fixdepth_resize.flatten())

                label_data.append(li)
                person_data.append(pi)

            print "Processed {} {}: {} examples".format(person, letter, found)

    color_data = np.stack(color_data)
    depth_data = np.stack(depth_data)
    label_data = np.array(label_data)
    person_data = np.array(person_data)

    print "color_data:", color_data.shape
    print "depth_data:", depth_data.shape
    print "label_data:", label_data.shape
    print "person_data:", person_data.shape

    np.savez_compressed(outfile, color=color_data, depth=depth_data, label=label_data, person=person_data)

parser = argparse.ArgumentParser(description='Preprocess ASL images')
parser.add_argument('datadir', help='Dataset directory')
parser.add_argument('outfile', help='Output file')
parser.add_argument('--letters', help='Letters to include')
parser.add_argument('--overfeat', help='Path to overfeat')

if __name__ == '__main__':
    namespace = parser.parse_args()
    args = vars(namespace)
    preprocess(**args)

