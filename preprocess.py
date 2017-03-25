import numpy as np
import cv2
import os
import argparse
import itertools

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

def preprocess(datadir, outfile, letters="abcdefghiklmnopqrstuvwxy"):
    
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
            for i in itertools.count(2,10):
                color_path = os.path.join(datadir, person, letter, "color_{}_{:04}.png".format(li, i))
                depth_path = os.path.join(datadir, person, letter, "depth_{}_{:04}.png".format(li, i))
                if not (os.path.isfile(color_path) and os.path.isfile(depth_path)):
                    break

                found += 1

                img = cv2.imread(color_path,cv2.IMREAD_COLOR)
                bandless_img = img[:-1]/2 + img[1:]/2
                bandless_img = bandless_img.astype('float') / 255.0
                bandless_resize = smartResize(bandless_img, 128, (1,1,1))

                color_data.append(bandless_resize.flatten())

                depth = cv2.imread(depth_path,cv2.IMREAD_UNCHANGED)
                depthmask = (depth>0)&(depth<100+np.percentile(depth[depth>0],.05))
                fixdepth = (np.max(depth[depthmask]) - depth)*depthmask
                fixdepth = fixdepth.astype('float') / 100.0
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

if __name__ == '__main__':
    namespace = parser.parse_args()
    args = vars(namespace)
    preprocess(**args)

