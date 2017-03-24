from PIL import Image
from numpy import array
from numpy import load
from numpy import save
from scipy import misc
import cv2
import os
import pickle

rootdir = 'C:\Users\danie\Dropbox\college\\2016_2017_2nd\\ML\\finalProject\\fingerspelling5'
pickleRoot = 'C:\Users\danie\Dropbox\college\\2016_2017_2nd\\ML\\finalProject\\aslRecognition'

denoisedArrays = []
filteredArrays = []
labels = []

i = 0
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file[:5] == "color":
            print file
            path = subdir + '\\' + file
            img = Image.open(path)
            imgArray = array(img)
            bandsRemovedArray = imgArray[:-1]/2 + imgArray[1:]/2
            denoiseArray = cv2.fastNlMeansDenoisingColored(bandsRemovedArray,None,9,9,3,21)
            filterArray = cv2.bilateralFilter(bandsRemovedArray, 15, 80, 80)
            denoiseImg = Image.fromarray(denoiseArray)
            filterImg = Image.fromarray(filterArray)
            denoisedArrays.append(denoiseArray)
            filteredArrays.append(filterArray)

            fileString = file.split('_')
            label = int(fileString[1])
            labels.append(label)

            i += 1

pickleFile = open(pickleRoot+"\pickledDenoised.npy", 'wb')
nparr = array(denoisedArrays)
print len(nparr)
print nparr
save(pickleFile, nparr)
pickleFile.close()

pickleFile = open(pickleRoot+"\pickledFiltered.npy", 'wb')
nparr = array(filteredArrays)
print len(nparr)
print nparr
save(pickleFile, nparr)
pickleFile.close()

pickleFile = open(pickleRoot+"\pickledLabels.npy", 'wb')
nparr = array(labels)
print len(nparr)
print nparr
save(pickleFile, nparr)
pickleFile.close()

pickleFile = open(pickleRoot+"\pickledDenoised.npy", 'rb')
newarr = load(pickleFile)
print newarr
pickleFile.close()

pickleFile = open(pickleRoot+"\pickledFiltered.npy", 'rb')
newarr = load(pickleFile)
print newarr
pickleFile.close()
