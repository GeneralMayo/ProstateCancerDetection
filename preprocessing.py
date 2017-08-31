
import dicom
import os
import csv
import time
import shutil
import scipy.misc
import numpy as np
from PIL import Image, ImageDraw
import skimage.feature as skf
import skimage.measure as skm
#import cv2


from matplotlib import pyplot as plt

MAKE_DIRS = True

# ROI size
ROISize_T2 = (40, 40)
ROISize_ADC = (20, 20)
ROISliceDistance = 2

# source locations
DicomPath = "Train/DOI"
ROIPath = "Train/PreprocessedDOI"
CSVPath = "Train/ProstateX2-DataInfo-Train"
CSVFindings = "ProstateX-2-Findings-Train.csv"
CSVImages = "ProstateX-2-Images-Train.csv"

#create testing and training directories
if(MAKE_DIRS):
    shutil.rmtree(ROIPath)
    os.mkdir(ROIPath)

    imageTypes = ['ADC','BVAL','t2_tse_sag','t2_tse_tra']
    for imageType in imageTypes:
        os.mkdir(ROIPath+'/'+imageType)
        for ggg in range(5):
            os.mkdir(ROIPath+'/'+imageType+'/'+str(ggg+1))

#load image findings into dict
findingsDict = {}
with open(os.path.join(CSVPath,CSVFindings), 'rb') as findingsCSV:
    findingsCSVReader = csv.reader(findingsCSV, delimiter=',')
    #row example = ProstateX-0000,1,25.7457 31.8707 -38.511,PZ,3
    for row in findingsCSVReader:
        if(len(row)>0 and row[0].split("-")[0] == 'ProstateX'):
            allPatients = findingsDict.keys()
            if(row[0] in allPatients):
                findingsDict[row[0]][row[1]] = int(row[4])
            else:
                findingsDict[row[0]] = {row[1]:int(row[4])}


with open(os.path.join(CSVPath, CSVImages), 'r') as CSV_images_object:
    CSV_images_reader = csv.DictReader(CSV_images_object,lineterminator='\n')
    i = 0
    for row in CSV_images_reader:
        #COLLECT .dcm file names
        #make file path to a particular MRI scan for a particular patient
        patientDirectory = next(os.walk(os.path.join(DicomPath,row['ProxID'])))[1][0]
        scanNumber = row['DCMSerUID']
        dicomFilesPath = os.path.join(DicomPath,row['ProxID'],patientDirectory,scanNumber)

        #collect file names of .dcm files
        lstFilesDCM = [] 
        for dirName, subdirList, fileList in os.walk(dicomFilesPath):
            for filename in fileList:
                if ".dcm" in filename.lower():
                    lstFilesDCM.append(os.path.join(dirName,filename))

        #get reference DICOM file
        RefDs = dicom.read_file(lstFilesDCM[0])
        #get dimentions of file
        ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
        #init pixel array
        ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
        #fill pixel array
        for filenameDCM in lstFilesDCM:
            #read the file
            ds = dicom.read_file(filenameDCM)
            #store the raw image data
            ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array

        #crop 3d image based on ijk and image size
        ijk = map(int,row['ijk'].split(' '))
        imageType = row['DCMSerDescr']
        if(imageType == 't2_tse_sag' or imageType == 't2_tse_tra'):
            ROISize = ROISize_T2
        elif(imageType.split('_')[-1] == 'ADC' or imageType.split('_')[-1] == 'BVAL'):
            ROISize = ROISize_ADC
        else:
            print('Unrecognized Image Type: '+imageType)
            break

        #crop slices
        fullScanPixels = []
        for sliceNum in range(ArrayDicom.shape[2]):
            if(abs(sliceNum - ijk[2]) <= ROISliceDistance):   
                #get bounds
                xStart = ijk[0]-int(ROISize[0]/2)
                xFin = ijk[0]+int(ROISize[0]/2)
                yStart = ijk[1]-int(ROISize[1]/2)
                yFin = ijk[1]+int(ROISize[1]/2)
                if(ConstPixelDims[0]<ROISize[0] or ConstPixelDims[1]<ROISize[1]):
                    raise NameError("ROI larger than image.")
                if(xStart<0):
                    xStart = 0
                    xFin = ROISize[0]
                elif(xFin>ConstPixelDims[0]):
                    xStart = ConstPixelDims[0]-ROISize[0]
                    xStart = ConstPixelDims[0]
                if(yStart<0):
                    yStart = 0
                    yFin = ROISize[1]
                elif(yFin>ConstPixelDims[1]):
                    yStart = ConstPixelDims[1]-ROISize[1]
                    yFin = ConstPixelDims[1]

                #save slice as png in appropriate folder
                slicePixels = ArrayDicom[xStart:xFin,yStart:yFin,sliceNum]
                if(imageType == 't2_tse_sag' or imageType == 't2_tse_tra'):
                    imageTypeFolderName = imageType
                elif(imageType.split('_')[-1] == 'ADC'):
                    imageTypeFolderName = 'ADC'
                elif(imageType.split('_')[-1] == 'BVAL'):
                    imageTypeFolderName = 'BVAL'
                else:
                    raise NameError('Image type issue should have been caught.')

                #get ggg
                ggg = findingsDict[row['ProxID']][row['fid']]

                #save slice
                imagePath = os.path.join(ROIPath,imageTypeFolderName,str(ggg))
                imageName = row['ProxID']+'_'+imageTypeFolderName+'_'+row['fid']+'_'+str(sliceNum)
                print('Saved: '+imageName)
                scipy.misc.imsave(os.path.join(imagePath, imageName) + ".png", slicePixels)
                #fullScanPixels.append(slicePixels)



                    


        #TODO test 3d CNN
        #fullScanPixels = np.array(fullScanPixels)
        #np.save(os.path.join(ROINumpyPath, sequence, 'ORI', image_filename), resized_ROI_image)

        