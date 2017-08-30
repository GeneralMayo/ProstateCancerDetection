
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

# Image Sequence
ImageSequence = ["T2tra", "T2sag", "ADC", "BVAL"]
ImageSequence_ROI = ["T2ax", "T2sag", "ADC", "DWI"]

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
    os.mkdir(ROIPath+'/Train')
    os.mkdir(ROIPath+'/Test')

    imageTypes = ['ADC','BVAL','t2_tse_sag','t2_tse_tra']
    testTrain = ['Test','Train']
    for ttName in testTrain:
        for imageType in imageTypes:
            os.mkdir(ROIPath+'/'+ttName+'/'+imageType)
            for ggg in range(5):
                os.mkdir(ROIPath+'/'+ttName+'/'+imageType+'/'+str(ggg+1))

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
                xStart = ijk[0]-int(ROISize[0]/2)
                xFin = ijk[0]+int(ROISize[0]/2)
                yStart = ijk[1]-int(ROISize[1]/2)
                yFin = ijk[1]+int(ROISize[1]/2)
                if(xStart>0 and yStart>0 and xFin<ConstPixelDims[0] and yFin<ConstPixelDims[1]):
                    slicePixels = ArrayDicom[xStart:xFin,yStart:yFin,sliceNum]

                    #save slice as png in appropriate folder
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
                    imagePath = os.path.join(ROIPath,'Train',imageTypeFolderName,str(ggg))
                    imageName = row['ProxID']+'_'+imageTypeFolderName+'_'+row['fid']+'_'+str(sliceNum)
                    print('Saved: '+imageName)
                    scipy.misc.imsave(os.path.join(imagePath, imageName) + ".png", slicePixels)
                    
                    #fullScanPixels.append(slicePixels)


        #TODO test 3d CNN
        #fullScanPixels = np.array(fullScanPixels)
        #np.save(os.path.join(ROINumpyPath, sequence, 'ORI', image_filename), resized_ROI_image)

        """
        for i in range(ArrayDicom.shape[2]):




        # Load spacing values (in mm)
        ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
        x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
        y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
        z = np.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])


        plt.figure(dpi=300)
        plt.axes().set_aspect('equal', 'datalim')
        plt.set_cmap(plt.gray())
        plt.pcolormesh(x, y, np.flipud(ArrayDicom[:, :, 9]))
        plt.show()
        input()
        """


"""

print("preprocessing start")
start_time = time.time()

# read csv file
with open(os.path.join(CSVPath, CSVImages), 'r', newline='') as CSV_images_object:
    # for each roi mask
    for roi_file in os.listdir(ROIPath):
        # parse roi filename
        roi_filename = os.path.splitext(roi_file)[0]
        roi_filename_split = roi_filename.split('_')

        patient = roi_filename_split[0]
        if len(roi_filename_split) == 4:
            fid = int(roi_filename_split[1])
        else:
            CSV_images_object.seek(0)
            CSV_images_reader = csv.DictReader(CSV_images_object)
            for row in CSV_images_reader:
                if row['ProxID'].split('-')[1] == patient:
                    fid = int(row['fid'])
                    break
        sequence = roi_filename_split[len(roi_filename_split)-2]
        slice = roi_filename_split[len(roi_filename_split)-1]

        # skip DCE image
        if sequence == 'kt':
            continue
        print(patient, fid, sequence, slice)

        # read roi file
        roi_object = read_roi_file(os.path.join(ROIPath, roi_file))
        roi_temp = roi_object[roi_filename]

        # find image dimension and path
        CSV_images_object.seek(0)
        CSV_images_reader = csv.DictReader(CSV_images_object)
        for row in CSV_images_reader:
            if row['ProxID'].split('-')[1] == patient:
                if ((len(roi_filename_split) == 4) and (row['fid'] == str(fid))) or (len(roi_filename_split) == 3):
                    if row['DCMSerDescr'] == sequence or \
                            ((sequence == ImageSequence_ROI[0]) and (row['DCMSerDescr'] == ImageSequence[0])) or \
                            ((sequence == ImageSequence_ROI[3]) and (row['DCMSerDescr'] == ImageSequence[3])):
                        image_dimension = row['Dim'].split('x')
                        image_path = row['DCMSerUID']

                    '''
                    # for ADC, DWI
                    if sequence in [ImageSequence_ROI[2], ImageSequence_ROI[3]]:
                        if row['DCMSerDescr'] == ImageSequence[2]:
                            image_dimension = row['Dim'].split('x')
                            image_path = row['DCMSerUID']
                        elif row['DCMSerDescr'] == ImageSequence[3]:
                            image_dimension = row['Dim'].split('x')
                            image_path_DWI = row['DCMSerUID']
                    # for T2
                    else:
                        if ((sequence == ImageSequence_ROI[0]) and (row['DCMSerDescr'] == ImageSequence[0]))\
                                or ((sequence == ImageSequence_ROI[1]) and (row['DCMSerDescr'] == ImageSequence[1])):
                            image_dimension = row['Dim'].split('x')
                            image_path = row['DCMSerUID']
                            break
                    '''

        # make polygon
        polygon = []
        for x, y in zip(roi_temp['x'], roi_temp['y']):
            polygon.append(x)
            polygon.append(y)

        # make mask
        width = int(image_dimension[0])
        height = int(image_dimension[1])
        img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        mask = np.array(img)

        # set dcm image path
        patient_root = "ProstateX-" + patient
        patient_root_path = os.path.join(DicomPath, patient_root)

        for images in os.listdir(patient_root_path):
            image_path = os.path.join(patient_root_path, images, image_path)

            # Read dicom files
            onlyfiles = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
            for file in onlyfiles:
                ds = dicom.read_file(os.path.join(image_path, file))

                # get ROI image
                slice_num = int(ds[0x20, 0x13].value)
                if slice_num == int(slice):
                    ROI_i0 = min(roi_temp['x'])
                    ROI_i1 = max(roi_temp['x'])
                    ROI_j0 = min(roi_temp['y'])
                    ROI_j1 = max(roi_temp['y'])

                    # apply mask
                    masked_array = ds.pixel_array * mask
                    ROI_image = masked_array[ROI_j0:ROI_j1, ROI_i0:ROI_i1]

                    # resize ROi image
                    if sequence in [ImageSequence_ROI[0], ImageSequence_ROI[1]]:
                        dim = ROISize_T2
                    else:
                        dim = ROISize_ADC
                        # sequence = ImageSequence_ROI[2]
                    resized_ROI_image = cv2.resize(ROI_image, dim, interpolation=cv2.INTER_CUBIC)

                    # display
                    # plt.imshow(resized_ROI_image, cmap='gray')
                    # plt.show()

                    # save ROI image
                    # as PNG
                    image_filename = patient_root + "_" + str(fid) + "_" + str(sequence) + "_" + str(slice_num)
                    scipy.misc.imsave(os.path.join(ROIImagePath, sequence, 'ORI', image_filename) + ".png", resized_ROI_image)

                    # as npy
                    np.save(os.path.join(ROINumpyPath, sequence, 'ORI', image_filename), resized_ROI_image)

                    #
                    # make textural feature map
                    #

                    # calculate textural feature for every voxel from glcm using 5x5 patch centered at each voxel
                    feature_map = np.zeros(shape=(ROI_image.shape[0], ROI_image.shape[1], len(TexturalFeature)))
                    y = 0
                    for j in range(ROI_j0, ROI_j1):
                        x = 0
                        for i in range(ROI_i0, ROI_i1):
                            ROI_image_texture = ds.pixel_array[j-2:j+2, i-2:i+2]

                            ROI_image_uint8 = np.zeros(shape=ROI_image_texture.shape, dtype=np.uint8)
                            ROI_image_uint8 = cv2.convertScaleAbs(ROI_image_texture, alpha=(255.0/65535.0))
                            glcm = skf.greycomatrix(ROI_image_uint8, [1], [0], symmetric=True, normed=True)

                            # make feature map
                            idx = 0
                            for feature in TexturalFeature:
                                if feature == "entropy":
                                    feature_map[y, x, idx] = skm.shannon_entropy(glcm)

                                else:
                                    feature_map[y, x, idx] = skf.greycoprops(glcm, feature)
                                idx += 1

                            x += 1
                        y += 1

                    # apply mask and resize
                    masked_feature_map = np.zeros(shape=(ROI_image.shape[0], ROI_image.shape[1], len(TexturalFeature)))
                    resized_feature_map = np.zeros(shape=(resized_ROI_image.shape[0], resized_ROI_image.shape[1], len(TexturalFeature)))
                    ROI_mask = mask[ROI_j0:ROI_j1, ROI_i0:ROI_i1]
                    idx = 0
                    for feature in TexturalFeature:
                        masked_feature_map[:, :, idx] = feature_map[:, :, idx] * ROI_mask
                        resized_feature_map[:, :, idx] = cv2.resize(masked_feature_map[:, :, idx], dim, interpolation=cv2.INTER_CUBIC)
                        idx += 1

                    # save feature map image
                    idx = 0
                    for feature in TexturalFeature:
                        # as PNG
                        image_filename = patient_root + "_" + str(fid) + "_" + str(sequence) + "_" + feature + "_" + str(slice_num)
                        scipy.misc.imsave(os.path.join(ROIImagePath, sequence, 'TF', image_filename) + ".png", resized_feature_map[:, :, idx])

                        # as npy
                        np.save(os.path.join(ROINumpyPath, sequence, 'TF', image_filename), resized_feature_map[:, :, idx])
                        idx += 1
"""
