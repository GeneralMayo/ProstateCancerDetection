import os, csv, shutil, random

#constants
MAKE_DIRS = True
PERCENT_TRAIN = .7

#create subdirectories
if(MAKE_DIRS):
    shutil.rmtree('BinaryClassificationData')
    os.mkdir('BinaryClassificationData')
    os.mkdir('BinaryClassificationData/Train')
    os.mkdir('BinaryClassificationData/Test')

    folderNames = ['ADC0','BVAL0','t2_tse_sag0','t2_tse_tra0']
    testTrain = ['Test','Train']
    for ttName in testTrain:
        for folderName in folderNames:
            os.mkdir('BinaryClassificationData/'+ttName+'/'+folderName)
            os.mkdir('BinaryClassificationData/'+ttName+'/'+folderName+'/malignant')
            os.mkdir('BinaryClassificationData/'+ttName+'/'+folderName+'/benign')

#get lieson classifications
findingsDict = {}
with open('ProstateX-2-Findings-Train.csv', 'rb') as findingsCSV:
    findingsCSVReader = csv.reader(findingsCSV, delimiter=',')
    #row example = ProstateX-0000,1,25.7457 31.8707 -38.511,PZ,3
    for row in findingsCSVReader:
        if(len(row)>0 and row[0].split("-")[0] == 'ProstateX'):
            allPatients = findingsDict.keys()
            if(row[0] in allPatients):
                liesonDict[row[0]][row[1]] = int(row[4])
            else:
                liesonDict[row[0]] = {row[1]:int(row[4])}

#organize images
for root, dirs, files in os.walk('ScreenshotsTrain'):
    #file name example = ProstateX-0000-Finding1-ep2d_diff_tra_DYNDIST_ADC0.bmp
    for fileName in files:
        #get glesion grade group
        allImageData = fileName.split('-')
        patientName = allImageData[0]+'-'+allImageData[1]
        print(patientName)
        findNumber = allImageData[2][-1]
        ggg = liesonDict[patientName][findNumber]

        #get image type (such that it can be placed in correct folder)
        imageTypeData = allImageData[-1].split('_')
        imageTypeData = imageTypeData[-1][0:-4]
        if(imageTypeData == 'ADC0' or imageTypeData == 'BVAL0'):
            folderName = imageTypeData
        elif(imageTypeData == 'sag0'):
            folderName = 't2_tse_sag0'
        elif(imageTypeData == 'tra0'):
            folderName = 't2_tse_tra0'
        else:
            print(imageTypeData)
            continue

        #copy image to correct folder
        src = 'ScreenshotsTrain/'+fileName
        if(ggg>1): #malignant image
            dst = 'BinaryClassificationData/Train/'+folderName+'/malignant/'+fileName
        else: #benign image
            dst = 'BinaryClassificationData/Train/'+folderName+'/benign/'+fileName
        shutil.copyfile(src,dst)

for root, dirs, files in os.walk('BinaryClassificationData'):
    if(len(files)>0 and root.split('/')[1] != 'Test'):
        #choose 30% of data to use for testing
        random.shuffle(files)
        splitPoint = int(round(len(files)*PERCENT_TRAIN))
        toMove = files[splitPoint:]


        #move that data
        for fileName in toMove:
            src = root+'/'+fileName

            dstFolders = src.split('/')
            dstFolders[1] = 'Test'
            dst = "/".join(dstFolders)
            shutil.move(src,dst)


