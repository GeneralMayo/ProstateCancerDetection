import os, shutil, random

ACTIVE_DATA_ROOT = 'Train/ActiveData'
INACTIVE_DATA_ROOT = 'Train/PreprocessedDOI'

#['ADC','BVAL','t2_tse_sag','t2_tse_tra']
IMAGE_TYPE = 'ADC'
CLASS_MAPPING_LIST = [[[1],[1]],[[2],[2]]]
PERCENT_TRAIN = .7

#make active data folders
shutil.rmtree(ACTIVE_DATA_ROOT)
os.mkdir(ACTIVE_DATA_ROOT)
os.mkdir(os.path.join(ACTIVE_DATA_ROOT,'Train'))
os.mkdir(os.path.join(ACTIVE_DATA_ROOT,'Test'))

ttNames = ['Test','Train']
for ttName in ttNames:
    for classMapping in CLASS_MAPPING_LIST:
        os.mkdir(os.path.join(ACTIVE_DATA_ROOT,ttName,str(classMapping[1][0])))

#move images to test train folders
for root, dirs, files in os.walk(os.path.join(INACTIVE_DATA_ROOT,IMAGE_TYPE)):
    if(len(files)>1):
        random.shuffle(files)
        splitPoint = int(round(len(files)*PERCENT_TRAIN))
        TrainFileNames = files[:splitPoint]
        TestFileNames = files[splitPoint:]
        
        #get class mapping
        curClass = int(root.split('/')[-1])
        dstClass = None
        for classMapping in CLASS_MAPPING_LIST:
            if(curClass in classMapping[0]):
                dstClass = str(classMapping[1][0])
                
                #move data
                for fileName in TrainFileNames:
                    src = root+'/'+fileName
                    dst = os.path.join(ACTIVE_DATA_ROOT,'Train',dstClass,fileName)
                    shutil.copyfile(src,dst)

                for fileName in TestFileNames:
                    src = root+'/'+fileName
                    dst = os.path.join(ACTIVE_DATA_ROOT,'Test',dstClass,fileName)
                    shutil.copyfile(src,dst)

                break

                


        

            

