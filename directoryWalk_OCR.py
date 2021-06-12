#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: saul
"""
#Travel through all directories and sub-directories and grab images. Then convert images to text.
import os 
import glob
data_list = [['']]
import os
from PIL import Image as PI
from pytesseract import image_to_string
import pytesseract
import pandas as pd 
import PyPDF2
from os import listdir
from os.path import isfile, join

tesseract_cmd = 'tesseract'
dirpath = '/home/saul/anaconda3/' 
#dirpath = '/home/saul/pythontraining/'  
dirpath = '/media/saul/UUI/GT4HistOCR(1)/' 
outputpath = '/home/saul/pythontraining/NLP/'                                                                                                       
enlargesize = 1

def convertImageString(folders):
    image_to_text_list = [['', '', '']] 
    image_list = []
    print ("Folders ", folders)
    #crawl through each folder
    for folder in range(len(folders)):
        image_to_text = [['', '','']]    
        print("Folder ", folder)
        evidencepath = dirpath + folders[folder] + '/'
        #evidenceFiles = [file for file in listdir(evidencepath) if isfile(join(evidencepath,file))]
        #evidenceFiles = [file for file in glob.glob(evidencepath +"**/*.pl", recursive=True)]
        evidenceFiles = [file for file in glob.glob(evidencepath +"**/*nrm.png", recursive=True)]
        #evidenceFiles  = [x for x in evidenceFiles if x !=[]]
        #print("Evidence Files ", evidenceFiles)
        #print("Evidence Path ", evidencepath )
    
        if evidenceFiles != []:
            for image in evidenceFiles:
                #print(folders[folder], " ", image, '\n')
                image_list.append(image)
            imagelenght = len(image_list)
            #print("Image List ", image_list)
       
            for img in range(imagelenght):
                image = image_list[img]
                #print("Image ", image)
                #print("Evidence Path ",evidencepath)
                #a = PI.open(evidencepath.format(image)).convert("RGBA")
                a = PI.open(image).convert("RGBA")
                #print("Image ", image.split('/')[-1])
                width, height = a.size
                new_size = width*enlargesize, height*enlargesize
                img = a.resize(new_size, PI.LANCZOS)
                img = img.convert('L')
                img= img.point(lambda x:0 if x < 155 else 255, '1')
                ocrtext = pytesseract.image_to_string(img)
                image_to_text_list.append([folders[folder],image.split('/')[-1], ocrtext.encode('utf-8') ])
                
        print("Image to Text ", image_to_text_list)
        evidences = pd.DataFrame(image_to_text_list, columns=['folderName', 'imageName', 'Text']) 
        evidences.to_csv(outputpath + 'OCROutput.csv', sep=',', index=False)
        
    #Remove empty lists
    #evidenceFiles  = [x for x in evidenceFiles if x !=[]]
    #print(folders[folder], " ",evidenceFiles, '\n')
    #evidenceFiles = [f for f in listdir(evidencepath) if isfile(join(evidencepath, f))]
    #script_path = os.path.dirname(os.path.realpath(__file__))
    #maps_path = os.path.join(script_path, evidencepath)
    
    a = PI.open("/home/saul/pythontraining/NLP/handwriting.jpg")
    b= image_to_string(a)
    data_list.append([[b]])
    #convert list to dataframe
    bb = pd.DataFrame(data_list, columns=['text'])

def list_files(dir):                                                                                                  
    r = []                                                                                                            
    subdirs = [x[0] for x in os.walk(dir)]                                                                            
    for subdir in subdirs:                                                                                            
        files = next(os.walk(subdir))[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files:                                                                                        
                r.append(subdir + "/" + file)                                                                         
    return r   

def getSubDirs(dir):
    #print(dir)
    dirs = [x[1] for x in os.walk(dir)]
    #delete empty lists
    dirs = [x for x in dirs if x !=[]]
    #print(dirs[0])
    #print("Number of directories", len(dirs[0])) #number of directories
    #print('var' in dirs)
    #print(dirs)
    #print("Direcrories ", dirs[0])
    convertImageString(dirs[0])
 
    ListFiles = os.walk(os.getcwd())
    SplitTypes = []
    for walk_output in ListFiles:
        for file_name in walk_output[-1]:
            SplitTypes.append(file_name.split(".")[-1])
    list(set(SplitTypes))  #remove duplicate elements in the list
 
    types = ['*.jpg', '*.pdf', '*.odg']
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(files))
print(SplitTypes)
    
if __name__ == '__main__':
    print("Test")
    getSubDirs(dirpath)
    
    
