import subprocess as sp
import numpy as np
import pandas as pd
from io import StringIO
import os
import re
import shutil
import sqlite3

def parse_filename_nsight(filename):
    #empty dicts
    result={}
    
    #add network name
    result["Cell"] = re.match(r'.*\.celltype_(.*?)\.',filename).groups()[0]
    result["Hidden Size"] = int(re.match(r'.*\.nneu_(.*?)\.',filename).groups()[0])
    result["Input Shape"] = re.match(r'.*\.input_(.*?)\.',filename).groups()[0]
    result["Batch Size"] = int(result["Input Shape"].split("x")[0])
    result["Time Steps"] = int(result["Input Shape"].split("x")[1])
    result["Features"] = int(result["Input Shape"].split("x")[2])
    result["Pass"] = re.match(r'.*\.pass_(.*?)\.',filename).groups()[0]
    prec = int(re.match(r'.*\.fp_(.*?)\.',filename).groups()[0])
    result["Precision"] = ("FP16" if prec==16 else "FP32")
    
    return result

def import_nsight_metric(filename, cuda_dir='/usr/local/cuda'):
    #execute nvprof and parse file
    args = [os.path.join(cuda_dir, "bin/nv-nsight-cu-cli"),"--csv","-i",filename]
    #skiprows = 2
        
    #open subprocess and communicate
    p = sp.Popen(args, stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = p.communicate()
    
    #get timeline from csv
    profiledf = pd.read_csv(StringIO(stdout.decode("utf-8")),skiprows=0) #.dropna(how="all").rename(columns={"Kernel": "Name"})
    
    #clean up
    del profiledf["Process ID"]
    del profiledf["Process Name"]
    del profiledf["Host Name"]
    del profiledf["Kernel Time"]
    del profiledf["Context"]
    #del profiledf["Stream"]
    del profiledf["Section Name"]
    
    profiledf.rename(columns={"Kernel Name": "Name"}, inplace=True)
    
    #return result
    return profiledf