
import os
import shutil

def make_directory(path):

    print(path)
    if os.path.exists(path) == False:
        os.mkdir(path)

    else:
        #os.rmdir(path)        
        #shutil.rmtree(path)
        pass

def remove_files(directory):

    if os.path.exists(directory) == True:
        shutil.rmtree(directory)
