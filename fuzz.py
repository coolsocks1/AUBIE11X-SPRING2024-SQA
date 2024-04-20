import random 
import os
import py_parser 
import constants 
import numpy as np 

def deleteRepo(dirName, type_):
    print(':::' + type_ + ':::Deleting ', dirName)
    try:
        if os.path.exists(dirName):
            shutil.rmtree(dirName)
    except OSError:
        print('Failed deleting, will try manually') 

def dumpContentIntoFile(strP, fileP):
    fileToWrite = open( fileP, 'w')
    fileToWrite.write(strP )
    fileToWrite.close()
    return str(os.stat(fileP).st_size)

def makeChunks(the_list, size_):
    for i in range(0, len(the_list), size_):
        yield the_list[i:i+size_]

def getPythonCount(path2dir): 
    usageCount = 0
    for root_, dirnames, filenames in os.walk(path2dir):
        for file_ in filenames:
            full_path_file = os.path.join(root_, file_) 
            if (file_.endswith('py') ):
                usageCount +=  1 
    return usageCount 

def getAllPythonFilesinRepo(path2dir):
	valid_list = []
	for root_, dirnames, filenames in os.walk(path2dir):
		for file_ in filenames:
			full_path_file = os.path.join(root_, file_) 
			if( os.path.exists( full_path_file ) ):
				if (file_.endswith( constants.PY_FILE_EXTENSION ) and (py_parser.checkIfParsablePython( full_path_file ) )   ):
					valid_list.append(full_path_file) 
	valid_list = np.unique( valid_list )
	return valid_list

def simpleFuzzer1(): 
    dirName = ''
    type_ = True
    print(f"dirName: {dirName}")
    print(f"type_: {type_}")
    try:
       deleteRepo(dirName, type_)  
    except Exception as e:
       print(f"{e}")

def simpleFuzzer2():
   strP = True
   fileP = 0
   print(f"strP: {strP}")
   print(f"fileP: {fileP}")
   try:
      dumpContentIntoFile(strP, type_)
   except Exception as e:
      print(f"{e}")

def simpleFuzzer3():
   the_list = []
   size_ = "5"
   print(f"the_list: {the_list}")
   print(f"size_: {size_}")
   try:
      makeChunks(value)
   except Exception as e:
      print(f"{e}")

def simpleFuzzer4():
   path2dir = ["1", 2, 3]
   print(data)
   try:
      getPythonCount(data)
   except Exception as e:
      print(f"{e}")

def simpleFuzzer5():
   path2dir = [1, 2, 3]
   print(path2dir)
   try:
      getAllPythonFilesinRepo(data)
   except Exception as e:
      print(f"{e}")

if __name__=='__main__':
    simpleFuzzer1()
    simpleFuzzer2()
    simpleFuzzer3()
    simpleFuzzer4()
    simpleFuzzer5()
