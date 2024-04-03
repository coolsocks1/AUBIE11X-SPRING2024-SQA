'''
Akond Rahman 
Nov 20, 2020 
Friday 
'''
import os 
import numpy as np 
import ast 
import constants 

PY_FILE_EXTENSION = '.py'
NAME_KW = 'name'
NAMES_KW = 'names' 
LOGGING_KW = 'logging'

def checkIfParsablePython( pyFile ):
	flag = True 
	try:
		full_tree = ast.parse( open( pyFile ).read())    
	except (SyntaxError, UnicodeDecodeError) as err_ :
		flag = False 
	return flag 	

def getAllPythonFilesinRepo(path2dir):
	valid_list = []
	for root_, dirnames, filenames in os.walk(path2dir):
		for file_ in filenames:
			full_path_file = os.path.join(root_, file_) 
			if( os.path.exists( full_path_file ) ):
				if (file_.endswith( PY_FILE_EXTENSION ) and (checkIfParsablePython( full_path_file ) )   ):
					valid_list.append(full_path_file) 
	valid_list = np.unique(  valid_list )
	return valid_list

def hasLogImport( file_ ):
    IMPORT_FLAG = False 
    tree_object = ast.parse( open( file_ ).read())    
    for stmt_ in tree_object.body:
        for node_ in ast.walk(stmt_):
            if isinstance(node_, ast.Import) :
                funcDict = node_.__dict__     
                # print(funcDict) 
                import_name_objects = funcDict[NAMES_KW]
                for obj in import_name_objects:
                    if ( LOGGING_KW in  obj.__dict__[NAME_KW]): 
                        IMPORT_FLAG = True 
    return IMPORT_FLAG     


def commonAttribCallBody(node_):
    full_list = []
    if isinstance(node_, ast.Call):
        funcDict = node_.__dict__ 
        func_, funcArgs, funcLineNo, funcKeys =  funcDict[ constants.FUNC_KW ], funcDict[constants.ARGS_KW], funcDict[constants.LINE_NO_KW], funcDict[constants.KEY_WORDS_KW]  
        if( isinstance(func_, ast.Attribute ) ):
            func_as_attrib_dict = func_.__dict__ 
            #print(func_as_attrib_dict ) 
            func_name    = func_as_attrib_dict[constants.ATTRIB_KW] 
            func_parent  = func_as_attrib_dict[constants.VALUE_KW]
            
            if( isinstance(func_parent, ast.Name ) ):     
                call_arg_list = []   
                index = 0                
                for x_ in range(len(funcArgs)):
                	index = x_ + 1
                	funcArg = funcArgs[x_] 
                	if( isinstance(funcArg, ast.Name ) )  :
                		call_arg_list.append( (  funcArg.id, constants.INDEX_KW + str(x_ + 1) )  ) 
                	elif( isinstance(funcArg, ast.Attribute) ): 
                		arg_dic  = funcArg.__dict__
                		arg_name = arg_dic[constants.ATTRIB_KW] 
                		call_arg_list.append( (  arg_name, constants.INDEX_KW + str(x_ + 1) )  ) 
                	elif(isinstance( funcArg, ast.Str ) ):
                	    call_arg_list.append( ( funcArg.s, constants.INDEX_KW + str( x_ + 1 )  ) )
                        
                for x_ in range(len(funcKeys)):
                	funcKey = funcKeys[x_] 
                	if( isinstance(funcKey, ast.keyword ) )  :
                		call_arg_list.append( ( funcKey.arg, constants.INDEX_KW + str( x_ + 1 + index ) ) ) 
                    			
                full_list.append( ( func_parent.id, func_name , funcLineNo, call_arg_list  ) )        
                
                
            if( isinstance(func_parent, ast.Attribute ) ):     
                call_arg_list = []   
                index = 0                
                for x_ in range(len(funcArgs)):
                	index = x_ + 1
                	funcArg = funcArgs[x_] 
                	if( isinstance(funcArg, ast.Name ) )  :
                	    call_arg_list.append( (  funcArg.id, constants.INDEX_KW + str(x_ + 1) )  ) 
                	elif( isinstance(funcArg, ast.Attribute) ): 
                	    arg_dic  = funcArg.__dict__
                	    arg_name = arg_dic[constants.ATTRIB_KW] 
                	    call_arg_list.append( (  arg_name, constants.INDEX_KW + str(x_ + 1) )  ) 
                	elif(isinstance( funcArg, ast.Str ) ):
                	    call_arg_list.append( ( funcArg.s, constants.INDEX_KW + str( x_ + 1 )  ) )
                        
                for x_ in range(len(funcKeys)):
                	funcKey = funcKeys[x_] 
                	if( isinstance(funcKey, ast.keyword ) )  :
                		call_arg_list.append( ( funcKey.arg, constants.INDEX_KW + str( x_ + 1 + index ) ) ) 
                		
                func_dic  = func_parent.__dict__
                func_parent_name = func_dic[constants.ATTRIB_KW] 
                full_list.append( ( func_parent_name, func_name , funcLineNo, call_arg_list  ) )    
                
            if( isinstance(func_parent, ast.Call ) ):     
                call_arg_list = []   
                index = 0                
                for x_ in range(len(funcArgs)):
                	index = x_ + 1
                	funcArg = funcArgs[x_] 
                	if( isinstance(funcArg, ast.Name ) )  :
                	    call_arg_list.append( (  funcArg.id, constants.INDEX_KW + str(x_ + 1) )  ) 
                	elif( isinstance(funcArg, ast.Attribute) ): 
                	    arg_dic  = funcArg.__dict__
                	    arg_name = arg_dic[constants.ATTRIB_KW] 
                	    call_arg_list.append( (  arg_name, constants.INDEX_KW + str(x_ + 1) )  ) 
                	elif(isinstance( funcArg, ast.Str ) ):
                		call_arg_list.append( ( funcArg.s, constants.INDEX_KW + str( x_ + 1 )  ) )
                        
                for x_ in range(len(funcKeys)):
                	funcKey = funcKeys[x_] 
                	if( isinstance(funcKey, ast.keyword ) )  :
                		call_arg_list.append( ( funcKey.arg, constants.INDEX_KW + str( x_ + 1 + index ) ) ) 
                		
                func_dic  = func_parent.__dict__
                func_parent_name = func_dic[constants.FUNC_KW] 
                if( isinstance(func_parent_name, ast.Name ) ):  
                	full_list.append( ( func_parent_name.id, func_name , funcLineNo, call_arg_list  ) )      
    return full_list      

def getPythonAtrributeFuncs(pyTree):
    '''
    detects func like class.funcName() 
    '''
    attrib_call_list  = [] 
    for stmt_ in pyTree.body:
        for node_ in ast.walk(stmt_):
            if isinstance(node_, ast.Call):
                attrib_call_list =  attrib_call_list + commonAttribCallBody( node_ )

    return attrib_call_list 

def getLogStatements( pyFile ): 
    tree_object = ast.parse( open( pyFile ).read())    
    func_decl_list = getPythonAtrributeFuncs(tree_object)
    for func_decl_ in func_decl_list:
        func_parent_id, func_name , funcLineNo, call_arg_list = func_decl_ # the class in which the method belongs, func_name, line no, arg_list 
        if ( LOGGING_KW in func_parent_id ) or ( LOGGING_KW in func_name) : 
            for arg_ in call_arg_list:       
                print(func_parent_id, func_name, call_arg_list, arg_)   

def printLogOps(repo_path):
    valid_py_files = getAllPythonFilesinRepo( repo_path ) 
    log_py_files   = [x_ for x_ in valid_py_files if hasLogImport(  x_ ) ]
    for py_file in log_py_files:
        print(py_file)
        print(getLogStatements( py_file ) )
        print('='*50)



if __name__=='__main__':
    repo_path = '/Users/arahman/FSE2021_ML_REPOS/MODELZOO/'
    # printLogOps( repo_path )