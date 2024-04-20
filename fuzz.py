import random 

def divide(v1, v2):
   return v1 / v2 

def getItem(data, index):
   element = data[index]
   return element

def absValue(number):
   if number < 0:
      raise ValueError("Input must be non-negative")
   return number

def sumList(numbers):
   total = 0
   for num in numbers:
      total += num
   return total

def isUpperCase(char):
   return char.isupper()

def fuzzValues(val1, val2):
   res = divide(val1, val2)
   return res  

def simpleFuzzer1(): 
    ls_ = ['123', 'True', 'False', [] , None, '/', '2e34r']
    for x in ls_:
      print(x)
      if isinstance(x, str):
         mod_x = x + str( random.randint(1, 10) )
      elif isinstance(x, int): 
         mod_x = x + random.random()
      try:
      	 fuzzValues( x, mod_x )  
      except Exception as e:
      	print(f"{e}")

def simpleFuzzer2():
   data = [1, 2, 3]
   index = 4
   print(data)
   print(index)
   try:
      getItem(data, index)
   except Exception as e:
      print(f"{e}")

def simpleFuzzer3():
   value = -4
   print(value)
   try:
      absValue(value)
   except Exception as e:
      print(f"{e}")

def simpleFuzzer4():
   data = ["1", 2, 3]
   print(data)
   try:
      sumList(data)
   except Exception as e:
      print(f"{e}")

def simpleFuzzer5():
   data = [1, 2, 3]
   print(data)
   try:
      isUpperCase(data)
   except Exception as e:
      print(f"{e}")

if __name__=='__main__':
    simpleFuzzer1()
    simpleFuzzer2()
    simpleFuzzer3()
    simpleFuzzer4()
    simpleFuzzer5()
