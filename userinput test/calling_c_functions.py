from ctypes import *



mydll = cdll.LoadLibrary(r"C:\Users\18014238\PycharmProjects\pythonProject\C Project\algo.dll")
result = mydll.main()
