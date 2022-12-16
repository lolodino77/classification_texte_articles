import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
print("liste des path =")
for e in sys.path:
    print(e)
from lib_classification import *