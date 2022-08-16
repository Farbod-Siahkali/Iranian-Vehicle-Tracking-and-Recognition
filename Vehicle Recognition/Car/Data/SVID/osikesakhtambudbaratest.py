import os
import glob
paths = glob.glob('./test/*/*.jpg')

for path in paths:
    os.replace(path, "./test/"+path.split('\\')[-1])
