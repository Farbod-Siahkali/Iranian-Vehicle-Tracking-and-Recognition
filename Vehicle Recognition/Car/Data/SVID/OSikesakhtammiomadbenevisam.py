import os
import glob
paths = glob.glob('./train/*/*.jpg')

for path in paths:
    os.replace(path, "./train/"+path.split('\\')[-1])
