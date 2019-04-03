import os
import sys
import time

t1 = time.time()

filename = sys.argv[1]

if len(sys.argv) == 3:
    video = sys.argv[2]
    if video == "video":
        os.system("python video.py " + filename)
        filename = "video/frame30.png";

os.system("python A3/proposedAlgorithm.py " + filename)
os.system("python A3/detect_shapes.py -i A3/outputs/ye.png --orig " + filename)
os.system("python A3/text_recognition.py --east A3/frozen_east_text_detection.pb --image A3/outputs/ye.png")
os.system("python A3/text_recognition.py --east A3/frozen_east_text_detection.pb --image " + filename)

t2 = time.time()
time = t2-t1
print(str(time) + " seconds")