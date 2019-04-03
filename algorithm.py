import os
import sys
filename = sys.argv[1]
os.system("python proposedAlgorithm.py " + filename)
os.system("python detect_shapes.py -i ye.png")
os.system("python A3/text_recognition.py --east A3/frozen_east_text_detection.pb --image ye.png")
os.system("python A3/text_recognition.py --east A3/frozen_east_text_detection.pb --image " + filename)