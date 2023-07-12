import os
import sys
import subprocess
sys.path.append("./")

print(os.path.abspath(os.path.curdir))
classifierLists = os.listdir("Classifiers")
#classifierLists = ["MNIST"]
for classifier in classifierLists:
        subprocess.call("python CodeBook/MultiLabelClassification/predictClassifier.py -pd=Evaluation -ds=MNIST -md=Classifiers -thr=0".format(classifier), shell=True)