import subprocess
import os


### NOT WORKING. NO IDEA WHY. SEEMS LIKE UNCONVERTABLE MODELS RUIN EVERYTHING.
def _getPbFile(files):
    for file in files:
        if file.endswith('pb'):
            return file

    return -1

currentDir = os.getcwd()
modelDir = '/home/jtang10/data/carml/dlframework/tensorflow_1.12/'
modelList = os.listdir(modelDir)

os.chdir(os.path.join(modelDir, '..'))
if not os.path.exists('uff'):
    os.mkdir('uff')
exportDir = os.path.join(modelDir, 'uff')

stdout = open('stdout.txt', 'w+')
stderr = open('stderr.txt', 'w+')
for modelName in modelList:
    curDir = os.path.join(modelDir, modelName)
    files = os.listdir(curDir)
    frozenGraph = _getPbFile(files)
    if frozenGraph == -1:
        continue
    frozenGraph = os.path.join(modelDir, modelName, frozenGraph)
    uffName = os.path.join(exportDir, modelName + '.uff')
    if not os.path.exists(uffName):
        print("creating", uffName)
        conversion = ['convert-to-uff', frozenGraph, '-o', uffName]
        subprocess.run(conversion, stdout=stdout, stderr=stderr)
    else:
        print(uffName, 'exists!!! skipping it.')
