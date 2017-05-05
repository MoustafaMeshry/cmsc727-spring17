"""
Experiment driver
"""

import os
import glob
import shutil
import numpy as np
#import adaboost_sampling
from adaboost_sampling import runIMDBExperiment

kSamplingRatio = 0.7
kNEpochs = 3
kBoostIters = 10
kDropoutProb = 0.4
nH = 32

runSamplingExp = False
runEpochsExp = False
runArchExp = False
runDropoutExp = False
baselineExp = False
runSingleExp = True

if runSamplingExp:
    for kSamplingRatio in np.arange(.1, 1.05, 0.1):
        runIMDBExperiment(kSamplingRatio, kNEpochs, kBoostIters, kDropoutProb, nH);
        expPath = os.path.join("models", "exp_arch" + str(nH) + "_sampling=" + str(kSamplingRatio))
        os.makedirs(expPath)

        for myFile in glob.glob(r'models/*tfl*'):
            shutil.move(myFile, expPath)
        shutil.move('models/checkpoint', expPath)
        shutil.move('models/modelsMeta.pckl', expPath)
        shutil.move(glob.glob(r'models/results*')[0], expPath)

# =================================================================

if runEpochsExp:
    for kNEpochs in range(1, 8):
#    for kNEpochs in range(5, 6):
        runIMDBExperiment(kSamplingRatio, kNEpochs, kBoostIters, kDropoutProb, nH);
        expPath = os.path.join("models", "exp_epochs=" + str(kNEpochs))
        os.makedirs(expPath)

        for myFile in glob.glob(r'models/*tfl*'):
            shutil.move(myFile, expPath)
        shutil.move('models/checkpoint', expPath)
        shutil.move('models/modelsMeta.pckl', expPath)
        shutil.move(glob.glob(r'models/results*')[0], expPath)

# =================================================================

if runArchExp:
    for logNH in range(1, 8):
#    for logNH in range(4, 8):
        nH = 1<<logNH;
        runIMDBExperiment(kSamplingRatio, kNEpochs, kBoostIters, kDropoutProb, nH);
        expPath = os.path.join("models", "exp_arch=" + str(nH))
        os.makedirs(expPath)

        for myFile in glob.glob(r'models/*tfl*'):
            shutil.move(myFile, expPath)
        shutil.move('models/checkpoint', expPath)
        shutil.move('models/modelsMeta.pckl', expPath)
        shutil.move(glob.glob(r'models/results*')[0], expPath)

# =================================================================

if baselineExp:
    runIMDBExperiment(1, kNEpochs, 1, kDropoutProb, 128)
    expPath = os.path.join("models", "baseline")
    os.makedirs(expPath)

    for myFile in glob.glob(r'models/*tfl*'):
        shutil.move(myFile, expPath)
    shutil.move('models/checkpoint', expPath)
    shutil.move('models/modelsMeta.pckl', expPath)
    shutil.move(glob.glob(r'models/results*')[0], expPath)

# =================================================================

if runDropoutExp:
    for kDropoutProb in np.arange(0.1, 1.05, 0.1):
        runIMDBExperiment(kSamplingRatio, kNEpochs, kBoostIters, kDropoutProb, nH);
        expPath = os.path.join("models", "exp_dropout=" + str(kDropoutProb))
        os.makedirs(expPath)

        for myFile in glob.glob(r'models/*tfl*'):
            shutil.move(myFile, expPath)
        shutil.move('models/checkpoint', expPath)
        shutil.move('models/modelsMeta.pckl', expPath)
        shutil.move(glob.glob(r'models/results*')[0], expPath)

# =================================================================

if runSingleExp:
    runIMDBExperiment(kSamplingRatio, kNEpochs, kBoostIters, kDropoutProb, nH)
    expPath = os.path.join("models", "baseline")
    os.makedirs(expPath)

    for myFile in glob.glob(r'models/*tfl*'):
        shutil.move(myFile, expPath)
    shutil.move('models/checkpoint', expPath)
    shutil.move('models/modelsMeta.pckl', expPath)
    shutil.move(glob.glob(r'models/results*')[0], expPath)

# =================================================================

# Remove checkpoints
trashPath = os.path.join('models', 'check_point_trash')
os.mkdir(trashPath)
for myFile in glob.glob(r'models/chkPt*'):
    shutil.move(myFile, trashPath)
shutil.rmtree(trashPath)

metaDataFile = os.path.join("models", "2nd set of experiments", "exp_arch=2", "modelsMeta.pckl")
with open(metaDataFile, 'rb') as f:
	metaData = pickle.load(f)

