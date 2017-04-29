"""
Experiment driver
"""

import os
import glob
import shutil
import numpy as np
#import adaboost_sampling
from adaboost_sampling import runIMDBExperiment

kSamplingRatio = 0.25
kNEpochs = 5
kBoostIters = 3
kDropoutProb = 0.8
nH = 4

runSamplingExp = False
runEpochsExp = False
runArchExp = False

if runSamplingExp:
    for kSamplingRatio in np.arange(.1, 1.05, 0.1):
    #    cmd = "python adaboost_sampling.py" + " " + str(kSamplingRatio) + " " + str(kNEpochs) + " " + str(nH)
    #    os.system(cmd)
        runIMDBExperiment(kSamplingRatio, kNEpochs, kBoostIters, kDropoutProb, nH);
        expPath = os.path.join("models", "exp_arch4_sampling=" + str(kSamplingRatio))
        os.makedirs(expPath)

        for myFile in glob.glob(r'models/*tfl*'):
            shutil.move(myFile, expPath)
        shutil.move('models/checkpoint', expPath)
        shutil.move('models/modelsMeta.pckl', expPath)
        shutil.move(glob.glob(r'models/results*')[0], expPath)

# =================================================================

if runEpochsExp:
    for kNEpochs in range(1, 8):
    #for kNEpochs in range(1, 3):
    #    cmd = "python adaboost_sampling.py" + " " + str(kSamplingRatio) + " " + str(kNEpochs)
    #    os.system(cmd)
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
    #for logNH in range(4, 8):
    #    cmd = "python adaboost_sampling.py" + " " + str(kSamplingRatio) + " " + str(kNEpochs) + " " + str(1<<logNH)
    #    os.system(cmd)
        nH = 1<<logNH;
        runIMDBExperiment(kSamplingRatio, kNEpochs, kBoostIters, kDropoutProb, nH);
        expPath = os.path.join("models", "exp_arch=" + str(nH))
        os.makedirs(expPath)

        for myFile in glob.glob(r'models/*tfl*'):
            shutil.move(myFile, expPath)
        shutil.move('models/checkpoint', expPath)
        shutil.move('models/modelsMeta.pckl', expPath)
        shutil.move(glob.glob(r'models/results*')[0], expPath)

