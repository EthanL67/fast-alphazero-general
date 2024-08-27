import pyximport; pyximport.install()

from torch import multiprocessing as mp

from Coach import Coach
from NNetWrapper import NNetWrapper as nn
from tangled.TangledGame import TangledGame as Game
from tangled.TangledVariants import *
from utils import *
import os
import warnings
import logging
from numba import NumbaPerformanceWarning

GAME_VARIANT = "Q5"  # K3, K4, P, Q5

args = dotdict({
    'run_name': 'tangled_' + GAME_VARIANT,
    'workers': mp.cpu_count() - 1,
    'startIter': 1,
    'numIters': 1000,
    'process_batch_size': 8,
    'train_batch_size': 128,
    'train_steps_per_iteration': 500,
    # should preferably be a multiple of process_batch_size and workers
    'gamesPerIteration': 1*(mp.cpu_count()-1),
    'numItersForTrainExamplesHistory': 10,
    'symmetricSamples': True,
    'numMCTSSims': 100,
    'numFastSims': 10,
    'probFastSim': 0.75,
    'tempThreshold': 10,
    'temp': 1,
    'compareWithRandom': True,
    'arenaCompareRandom': 100,
    'arenaCompare': 100,
    'arenaTemp': 0.1,
    'arenaMCTS': False,
    'randomCompareFreq': 1,
    'compareWithPast': True,
    'pastCompareFreq': 3,
    'expertValueWeight': dotdict({
        'start': 0,
        'end': 0,
        'iterations': 35
    }),
    'cpuct': 3,
    'checkpoint': 'checkpoint/' + GAME_VARIANT,
    'data': 'data/' + GAME_VARIANT,
    'game_variant': GAME_VARIANT,
})

os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = '1'
logging.getLogger('numba').setLevel(logging.ERROR)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

if __name__ == "__main__":
    mp.set_start_method('spawn')

    g = Game(args.game_variant)
    nnet = nn(g)
    c = Coach(g, nnet, args)
    c.learn()
