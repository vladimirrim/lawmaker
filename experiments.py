from pdb import set_trace as T
from configs import Law, Chaos, Kite
import os

# Oversimplified user specification
# for 1-3 collaborators
USER = 'vladimir'
if USER == 'vladimir':
    # Thousandth
    prefix = 'test'
    remote = False
    local = not remote

    test = False  # local
    best = False  # local
    load = False  # local

    sample = not test
    singles = True
    tournaments = False

    exps = {}
    szs = [128]
    # For full distributed runs
    # szs = (16, 32, 64, 128)
    names = 'law chaos'.split()
    confs = (Law, Chaos, Kite)


    def makeExp(name, conf, sz, test=False):
        NENT, NPOP = sz, sz // 16
        ROOT = 'resource/exps/' + name + '/'
        try:
            os.mkdir(ROOT)
            os.mkdir(ROOT + 'model')
            os.mkdir(ROOT + 'train')
            os.mkdir(ROOT + 'test')
        except FileExistsError:
            pass
        MODELDIR = ROOT + 'model/'

        exp = conf(remote,
                   NENT=NENT, NPOP=NPOP,
                   MODELDIR=MODELDIR,
                   SAMPLE=sample,
                   BEST=best,
                   LOAD=load,
                   TEST=test)
        exps[name] = exp
        print(name, ', NENT: ', NENT, ', NPOP: ', NPOP)


    def makeExps():
        # Training runs
        for label, conf in zip(names, confs):
            for sz in szs:
                name = prefix + label + str(sz)
                makeExp(name, conf, sz, test=test)


    # Sample config
    makeExps()
    makeExp('exploreIntensifies', Law, 128, test=False)
