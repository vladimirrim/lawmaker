import numpy as np
import pickle
from os import mkdir
from os.path import exists

from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt

from forge.blade.lib.enums import Neon


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div


def readLmLogs():
    with open(logDir + name + logName, 'rb') as f:
        logs = pickle.load(f)
    p, v, s, r = [], [], [], []
    for log in logs:
        p += log[0]
        v += log[1]
        s += log[2]
        r += log[3]
    return p, v, s, r


def filterByPunishments(p, v, s, r, minP=0.9, maxP=1):

    p = np.array(p)
    v = np.array(v)
    s = np.array(s)
    r = np.array(r)

    mask = ((p >= minP) & (p <= maxP)).reshape((-1,))
    p = p[mask]
    v = v[mask]
    s = s[mask]
    r = r[mask]
    return p, v, s, r


def cluster(r):

    # choose clustering model and parameters
    model, nm = KMeans(n_clusters=10, random_state=42, n_jobs=4), 'kmeans/'
    # model, nm = DBSCAN(eps=0.1, min_samples=5, n_jobs=4), 'dbscan/'

    model = model.fit(r)
    return model.labels_, nm


def createPic(state, label, fig, pic_num, dr):
    indicies = state.conv[0, :, :].detach().numpy()
    ents = state.ents.detach().numpy()
    policy = softmax(state.policy.detach().numpy())[0]

    for i in range(indicies.shape[0]):
        for j in range(indicies.shape[1]):
            idx = indicies[i, j]
            plt.plot([i + 0.5], [j + 0.5], linestyle="None", marker="s",
                     markersize=37, mfc=colors[idx], mec='black', zorder=0)

    for ent in ents:
        delta = (ent[6], ent[7])
        if (delta[0] != 0) and (delta[1] != 0):
            friend = int(np.ceil(ent[9]))
            plt.plot([7.5 - delta[0]], [7.5 - delta[1]], linestyle="None", marker="o",
                     markersize=15, mfc=colorsEnt[friend], mec='black', zorder=1)

    plt.plot([7.5], [7.5], linestyle="None", marker="o",
                     markersize=10, mfc=Neon.BLACK.norm, mec='black', zorder=1)

    for i, delta in enumerate(((0, 1), (1, 0), (0, -1), (-1, 0))):
        plt.arrow(7.5, 7.5, delta[0], -delta[1], width=policy[i+1]*0.1, length_includes_head=True,
                  color='black', zorder=2)
        plt.text(7.5-0.25 + delta[0]/2 + abs(delta[1]/4), 7.5 + abs(delta[0]/10) - delta[1]/2, round(policy[i+1], 2),
                 zorder=2, fontsize=10)

    plt.xticks(())
    plt.yticks(())
    plt.xlim((0, 15))
    plt.ylim((0, 15))

    if not exists(dr + 'cluster_' + str(label)):
        mkdir(dr + 'cluster_' + str(label))
    plt.savefig(dr + 'cluster_' + str(label) + '/' + str(pic_num) + '.png', dpi=100)
    plt.clf()


def createPics(states, labels):
    dr = logDir + name + saveFolder
    if not exists(dr):
        mkdir(dr)
    dr = dr + clusterName
    if not exists(dr):
        mkdir(dr)
    fig = plt.figure(figsize=(10, 10))
    for pic_num, (state, label) in enumerate(zip(states, labels)):
        createPic(state, label, fig, pic_num, dr)


if __name__ == '__main__':

    logDir = 'resource/exps/'
    name = 'exploreIntensifiesAuto'
    logName = '/model/logsLm.p'
    saveFolder = '/clustering/'

    # reading logs
    p, v, s, r = readLmLogs()  # punishments, values, states, representations
    p, v, s, r = filterByPunishments(p, v, s, r, minP=0.9, maxP=1)

    # clustering
    labels, clusterName = cluster(r)
    print('{}: {} clusters'.format(clusterName, len(np.unique(labels))))

    # creating pictures
    colors = {0: Neon.ORANGE.norm,  # lava
              1: Neon.BLUE.norm,  # water
              2: Neon.WHITE.norm,  # grass
              3: Neon.YELLOW.norm,  # scrub
              4: Neon.DARKGREEN.norm,  # forest
              5: Neon.GRAY.norm,  # stone
              6: Neon.BLACK.norm,  # iron_ore
              }
    colorsEnt = {0: Neon.RED.norm,  # other color
                 1: Neon.GREEN.norm,  # same color
                 }
    createPics(s, labels)

    # creating TSNE plot
