import matplotlib.pyplot as plt

if __name__ == '__main__':
    ROOT = 'resource/exps/laws/'
    with open(ROOT + 'actions.txt', 'r') as f:
        actions = [[int(x) for x in line.split()] for line in f]
        steps = range(len(actions))
        punishments = [[] for _ in range(9)]

        for action in actions:
            avg = 0
            for i in range(8):
                avg += action[i]
                punishments[i].append(action[i])
            punishments[8].append(avg / 8)

        for i in range(8):
            plt.plot(steps, punishments[i])
            plt.xlabel('steps')
            plt.ylabel('Divine punishment')
            plt.title('Nation ' + str(i))
            plt.savefig(ROOT + 'plots/nation_' + str(i) + '.png', )
            plt.clf()

        plt.plot(steps, punishments[8])
        plt.xlabel('steps')
        plt.ylabel('Divine punishment')
        plt.title('Nations')
        plt.savefig(ROOT + 'plots/nations.png')