from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # set up color generator and colormap
    markers = ('s', 'o', 'x', 'v', '^')
    colors = ('red', 'blue', 'lightgreen', 'cyan', 'grey')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot decision surface
    