import matplotlib.pyplot as plt
def plot(data, filename,do:int=0):
    pass
    if do:
        plt.figure()
        plt.plot(data)
        plt.savefig(filename)
        plt.close()