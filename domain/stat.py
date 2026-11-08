class Stat:
    def __init__(self):
        self.Pause = 1
        self.reset = 0
        self.frameready = 0
        self.freqready = 0
        self.pwready = 0
        self.roiinserted = 0
        self.threadrunning = 0
        self.wrapperoutputready = 0
        self.stopPlotting = 0 

    def clear_runtime_flags(self):
        self.frameready = 0
        self.freqready = 0
        self.pwready = 0
        self.roiinserted = 0
        self.threadrunning = 0
        self.wrapperoutputready = 0
