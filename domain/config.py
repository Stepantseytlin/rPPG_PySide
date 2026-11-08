class Config:
    cad : int = 30
    fMa : int = 240
    fMi : int = 40
    segLength : int = 1024
    strapLen : int = 40
    SourceType : str = "txt"

    def __init__(self, cad=30, fMa=240, fMi=40, segLength=1024, straplen=20, instant: int = 0, source_type: str = "txt"):
        self.cad = cad
        self.fMa = fMa
        self.fMi = fMi
        self.segLength = segLength
        self.strapLen = straplen
        self.SourceType = source_type
        self.totalFrames = 0
        self.instant = instant
        self.plotSource = "PW"
        self.deriv = True
        self.Hilbert = True
    def getsegLength(self):
        return self.segLength

    def getSourceType(self):
        return self.SourceType

    def getstrapLen(self):
        return self.strapLen
