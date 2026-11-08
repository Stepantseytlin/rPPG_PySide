import cv2 as cv
class Frame_:
    value : cv.typing.MatLike
    def __init__(self, value = None):
        self.value = value
        pass