


class Rotate(object):
    """Rotate PIL sample."""

    def __init__(self, rotation):
        self.rotation = rotation
    
    def __call__(self, sample):
        
        return sample.rotate(self.rotation)



class LabelFlip(object):
    """Flip labels of the dataset."""

    def __call__(self, label):

        return (label-1) % 10