


class Rotate(object):
    """Rotate PIL sample."""

    def __init__(self, rotation):
        self.rotation = rotation
    
    def __call__(self, sample):
        
        return sample.rotate(self.rotation)