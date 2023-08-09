from torchvision.transforms.functional import invert, solarize, equalize


class Rotate(object):
    """Rotate PIL sample."""

    def __init__(self, rotation):
        self.rotation = rotation
    
    def __call__(self, sample):
        
        return sample.rotate(self.rotation)



class LabelFlip(object):
    """Flip labels of the dataset."""

    def __init__(self, shift):
        self.shift = shift

    def __call__(self, label):

        return (label-self.shift) % 10



class Invert(object):
    """Invert colors of PIL sample"""
    def __call__(self, sample):
        
        return invert(sample)


class Solarize(object):
    """Solarize PIL sample"""
    def __call__(self, sample):
        
        return solarize(sample, threshold=0)


class Equalize(object):
    """Solarize PIL sample"""
    def __call__(self, sample):
        
        return equalize(sample)