class Detection:

    def __init__(self, box, conf, cls, label):
        self.box = box
        self.conf = conf
        self.cls = cls
        self.label = label

    