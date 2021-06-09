from detectron2.engine import DefaultPredictor
from torchvision.ops import nms

class Predictor(DefaultPredictor):
    """
    Override default predictor to run nms across set classes
    """
    def __init__(self, cfg, cross:list):
        super().__init__(cfg)
        self.cross = cross

    def __call__(self, img):
        preds = super().__call__(img)
        # run nms on new preds
        inst = preds["instances"]

        inds = nms(inst.pred_boxes.tensor, inst.scores, self.cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS[0])



        return preds
