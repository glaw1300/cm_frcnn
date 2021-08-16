from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import PeriodicWriter
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper, build_detection_train_loader
from lossevalhook import LossEvalHook
from plotlosshook import PlotLossHook
import os

class Trainer(DefaultTrainer):
    """
    Override the default trainer to include validation loss on the TensorBoard (and in eval)
    """
    def __init__(self, cfg, plot=True):
        self._plot = plot
        super().__init__(cfg)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")

        return COCOEvaluator(dataset_name, tasks=("bbox",), distributed=True, output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        augs = [T.RandomFlip(),
                T.RandomFlip(horizontal=False, vertical=True),
                T.RandomSaturation(.8, 1.2),
                T.RandomCrop("relative", (.5,.5)),
                T.RandomBrightness(.8, 1.2),
                T.RandomContrast(.8, 1.2)]
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=augs))

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))

        if self._plot:
            hooks.insert(-1, PlotLossHook(
                self.cfg
            ))

        #build hook for Periodic Writing losses and stuff, each n periods
        hooks[-1] = PeriodicWriter(self.build_writers(), period=20)

        return hooks
