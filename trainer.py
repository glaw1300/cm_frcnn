from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import PeriodicWriter
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetMapper, build_detection_test_loader
from lossevalhook import LossEvalHook
import os

class Trainer(DefaultTrainer):
    """
    Override the default trainer to include validation loss on the TensorBoard (and in eval)
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")

        return COCOEvaluator(dataset_name, tasks=("bbox",), distributed=True, output_dir=output_folder)

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

        #build hook for Periodic Writing losses and stuff, each n periods
        hooks[-1] = PeriodicWriter(self.build_writers(), period=1)

        return hooks
