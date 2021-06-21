# https://ortegatron.medium.com/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e

from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import logging
from utils import load_json_arr
import numpy as np
import os
import matplotlib.pyplot as plt

class PlotLossHook(HookBase):
    """
    Hook to include validation loss on the TensorBoard (and in evaluation)
    """
    def __init__(self, cfg):
        self._period = cfg.TEST.EVAL_PERIOD
        self._output_dir = cfg.OUTPUT_DIR

    def _plot_losses(self):
        logging.log(logging.INFO, "Beginning plotting procedure")
        metrics = load_json_arr(os.path.join(self._output_dir, "metrics.json"))
        tot = []
        val = []
        iters = []
        for line in metrics:
            if "validation_loss" in line and "total_loss" in line and "iteration" in line:
                tot.append(line["total_loss"])
                val.append(line["validation_loss"])
                iters.append(line["iteration"])

        if len(tot) > 0 and len(val) > 0 and len(iters) > 0:
            plt.figure(999)
            plt.clf()

            plt.plot(iters, tot, "r-", label="total loss")
            plt.plot(iters, val, "b-", label="val loss")
            plt.title(f"Total and Validation Loss epochs {min(iters)} to {max(iters)}")
            plt.legend()

            plt.savefig(os.path.join(self._output_dir, "val_v_tot_loss.png"), dpi=100)

            plt.show(block=False)
        else:
            logging.log(logging.WARNING, "No info to plot, continuing with training")


    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._plot_losses()
