import os
import torch
import numpy as np
import yaml

from dover.datasets import (
    UnifiedFrameSampler,
    spatial_temporal_view_decomposition,
)
from dover.models import DOVER


class EvaluateRunner():
    def __init__(self, opt_path):
        with open(opt_path, "r") as f:
            self.opt = yaml.safe_load(f)
        self.mean= torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])
        self.model = self.load_model()

    def load_model(self):
        evaluator = DOVER(**self.opt["model"]["args"]).to("cuda")### Load DOVER
        evaluator.load_state_dict(
            torch.load(self.opt["test_load_path"], map_location="cuda")
        )

        return evaluator

    def fuse_results(self, results: list):
        ## results[0]: aesthetic, results[1]: technical
        t, a = (results[1] - 0.1107) / 0.07355, (results[0] + 0.08285) / 0.03774
        x = t * 0.6104 + a * 0.3896
        fuse_rst = {
            "aesthetic": 1 / (1 + np.exp(-a)),
            "technical": 1 / (1 + np.exp(-t)),
            "overall": 1 / (1 + np.exp(-x)),
        }

        return fuse_rst

    def run(self, clip_path):
        dopt = self.opt["data"]["val-l1080p"]["args"]

        temporal_samplers = {}
        for stype, sopt in dopt["sample_types"].items():
            if "t_frag" not in sopt:
                # resized temporal sampling for TQE in DOVER
                temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
                )
            else:
                # temporal sampling for AQE in DOVER
                temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"] // sopt["t_frag"],
                    sopt["t_frag"],
                    sopt["frame_interval"],
                    sopt["num_clips"],
                )

        ### View Decomposition
        views, _ = spatial_temporal_view_decomposition(
            clip_path, dopt["sample_types"], temporal_samplers
        )

        for k, v in views.items():
            num_clips = dopt["sample_types"][k].get("num_clips", 1)
            views[k] = (
                ((v.permute(1, 2, 3, 0) - self.mean) / self.std)
                .permute(3, 0, 1, 2)
                .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                .transpose(0, 1)
                .to("cuda")
            )


        evaluate_results = [r.mean().item() for r in self.model(views)]
        fuse_results = self.fuse_results(evaluate_results)

        return fuse_results

if __name__ == '__main__':
    evaluate_runner = EvaluateRunner("./dover.yml")
    fuse_results = evaluate_runner.run("/home/higher/hdisk/wangzepeng5/lm/GenData/output/v_ZZ71FIfxX-c_1/v_ZZ71FIfxX-c_1.mp4")
    print(fuse_results)