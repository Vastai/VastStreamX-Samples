#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
from common.resnet import Resnet, vsx

attr = vsx.AttrKey

class DocImgOrientClassifier(Resnet):
    def __init__(
        self, model_prefix, vdsp_config, batch_size=1, device_id=0, hw_config=""
    ) -> None:
        super().__init__(model_prefix, vdsp_config, batch_size, device_id, hw_config)

    def process_impl(self, input):
        params = {"crop_resize_ext": []}
        for inp in input:
            resize_w, resize_h = self.get_resize(
                inp.width, inp.height, self.resize_size_
            )
            crop_x, crop_y = self.get_crop(resize_w, resize_h, self.model_size_)
            params["crop_resize_ext"].append((crop_x, crop_y, resize_w, resize_h))
        outputs = self.stream_.run_sync(input, params)
        return self.postprocess(outputs)
       
    def postprocess(self, outputs):
        final_outputs = []
        for output in outputs:
            output = vsx.as_numpy(output[0])
            index = np.argmax(output, axis=1)[0]
            score = output[0][index]
            final_outputs.append((index, score))
        return final_outputs

        