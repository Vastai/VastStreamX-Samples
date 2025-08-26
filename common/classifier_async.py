#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from .model_cv_async import ModelCVAsync, vsx
import numpy as np

attr = vsx.AttrKey


class ClassifierAsync(ModelCVAsync):
    def __init__(
        self, model_prefix, vdsp_config, batch_size=1, device_id=0, hw_config=""
    ) -> None:
        super().__init__(model_prefix, vdsp_config, batch_size, device_id, hw_config)
        width, height = self.get_fusion_op_resize()
        self.resize_size_ = width if width > height else height
        self.model_size_ = self.input_shape[0][-1]

    def get_test_data(self, dtype, input_shape, batch_size, ctx="VACC"):
        assert len(input_shape) >= 2
        height = input_shape[-2]
        width = input_shape[-1]
        dummy = np.zeros((height, width, 3), dtype=dtype)
        if ctx == "CPU":
            return [dummy] * batch_size
        else:
            device_dummy = vsx.create_image(
                dummy, vsx.ImageFormat.BGR_INTERLEAVE, width, height, self.device_id_
            )
            return [device_dummy] * batch_size

    def get_fusion_op_resize(self):
        width, height = -1, -1
        for op in self.preproc_ops_:
            if op.op_type >= 100:
                buildin_op = op.cast_to_buildin_operator()
                if "RESIZE_WIDTH" in list(buildin_op.attributes.keys()):
                    width = buildin_op.get_attribute(attr.RESIZE_WIDTH)
                if "RESIZE_HEIGHT" in list(buildin_op.attributes.keys()):
                    height = buildin_op.get_attribute(attr.RESIZE_HEIGHT)
        assert (
            width > -1 and height > -1
        ), f"Failed to get resize_width resize_height in fusion_op"
        return (width, height)

    def get_resize(self, img_w, img_h, model_size):
        if img_w > img_h:
            resize_w = model_size * img_w // img_h
            resize_h = model_size
        else:
            resize_w = model_size
            resize_h = model_size * img_h // img_w
        return (resize_w, resize_h)

    def get_crop(self, img_w, img_h, model_size):
        crop_x = (img_w - model_size + 1) // 2
        crop_y = (img_h - model_size + 1) // 2
        return (crop_x, crop_y)

    def get_test_data(self, dtype, input_shape, batch_size, ctx="VACC"):
        assert len(input_shape) >= 2
        height = input_shape[-2]
        width = input_shape[-1]
        dummy = np.zeros((height, width, 3), dtype=dtype)
        if ctx == "CPU":
            return [dummy] * batch_size
        else:
            device_dummy = vsx.create_image(
                dummy, vsx.ImageFormat.BGR_INTERLEAVE, width, height, self.device_id_
            )
            return [device_dummy] * batch_size

    def process_async_impl(self, input):
        params = {"crop_resize_ext": []}
        for inp in input:
            resize_w, resize_h = self.get_resize(
                inp.width, inp.height, self.resize_size_
            )
            crop_x, crop_y = self.get_crop(resize_w, resize_h, self.model_size_)
            params["crop_resize_ext"].append((crop_x, crop_y, resize_w, resize_h))
        return self.stream_.run_async(input, params)

    def get_output(self):
        outputs = self.stream_.get_operator_output(self.model_op_)
        return [vsx.as_numpy(out[0]) for out in outputs]
