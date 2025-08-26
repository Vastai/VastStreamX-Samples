from .model_cv import ModelCV, vsx
import numpy as np

attr = vsx.AttrKey


class MobileVit(ModelCV):
    def __init__(
        self, model_prefix, vdsp_config, batch_size=1, device_id=0, hw_config=""
    ) -> None:
        super().__init__(model_prefix, vdsp_config, batch_size, device_id, hw_config)
        self.model_input_height, self.model_input_width = self.model_.input_shape[0][
            -2:
        ]
        self.resize_height = int(256.0 / 224 * self.model_input_height)
        self.fusion_op_ = None
        for op in self.preproc_ops_:
            if op.op_type >= 100:
                self.fusion_op_ = op.cast_to_buildin_operator()
                break
        assert self.fusion_op_ is not None, "Can't find fusion op in vdsp op json file"

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

    def compute_size(self, img_w, img_h, size):
        if isinstance(size, int):
            size_h, size_w = size, size
        elif len(size) < 2:
            size_h, size_w = size[0], size[0]
        else:
            size_h, size_w = size[-2:]

        r = max(size_w / img_w, size_h / img_h)

        new_w = int(r * img_w)
        new_h = int(r * img_h)
        return (new_w, new_h)

    def process_impl(self, inputs):
        outputs = []
        for input in inputs:
            resize_width, resize_height = self.compute_size(
                input.width, input.height, self.resize_height
            )
            left = (resize_width - self.model_input_width) // 2
            top = (resize_height - self.model_input_height) // 2
            self.fusion_op_.set_attribute(
                {
                    attr.IIMAGE_WIDTH: input.width,
                    attr.IIMAGE_HEIGHT: input.height,
                    attr.IIMAGE_WIDTH_PITCH: input.width,
                    attr.IIMAGE_HEIGHT_PITCH: input.height,
                    attr.RESIZE_WIDTH: resize_width,
                    attr.RESIZE_HEIGHT: resize_height,
                    attr.CROP_X: left,
                    attr.CROP_Y: top,
                }
            )
            model_outs = self.stream_.run_sync([input])[0]
            outputs.append(vsx.as_numpy(model_outs[0]).astype(np.float32))
        return outputs
