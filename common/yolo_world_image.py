#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import vaststreamx as vsx
import common.utils as utils
import numpy as np

attr = vsx.AttrKey


class YoloWorldImage:
    def __init__(
        self, model_prefix, vdsp_config, batch_size=1, device_id=1, hw_config=""
    ):
        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0

        self.model_ = vsx.Model(model_prefix, batch_size, hw_config)
        self.model_op_ = vsx.ModelOperator(self.model_)
        self.graph_ = vsx.Graph(vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_NCHW_DEVICE)
        self.graph_.add_operators(self.model_op_)
        self.stream_ = vsx.Stream(self.graph_, vsx.StreamBalanceMode.RUN)
        self.stream_.register_operator_output(self.model_op_)
        self.stream_.build()

        self.input_shape_ = self.model_.input_shape[0]
        self.preproc_ops_ = vsx.Operator.load_ops_from_json_file(vdsp_config)
        self.fusion_op_ = self.preproc_ops_[0].cast_to_buildin_operator()
        self.oimage_height_, self.oimage_width_ = self.input_shape_[-2:]
        self.fusion_op_.set_attribute(
            {
                attr.OIMAGE_WIDTH: self.oimage_width_,
                attr.OIMAGE_HEIGHT: self.oimage_height_,
            }
        )

    @property
    def input_shape(self):
        return self.model_.input_shape

    def get_fusion_op_iimage_format(self):
        for op in self.preproc_ops_:
            if op.op_type >= 100:
                buildin_op = op.cast_to_buildin_operator()
                if "IIMAGE_FORMAT" in list(buildin_op.attributes.keys()):
                    imagetype = buildin_op.get_attribute(attr.IIMAGE_FORMAT)
                    return utils.imagetype_to_vsxformat(imagetype)
                else:
                    return vsx.ImageFormat.YUV_NV12
        assert False, "Can't find fusion op that op_type >= 100"

    def get_test_data(self, dtype, input_shape, batch_size, ctx="VACC"):
        assert len(input_shape) >= 2
        height = input_shape[-2]
        width = input_shape[-1]
        dummy_image = np.zeros((height, width, 3), dtype=dtype)
        dummy_txt = np.zeros((1203, 512), dtype=np.float32)
        dummy_txt = utils.bert_get_activation_fp16_A(dummy_txt)
        if ctx == "CPU":
            return ([dummy_image] * batch_size, [dummy_txt] * batch_size)
        else:
            vacc_image = vsx.create_image(
                dummy_image,
                vsx.ImageFormat.BGR_INTERLEAVE,
                width,
                height,
                self.device_id_,
            )
            vacc_txt = vsx.from_numpy(dummy_txt, self.device_id_)
            return ([vacc_image] * batch_size, [vacc_txt] * batch_size)

    def process(self, input):
        image, tensor = input
        vacc_images = []
        if isinstance(image, list):
            if isinstance(image[0], np.ndarray):
                vacc_images = [
                    utils.cv_rgb_image_to_vastai(x, self.device_id_) for x in image
                ]
            else:
                vacc_images = image
        else:
            if isinstance(image, np.ndarray):
                vacc_images.append(utils.cv_rgb_image_to_vastai(image, self.device_id_))
            else:
                vacc_images.append(image)

        vacc_tensors = []
        if isinstance(tensor, list):
            if isinstance(tensor[0], np.ndarray):
                vacc_tensors = [vsx.from_numpy(t, self.device_id_) for t in tensor]
            else:
                vacc_tensors = tensor
        else:
            if isinstance(tensor, np.ndarray):
                vacc_tensors.append(vsx.from_numpy(tensor, self.device_id_))
            else:
                vacc_tensors.append(tensor)
        res = self.process_impl(vacc_images, vacc_tensors)
        if isinstance(image, list):
            return res
        else:
            return res[0]

    def process_impl(self, images, tensors):
        preproc_images = []
        for image in images:
            height, width = image.height, image.width
            self.fusion_op_.set_attribute(
                {
                    attr.IIMAGE_WIDTH: width,
                    attr.IIMAGE_WIDTH_PITCH: width,
                    attr.IIMAGE_HEIGHT: height,
                    attr.IIMAGE_HEIGHT_PITCH: height,
                }
            )
            vdsp_out = self.fusion_op_.execute(
                tensors=[image],
                output_info=[(([1, 160, 160, 1, 256]), vsx.TypeFlag.FLOAT16)],
            )[0]

            preproc_images.append(vdsp_out)

        inputs = [
            [vdsp_out, txt_out] for vdsp_out, txt_out in zip(preproc_images, tensors)
        ]
        outputs = self.stream_.run_sync(inputs)

        results = []
        for output in outputs:
            outs = [vsx.as_numpy(out).astype(np.float32) for out in output]
            result = []
            for i, out in enumerate(outs):
                if i < 3:
                    result.append(out)
                else:
                    out = np.reshape(out, newshape=(1, -1, 4))
                    out = np.transpose(out, axes=(0, 2, 1))
                    out = np.reshape(
                        out, newshape=(1, 4, int(np.sqrt(out.shape[2])), -1)
                    )
                    result.append(out)
            results.append(result)

        return results
