from .model_cv import ModelCV, vsx
import numpy as np


class FaceDetector(ModelCV):
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        batch_size=1,
        device_id=0,
        threshold=0.01,
        hw_config="",
    ) -> None:
        super().__init__(model_prefix, vdsp_config, batch_size, device_id, hw_config)
        self.threshold_ = threshold

    def set_threshold(self, threshold):
        self.threshold_ = threshold

    def process_impl(self, input):
        outputs = self.stream_.run_sync(input)
        return [
            self.post_process(output, input[i].width, input[i].height)
            for i, output in enumerate(outputs)
        ]

    def post_process(self, fp16_tensors, image_width, image_height):
        score_data = vsx.as_numpy(fp16_tensors[0]).squeeze().astype(np.float32)
        bbox_data = vsx.as_numpy(fp16_tensors[1]).squeeze().astype(np.float32)
        landmark_data = vsx.as_numpy(fp16_tensors[2]).squeeze().astype(np.float32)
        score_size = fp16_tensors[0].size
        landmark_size = int(fp16_tensors[2].size / score_size)
        one_face_len = 1 + 4 + landmark_size

        face_count = 0
        for i in range(score_size):
            if score_data[i] < self.threshold_:
                break
            face_count += 1

        result_np = np.zeros((face_count, one_face_len), dtype=np.float32)
        if face_count == 0:
            return result_np

        result_np[0][0] = -1.0

        model_width = self.model_.input_shape[0][3]
        model_height = self.model_.input_shape[0][2]

        r = min(model_width / image_width, model_height / image_height)
        unpad_w = image_width * r
        unpad_h = image_height * r
        dw = (model_width - unpad_w) / 2
        dh = (model_height - unpad_h) / 2

        for i in range(score_size):
            if score_data[i] < self.threshold_:
                break
            score = score_data[i]

            bbox_xmin = (bbox_data[i][0] - dw) / r
            bbox_ymin = (bbox_data[i][1] - dh) / r
            bbox_xmax = (bbox_data[i][2] - dw) / r
            bbox_ymax = (bbox_data[i][3] - dh) / r
            bbox_width = bbox_xmax - bbox_xmin
            bbox_height = bbox_ymax - bbox_ymin
            result_np[i][0] = score
            result_np[i][1] = bbox_xmin
            result_np[i][2] = bbox_ymin
            result_np[i][3] = bbox_width
            result_np[i][4] = bbox_height

            for s in range(landmark_size):
                if s % 2 == 0:
                    result_np[i][s + 5] = (landmark_data[i][s] - dw) / r
                else:
                    result_np[i][s + 5] = (landmark_data[i][s] - dh) / r
        return result_np
