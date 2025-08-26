from .model_nlp import ModelNLP, vsx
import numpy as np


class Bert(ModelNLP):
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        batch_size=1,
        device_id=0,
        hw_config="",
    ) -> None:
        super().__init__(model_prefix, vdsp_config, batch_size, device_id, hw_config)

    def get_test_data(self, dtype, input_shape, batch_size, ctx="VACC"):
        if ctx == "CPU":
            return [
                [np.zeros(shape, dtype=dtype) for shape in input_shape]
            ] * batch_size
        else:
            return [
                [
                    vsx.from_numpy(np.zeros(shape, dtype=dtype), self.device_id_)
                    for shape in input_shape
                ]
            ] * batch_size

    def process_impl(self, input):
        outputs = self.stream_.run_sync(input)
        return [[vsx.as_numpy(o).astype(np.float32) for o in out] for out in outputs]
