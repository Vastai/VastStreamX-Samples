from .model_cv import ModelCV, vsx

import numpy as np
from typing import Union, List
import clip
import copy


class ClipText(ModelCV):
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        batch_size=1,
        device_id=0,
        hw_config="",
        output_type=vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_NCHW_DEVICE,
    ):
        super().__init__(
            model_prefix=model_prefix,
            vdsp_config=vdsp_config,
            batch_size=batch_size,
            device_id=device_id,
            hw_config=hw_config,
            output_type=output_type,
        )

    def get_test_data(self, dtype, input_shape, batch_size, ctx="VACC"):
        tokens = self.make_tokens("test string")
        if ctx == "CPU":
            return [tokens] * batch_size
        else:
            return [
                [vsx.from_numpy(token, self.device_id_) for token in tokens]
            ] * batch_size

    def make_tokens(self, text):
        assert isinstance(text, str), f"input type must be str"
        token = clip.tokenize(text)[0]
        token_padding = np.pad(token.numpy(), pad_width=(0, 3)).astype(np.int32)
        # make mask
        index = np.argmax(token_padding)
        token_mask = copy.deepcopy(token_padding)
        token_mask[: index + 1] = 1
        # make input
        zero_arr = np.zeros(token_padding.shape, dtype=np.int32)
        tokens = []
        tokens.append(token_padding)
        tokens.append(zero_arr)
        tokens.append(zero_arr)
        tokens.append(token_mask)
        tokens.append(zero_arr)
        tokens.append(zero_arr)

        return tokens

    def process(
        self,
        input: Union[
            List[List[vsx.Tensor]],
            List[List[np.ndarray]],
            List[vsx.Tensor],
            List[np.ndarray],
            List[str],
            str,
        ],
    ):
        if isinstance(input, list):
            if isinstance(input[0], list):
                if isinstance(input[0][0], np.ndarray):
                    return self.process(
                        [
                            [
                                vsx.from_numpy(
                                    np.array(x, dtype=np.int32), self.device_id_
                                )
                                for x in one
                            ]
                            for one in input
                        ]
                    )
                else:
                    return self.process_impl(input)
            elif isinstance(input[0], str):
                return self.process([self.make_tokens(x) for x in input])
            elif isinstance(input[0], np.ndarray):
                tensors = [
                    vsx.from_numpy(np.array(x, dtype=np.int32), self.device_id_)
                    for x in input
                ]
                return self.process_impl([tensors])[0]
            else:
                return self.process_impl([input])[0]
        else:
            tokens = self.make_tokens(input)
            return self.process(tokens)

    def process_impl(self, input):
        outputs = self.stream_.run_sync(input)
        return [vsx.as_numpy(out[0]).astype(np.float32) for out in outputs]
