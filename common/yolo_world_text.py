#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from .model_nlp import ModelNLP, vsx

import numpy as np
from typing import Union, List
from transformers import AutoTokenizer


class YoloWorldText(ModelNLP):
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        tokenizer_path,
        batch_size=1,
        device_id=0,
        hw_config="",
    ):
        super().__init__(
            model_prefix=model_prefix,
            vdsp_config=vdsp_config,
            batch_size=batch_size,
            device_id=device_id,
            hw_config=hw_config,
        )
        self.tokenizer_ = AutoTokenizer.from_pretrained(tokenizer_path)

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
        token_dict = self.tokenizer_(text=text, return_tensors="pt", padding=True)
        token = token_dict["input_ids"][0]
        input_seq_len = 16
        token_padding = np.full([input_seq_len], 49407, dtype=np.int32)  # pad
        token_padding[: len(token)] = token
        # make mask
        token_mask = np.ones(shape=(input_seq_len), dtype=np.int32) * (-1)
        mask = token_dict["attention_mask"][0]
        token_mask[: len(mask)] = mask
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
        return [
            [vsx.as_numpy(out).astype(np.float32) for out in output]
            for output in outputs
        ]
