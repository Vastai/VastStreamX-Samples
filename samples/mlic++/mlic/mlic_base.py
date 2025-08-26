#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../../..")
utils_path = os.path.join(current_file_path, "..")
sys.path.append(common_path)
sys.path.append(utils_path)

import vaststreamx as vsx
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from compressai.models import CompressionModel
from compressai.ans import BufferedRansEncoder, RansDecoder
from utils.fun import update_registered_buffers, get_scale_table
from utils.ckbd import *
from modules.transform import *
from utils.vastai import VastaiGaHa, VastaiGs, VastaiHs


class MLICPlusPlus(CompressionModel):
    def __init__(
        self,
        gaha_model_prefix,
        gaha_vdsp_config,
        hs_model_prefix,
        gs_model_prefix,
        tensorize_elf_path,
        batch_size=1,
        device_id=0,
        gaha_hw_config="",
        hs_hw_config="",
        gs_hw_config="",
        patch=64,  # if the model support dynamic shape, then the patch is valid.
        N=192,  # stable value
        M=320,  # stable value
        slice_num=10,  # stable value
    ) -> None:
        super().__init__(entropy_bottleneck_channels=N)
        slice_ch = M // slice_num
        assert slice_ch * slice_num == M

        self.N = N
        self.M = M
        self.slice_num = slice_num
        self.slice_ch = slice_ch

        self.g_a = AnalysisTransform(N=N, M=M)
        self.g_s = SynthesisTransform(N=N, M=M)

        self.h_a = HyperAnalysis(M=M, N=N)
        self.h_s = HyperSynthesis(M=M, N=N)

        assert vsx.set_device(device_id) == 0
        self.vastai_ga_ha = VastaiGaHa(
            gaha_model_prefix,
            gaha_vdsp_config,
            batch_size,
            device_id,
            gaha_hw_config,
            patch,
        )

        self.vastai_g_s = VastaiGs(
            gs_model_prefix,
            tensorize_elf_path,
            batch_size,
            device_id,
            gs_hw_config,
        )

        self.vastai_hs = VastaiHs(hs_model_prefix, batch_size, device_id, hs_hw_config)

        # Gussian Conditional
        self.gaussian_conditional = GaussianConditional(None)

        self.local_context = nn.ModuleList(
            LocalContext(dim=slice_ch) for _ in range(slice_num)
        )

        self.channel_context = nn.ModuleList(
            ChannelContext(in_dim=slice_ch * i, out_dim=slice_ch) if i else None
            for i in range(slice_num)
        )

        # Global Reference for non-anchors
        self.global_inter_context = nn.ModuleList(
            (
                LinearGlobalInterContext(
                    dim=slice_ch * i, out_dim=slice_ch * 2, num_heads=slice_ch * i // 32
                )
                if i
                else None
            )
            for i in range(slice_num)
        )
        self.global_intra_context = nn.ModuleList(
            LinearGlobalIntraContext(dim=slice_ch) if i else None
            for i in range(slice_num)
        )
        self.entropy_parameters_anchor = nn.ModuleList(
            (
                EntropyParameters(in_dim=M * 2 + slice_ch * 6, out_dim=slice_ch * 2)
                if i
                else EntropyParameters(in_dim=M * 2, out_dim=slice_ch * 2)
            )
            for i in range(slice_num)
        )
        self.entropy_parameters_nonanchor = nn.ModuleList(
            (
                EntropyParameters(in_dim=M * 2 + slice_ch * 10, out_dim=slice_ch * 2)
                if i
                else EntropyParameters(
                    in_dim=M * 2 + slice_ch * 2, out_dim=slice_ch * 2
                )
            )
            for i in range(slice_num)
        )

        # Latent Residual Prediction
        self.lrp_anchor = nn.ModuleList(
            LatentResidualPrediction(in_dim=M + (i + 1) * slice_ch, out_dim=slice_ch)
            for i in range(slice_num)
        )
        self.lrp_nonanchor = nn.ModuleList(
            LatentResidualPrediction(in_dim=M + (i + 1) * slice_ch, out_dim=slice_ch)
            for i in range(slice_num)
        )

    def update_resolutions(self, H, W):
        for i in range(len(self.global_intra_context)):
            if i == 0:
                self.local_context[i].update_resolution(
                    H, W, next(self.parameters()).device, mask=None
                )
            else:
                self.local_context[i].update_resolution(
                    H,
                    W,
                    next(self.parameters()).device,
                    mask=self.local_context[0].attn_mask,
                )

    def get_fusion_op_iimage_format(self):
        return self.vastai_ga_ha.get_fusion_op_iimage_format()

    def cal_run_sync_params(self, input):
        params = {"rgb_letterbox_ext": []}
        w, h = input.width, input.height
        model_h, model_w = self.vastai_ga_ha.input_shape_[2:]
        scale = 0.0

        if w <= model_w and h <= model_h:
            params["rgb_letterbox_ext"].append((w, h, 0, model_h - h, 0, model_w - w))
        else:
            scale = min(model_w / w, model_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            params["rgb_letterbox_ext"].append(
                (new_w, new_h, 0, model_h - new_h, 0, model_w - new_w)
            )
        return scale, params

    def compress(self, x):
        start_time = time.time()
        self.update_resolutions(16, 16)

        # vacc
        self.update_resolutions(x.height // 16, x.width // 16)
        scale, params = self.cal_run_sync_params(x)
        y, z = self.vastai_ga_ha.process(x, params)
        y = torch.from_numpy(y)
        z = torch.from_numpy(z)
        end1_time = time.time()
        # print(f"vastai_ga_ha time: {end1_time - start_time}")

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        end2_time = time.time()
        # print(f"entropy_bottleneck time: {end2_time - end1_time}")

        # vacc
        hyper_params = self.vastai_hs.process(z_hat.numpy())
        hyper_params = torch.from_numpy(hyper_params[0])

        hyper_scales, hyper_means = hyper_params.chunk(2, 1)
        y_slices = y.chunk(self.slice_num, dim=1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []
        end3_time = time.time()
        # print(f"vastai_hs time: {end3_time - end2_time}")

        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round and compress anchor
                slice_anchor = compress_anchor(
                    self.gaussian_conditional,
                    slice_anchor,
                    scales_anchor,
                    means_anchor,
                    symbols_list,
                    indexes_list,
                )
                # predict residuals caused by round
                lrp_anchor = self.lrp_anchor[idx](
                    torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1)
                )
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](
                    torch.cat([local_ctx, hyper_params], dim=1)
                )
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # round and compress nonanchor
                slice_nonanchor = compress_nonanchor(
                    self.gaussian_conditional,
                    slice_nonanchor,
                    scales_nonanchor,
                    means_nonanchor,
                    symbols_list,
                    indexes_list,
                )
                # predict residuals caused by round
                lrp_nonanchor = self.lrp_nonanchor[idx](
                    torch.cat(
                        (
                            [hyper_means]
                            + y_hat_slices
                            + [slice_nonanchor + slice_anchor]
                        ),
                        dim=1,
                    )
                )
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)

            else:
                # Anchor
                global_inter_ctx = self.global_inter_context[idx](
                    torch.cat(y_hat_slices, dim=1)
                )
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](
                    torch.cat([global_inter_ctx, channel_ctx, hyper_params], dim=1)
                )
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round and compress anchor
                slice_anchor = compress_anchor(
                    self.gaussian_conditional,
                    slice_anchor,
                    scales_anchor,
                    means_anchor,
                    symbols_list,
                    indexes_list,
                )
                # predict residuals caused by round
                lrp_anchor = self.lrp_anchor[idx](
                    torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1)
                )
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                global_intra_ctx = self.global_intra_context[idx](
                    y_hat_slices[-1], slice_anchor
                )
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](
                    torch.cat(
                        [
                            local_ctx,
                            global_intra_ctx,
                            global_inter_ctx,
                            channel_ctx,
                            hyper_params,
                        ],
                        dim=1,
                    )
                )
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # round and compress nonanchor
                slice_nonanchor = compress_nonanchor(
                    self.gaussian_conditional,
                    slice_nonanchor,
                    scales_nonanchor,
                    means_nonanchor,
                    symbols_list,
                    indexes_list,
                )
                # predict residuals caused by round
                lrp_nonanchor = self.lrp_nonanchor[idx](
                    torch.cat(
                        (
                            [hyper_means]
                            + y_hat_slices
                            + [slice_nonanchor + slice_anchor]
                        ),
                        dim=1,
                    )
                )
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)
        end4_time = time.time()
        # print(f"loop time: {end4_time - end3_time}")

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )
        y_string = encoder.flush()
        y_strings.append(y_string)
        end_time = time.time()
        end5_time = time.time()
        # print(f"encode_with_indexes time: {end5_time - end4_time}")

        cost_time = end_time - start_time
        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:],
            "cost_time": cost_time,
            "scale": scale,
            "params": (
                params["rgb_letterbox_ext"][0][0],  # new_w
                params["rgb_letterbox_ext"][0][1],  # new_h
                params["rgb_letterbox_ext"][0][3],  # padding_bottom
                params["rgb_letterbox_ext"][0][5],  # padding_right
            ),
        }

    def decompress(self, strings, shape):
        start_time = time.time()
        self.update_resolutions(16, 16)

        y_strings = strings[0][0]
        z_strings = strings[1]
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        self.update_resolutions(z_hat.size(2) * 4, z_hat.size(3) * 4)
        end1_time = time.time()
        # print(f"entropy_bottleneck time: {end1_time - start_time}")

        # vacc
        hyper_params = self.vastai_hs.process(z_hat.numpy())
        hyper_params = torch.from_numpy(hyper_params[0])
        end2_time = time.time()
        # print(f"vastai_hs time: {end2_time - end1_time}")

        hyper_scales, hyper_means = hyper_params.chunk(2, 1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_strings)

        end3_time = time.time()
        # print(f"gaussian_conditional time: {end3_time - end2_time}")
        for idx in range(self.slice_num):
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # decompress anchor
                slice_anchor = decompress_anchor(
                    self.gaussian_conditional,
                    scales_anchor,
                    means_anchor,
                    decoder,
                    cdf,
                    cdf_lengths,
                    offsets,
                )
                # predict residuals caused by round
                lrp_anchor = self.lrp_anchor[idx](
                    torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1)
                )
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](
                    torch.cat([local_ctx, hyper_params], dim=1)
                )
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # decompress non-anchor
                slice_nonanchor = decompress_nonanchor(
                    self.gaussian_conditional,
                    scales_nonanchor,
                    means_nonanchor,
                    decoder,
                    cdf,
                    cdf_lengths,
                    offsets,
                )
                # predict residuals caused by round
                lrp_nonanchor = self.lrp_nonanchor[idx](
                    torch.cat(
                        (
                            [hyper_means]
                            + y_hat_slices
                            + [slice_nonanchor + slice_anchor]
                        ),
                        dim=1,
                    )
                )
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)

            else:
                # Anchor
                global_inter_ctx = self.global_inter_context[idx](
                    torch.cat(y_hat_slices, dim=1)
                )
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](
                    torch.cat([global_inter_ctx, channel_ctx, hyper_params], dim=1)
                )
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # decompress anchor
                slice_anchor = decompress_anchor(
                    self.gaussian_conditional,
                    scales_anchor,
                    means_anchor,
                    decoder,
                    cdf,
                    cdf_lengths,
                    offsets,
                )
                # predict residuals caused by round
                lrp_anchor = self.lrp_anchor[idx](
                    torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1)
                )
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                # Non-anchor
                global_intra_ctx = self.global_intra_context[idx](
                    y_hat_slices[-1], slice_anchor
                )
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](
                    torch.cat(
                        [
                            local_ctx,
                            global_intra_ctx,
                            global_inter_ctx,
                            channel_ctx,
                            hyper_params,
                        ],
                        dim=1,
                    )
                )
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # decompress non-anchor
                slice_nonanchor = decompress_nonanchor(
                    self.gaussian_conditional,
                    scales_nonanchor,
                    means_nonanchor,
                    decoder,
                    cdf,
                    cdf_lengths,
                    offsets,
                )
                # predict residuals caused by round
                lrp_nonanchor = self.lrp_nonanchor[idx](
                    torch.cat(
                        (
                            [hyper_means]
                            + y_hat_slices
                            + [slice_nonanchor + slice_anchor]
                        ),
                        dim=1,
                    )
                )
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)
        end4_time = time.time()
        # print(f"loop time: {end4_time - end3_time}")

        y_hat = torch.cat(y_hat_slices, dim=1)

        # vacc
        x_hat = self.vastai_g_s.process(y_hat.numpy())
        x_hat = torch.from_numpy(x_hat[0])

        end_time = time.time()
        end5_time = time.time()
        # print(f"vastai_g_s time: {end5_time - end4_time}")

        cost_time = end_time - start_time

        return {"x_hat": x_hat, "cost_time": cost_time}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated
