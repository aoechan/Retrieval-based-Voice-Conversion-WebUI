import traceback
from time import time as ttime
import faiss
import numpy as np
import parselmouth
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Resample

from infer.hubert import extract_hubert_features, load_hubert_model
from i18n.i18n import I18nAuto
from tools.cuda_graph import run_cuda_graph


i18n = I18nAuto()


def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)


# local mod: unvoiced handling. The original code filled every unvoiced frame
# by interpolation, so whispers, breaths and pauses were given a pitch and
# came out as buzzy "kerokero" voice. Now only very short dropouts are filled.
F0_MAX_FILL_GAP = 3  # frames (10 ms each): fill unvoiced gaps up to 30 ms
F0_MIN_VOICED_RUN = 5  # frames: drop isolated voiced blips shorter than 50 ms


def _runs(mask):
    """Return (start, end) pairs for each run of True values."""
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.diff(padded)
    return zip(np.where(edges == 1)[0], np.where(edges == -1)[0])


def clean_f0(f0):
    f0 = np.asarray(f0, dtype=np.float32).copy()
    n = len(f0)
    for s, e in _runs(f0 > 0):
        if e - s < F0_MIN_VOICED_RUN and s > 0 and e < n:
            f0[s:e] = 0
    voiced = f0 > 0
    if not voiced.any():
        return f0
    for s, e in _runs(~voiced):
        if e - s > F0_MAX_FILL_GAP:
            continue
        if s > 0 and e < n:
            f0[s:e] = np.interp(np.arange(s, e), [s - 1, e], [f0[s - 1], f0[e]])
        elif s == 0:
            f0[s:e] = f0[e]
        else:
            f0[s:e] = f0[s - 1]
    return f0


def read_index_for_realtime(path):
    """local mod: read a faiss index and probe more lists.

    With the default nprobe=1, IVF indexes that have small lists (e.g. made by
    Applio) often return -1 (no neighbour), which made every block fail.
    """
    index = faiss.read_index(path)
    try:
        faiss.extract_index_ivf(index).nprobe = 8
    except Exception:
        pass
    return index


def get_synthesizer(pth_path, device=torch.device("cpu")):
    from infer.module.models import (
        SynthesizerTrnMs256NSFsid,
        SynthesizerTrnMs256NSFsid_nono,
        SynthesizerTrnMs768NSFsid,
        SynthesizerTrnMs768NSFsid_nono,
    )

    cpt = torch.load(pth_path, map_location=torch.device("cpu"))
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=False)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=False)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g = net_g.float()
    net_g.eval().to(device)
    net_g.remove_weight_norm()
    return net_g, cpt


# config.device=torch.device("cpu")########强制cpu测试
# config.is_half=False########强制cpu测试
class RVC:
    def __init__(
        self,
        key,
        formant,
        pth_path,
        index_path,
        index_rate,
        config,
        last_rvc=None,
    ) :
        """
        初始化
        """
        try:
            # global config
            self.config = config
            # device="cpu"########强制cpu测试
            self.device = config.device
            self.f0_up_key = key
            self.formant_shift = formant
            self.f0_min = 50
            self.f0_max = 1100
            self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
            self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
            self.is_half = config.is_half
            if index_rate != 0:
                self.index = read_index_for_realtime(index_path)
                self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
                printt(i18n("已启用索引检索"))
            self.pth_path = pth_path
            self.index_path = index_path
            self.index_rate = index_rate
            self.cache_pitch = torch.zeros(
                1024, device=self.device, dtype=torch.long
            )
            self.cache_pitchf = torch.zeros(
                1024, device=self.device, dtype=torch.float32
            )
            self.infer_count = 0

            self.resample_kernel = {}

            if last_rvc is None:
                self.model = load_hubert_model(self.device, self.is_half)
            else:
                self.model = last_rvc.model

            self.net_g = None

            def set_synthesizer():
                self.net_g, cpt = get_synthesizer(self.pth_path, self.device)
                self.tgt_sr = cpt["config"][-1]
                cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
                self.if_f0 = cpt.get("f0", 1)
                self.version = cpt.get("version", "v1")
                if self.is_half:
                    self.net_g = self.net_g.half()
                else:
                    self.net_g = self.net_g.float()

            if last_rvc is None or last_rvc.pth_path != self.pth_path:
                set_synthesizer()
            else:
                self.tgt_sr = last_rvc.tgt_sr
                self.if_f0 = last_rvc.if_f0
                self.version = last_rvc.version
                self.is_half = last_rvc.is_half
                self.net_g = last_rvc.net_g

            if last_rvc is not None and hasattr(last_rvc, "model_rmvpe"):
                self.model_rmvpe = last_rvc.model_rmvpe
            if last_rvc is not None and hasattr(last_rvc, "model_fcpe"):
                self.model_fcpe = last_rvc.model_fcpe
        except:
            printt(traceback.format_exc())

    def change_key(self, new_key):
        self.f0_up_key = new_key

    def change_formant(self, new_formant):
        self.formant_shift = new_formant

    def change_index_rate(self, new_index_rate):
        if new_index_rate != 0 and self.index_rate == 0:
            self.index = read_index_for_realtime(self.index_path)
            self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
            printt(i18n("已启用索引检索"))
        self.index_rate = new_index_rate

    def get_f0_post(self, f0):
        if not torch.is_tensor(f0):
            f0 = torch.from_numpy(f0)
        f0 = f0.float().to(self.device).squeeze()
        f0_mel = 1127 * torch.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (
            self.f0_mel_max - self.f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = torch.round(f0_mel).long()
        return f0_coarse, f0

    def get_f0(self, x, f0_up_key, method="rmvpe"):
        if method == "rmvpe":
            return self.get_f0_rmvpe(x, f0_up_key)
        if method == "fcpe":
            return self.get_f0_fcpe(x, f0_up_key)
        if method != "pm":
            raise ValueError(f"Unsupported F0 method: {method}")
        x = x.cpu().numpy()
        p_len = x.shape[0] // 160 + 1
        f0_min = 65
        l_pad = int(np.ceil(1.5 / f0_min * 16000))
        r_pad = l_pad + 1
        s = parselmouth.Sound(np.pad(x, (l_pad, r_pad)), 16000).to_pitch_ac(
            time_step=0.01,
            voicing_threshold=0.6,
            pitch_floor=f0_min,
            pitch_ceiling=1100,
        )
        assert np.abs(s.t1 - 1.5 / f0_min) < 0.001
        f0 = s.selected_array["frequency"]
        if len(f0) < p_len:
            f0 = np.pad(f0, (0, p_len - len(f0)))
        f0 = f0[:p_len]
        f0 = clean_f0(f0)  # local mod
        if getattr(self, "unvoiced_renderer", None) is not None:  # local mod
            self.unvoiced_renderer.mark(f0, x)
        f0 *= pow(2, f0_up_key / 12)
        return self.get_f0_post(f0)

    def get_f0_rmvpe(self, x, f0_up_key):
        if hasattr(self, "model_rmvpe") == False:
            from infer.rmvpe import RMVPE

            printt(i18n("正在加载RMVPE模型"))
            self.model_rmvpe = RMVPE(
                "assets/rmvpe/rmvpe.pt",
                is_half=self.is_half,
                device=self.device,
            )
        f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        f0 = clean_f0(f0)  # local mod
        if getattr(self, "unvoiced_renderer", None) is not None:  # local mod
            self.unvoiced_renderer.mark(f0, x)
        f0 *= pow(2, f0_up_key / 12)
        return self.get_f0_post(f0)

    def get_f0_fcpe(self, x, f0_up_key):
        if hasattr(self, "model_fcpe") == False:
            from infer.fcpe import FCPEInfer

            printt("Loading fcpe model")
            self.model_fcpe = FCPEInfer(self.device)
        f0 = self.model_fcpe.infer(
            x.unsqueeze(0).float(),
            sr=16000,
            decoder_mode="local_argmax",
            threshold=0.006,
        ).squeeze().detach().cpu().numpy()
        f0 = clean_f0(f0)  # local mod
        if getattr(self, "unvoiced_renderer", None) is not None:  # local mod
            self.unvoiced_renderer.mark(f0, x)
        f0 *= pow(2, f0_up_key / 12)
        return self.get_f0_post(f0)

    def infer(
        self,
        input_wav,
        block_frame_16k,
        skip_head,
        return_length,
        f0method,
    ) :
        report_status = self.infer_count < 3 or self.infer_count % 100 == 0
        self.infer_count += 1
        t1 = ttime()
        with torch.no_grad():
            if self.config.is_half:
                feats = input_wav.half().view(1, -1)
            else:
                feats = input_wav.float().view(1, -1)
            padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
            feats = extract_hubert_features(
                self.model,
                feats,
                self.version,
                padding_mask=padding_mask,
            )
            feats = torch.cat((feats, feats[:, -1:, :]), 1)
            if getattr(self, "unvoiced_renderer", None) is not None:  # local mod
                self.unvoiced_renderer.set_content_features(feats)
        t2 = ttime()
        try:
            if hasattr(self, "index") and self.index_rate != 0:
                npy = feats[0][skip_head // 2 :].cpu().numpy().astype("float32")
                score, ix = self.index.search(npy, k=8)
                # local mod: skip missing neighbours (-1) per frame instead of
                # dropping the whole block; frames with no neighbour keep their features.
                valid = ix >= 0
                if valid.any():
                    weight = np.where(valid, np.square(1 / np.maximum(score, 1e-9)), 0.0)
                    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-12)
                    retrieved = np.sum(
                        self.big_npy[np.maximum(ix, 0)] * np.expand_dims(weight, axis=2),
                        axis=1,
                    )
                    no_hit = ~valid.any(axis=1)
                    retrieved[no_hit] = npy[no_hit]
                    npy = retrieved.astype("float32")
                    if self.config.is_half:
                        npy = npy.astype("float16")
                    feats[0][skip_head // 2 :] = (
                        torch.from_numpy(npy).unsqueeze(0).to(self.device)
                        * self.index_rate
                        + (1 - self.index_rate) * feats[0][skip_head // 2 :]
                    )
                else:
                    printt(
                        i18n("索引无效：必须使用added_xxxx.index，不能使用trained_xxxx.index")
                    )
            else:
                if report_status:
                    printt(i18n("索引检索失败或未启用"))
        except Exception:
            traceback.print_exc()
            printt(i18n("索引检索失败"))
        t3 = ttime()
        p_len = input_wav.shape[0] // 160
        factor = pow(2, self.formant_shift / 12)
        return_length2 = int(np.ceil(return_length * factor))
        if self.if_f0 == 1:
            f0_extractor_frame = block_frame_16k + 800
            if f0method == "rmvpe":
                f0_extractor_frame = 5120 * ((f0_extractor_frame - 1) // 5120 + 1) - 160
            if getattr(self, "unvoiced_renderer", None) is not None:  # local mod
                self.unvoiced_renderer.shift_frames = block_frame_16k // 160
            pitch, pitchf = self.get_f0(
                input_wav[-f0_extractor_frame:],
                self.f0_up_key - self.formant_shift,
                f0method,
            )
            shift = block_frame_16k // 160
            self.cache_pitch[:-shift] = self.cache_pitch[shift:].clone()
            self.cache_pitchf[:-shift] = self.cache_pitchf[shift:].clone()
            self.cache_pitch[4 - pitch.shape[0] :] = pitch[3:-1]
            self.cache_pitchf[4 - pitch.shape[0] :] = pitchf[3:-1]
            cache_pitch = self.cache_pitch[None, -p_len:]
            cache_pitchf = self.cache_pitchf[None, -p_len:] * return_length2 / return_length
        # local mod: pitch for a second, "pseudo-voiced" synthesis of whisper/breath
        # stretches (see unvoiced_render.py). The normal synthesis is left untouched.
        pseudo = None
        self.last_pseudo_audio = None
        ren = getattr(self, "unvoiced_renderer", None)
        if self.if_f0 == 1 and ren is not None and ren.win_mask is not None and ren.win_mask.any():
            W = len(ren.win_mask)
            k = np.where(ren.win_mask)[0]
            k = k[(k >= 3) & (k <= W - 2)]  # frames that were written to the cache
            idx = p_len + (k + 1 - W)
            idx = idx[(idx >= 0) & (idx < p_len)]
            if len(idx):
                f_ref = ren.f0_ref * pow(2, (self.f0_up_key - self.formant_shift) / 12)
                coarse_ref, f_ref_t = self.get_f0_post(np.array([f_ref, f_ref], dtype=np.float32))
                idx_t = torch.from_numpy(idx).long().to(self.device)
                pseudo_pitch = cache_pitch.clone()
                pseudo_pitchf = cache_pitchf.clone()
                pseudo_pitch[0, idx_t] = coarse_ref[0]
                pseudo_pitchf[0, idx_t] = f_ref_t[0] * return_length2 / return_length
                pseudo = (pseudo_pitch, pseudo_pitchf)
        t4 = ttime()
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        feats = feats[:, :p_len, :]
        p_len_tensor = torch.LongTensor([p_len]).to(self.device)
        sid = torch.LongTensor([0]).to(self.device)
        skip_head_value = int(skip_head)
        return_length_value = int(return_length)
        return_length2_value = int(return_length2)
        with torch.no_grad():
            if self.if_f0 == 1:
                graph_key = "rvc-realtime-f0-%s-%s-%s" % (
                    skip_head_value,
                    return_length_value,
                    return_length2_value,
                )
                synth = lambda phone, lengths, coarse, continuous, speaker: self.net_g.infer(
                    phone,
                    lengths,
                    coarse,
                    continuous,
                    speaker,
                    skip_head_value,
                    return_length_value,
                    return_length2_value,
                )[0]
                infered_audio = run_cuda_graph(
                    self.net_g, graph_key, synth, feats, p_len_tensor, cache_pitch, cache_pitchf, sid
                )
                if pseudo is not None:  # local mod: second synthesis, same graph
                    if getattr(self, "pseudo_fork_rng", False):  # tests only: keep the RNG stream of the normal synthesis
                        with torch.random.fork_rng(devices=[self.device] if "cuda" in str(self.device) else []):
                            pseudo_audio = run_cuda_graph(
                                self.net_g, graph_key, synth, feats, p_len_tensor, pseudo[0], pseudo[1], sid
                            )
                    else:
                        pseudo_audio = run_cuda_graph(
                            self.net_g, graph_key, synth, feats, p_len_tensor, pseudo[0], pseudo[1], sid
                        )
                    self.last_pseudo_audio = pseudo_audio.squeeze(1).float()
            else:
                infered_audio = run_cuda_graph(
                    self.net_g,
                    "rvc-realtime-no-f0-%s-%s-%s"
                    % (skip_head_value, return_length_value, return_length2_value),
                    lambda phone, lengths, speaker: self.net_g.infer(
                        phone,
                        lengths,
                        speaker,
                        skip_head_value,
                        return_length_value,
                        return_length2_value,
                    )[0],
                    feats,
                    p_len_tensor,
                    sid,
                )
        infered_audio = infered_audio.squeeze(1).float()
        upp_res = int(np.floor(factor * self.tgt_sr // 100))
        if upp_res != self.tgt_sr // 100:
            if upp_res not in self.resample_kernel:
                self.resample_kernel[upp_res] = Resample(
                    orig_freq=upp_res,
                    new_freq=self.tgt_sr // 100,
                    dtype=torch.float32,
                ).to(self.device)
            infered_audio = self.resample_kernel[upp_res](
                infered_audio[:, : return_length * upp_res]
            )
            if self.last_pseudo_audio is not None:  # local mod
                self.last_pseudo_audio = self.resample_kernel[upp_res](
                    self.last_pseudo_audio[:, : return_length * upp_res]
                )
        if self.last_pseudo_audio is not None:  # local mod
            self.last_pseudo_audio = self.last_pseudo_audio.squeeze()
        t5 = ttime()
        if report_status:
            printt(
                i18n("耗时：特征=%.3f秒，索引=%.3f秒，音高=%.3f秒，模型=%.3f秒"),
                t2 - t1,
                t3 - t2,
                t4 - t3,
                t5 - t4,
            )
        return infered_audio.squeeze()
