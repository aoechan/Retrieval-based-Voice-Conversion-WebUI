"""local mod: render whispers and breaths in the target voice.

RVC models trained on normal speech render long unvoiced stretches (whisper,
breath, sigh) poorly. rtrvc.RVC runs a second synthesis in which those
stretches get a steady pseudo pitch, so the model renders ordinary voiced
speech there (what it does well). We then remove the voicing with LPC: keep
each frame's vocal-tract envelope, replace the excitation with noise, follow
the input's loudness, and put the result into the normal synthesis only in
those stretches. The normal synthesis is never altered elsewhere.

Step 1 `mark()` runs on the F0 window (called from rtrvc.get_f0_*).
Step 2 `render()` runs on the converted audio before SOLA.
"""

import numpy as np
import torch
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter


def _runs(mask):
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.diff(padded)
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def whisperize(y, sr, order=40, frame_ms=21.3, hop_ms=5.3, deemph=0.9, rng=None):
    """Replace the excitation of y with noise, keeping the LPC envelope per frame."""
    rng = rng or np.random.default_rng()
    n = int(round(frame_ms / 1000 * sr)) // 2 * 2
    h = max(1, int(round(hop_ms / 1000 * sr)))
    if len(y) == 0:
        return np.zeros_like(y)
    # Pad both ends (mirror) so every output sample is covered by full window
    # overlap. Without this, the overlap-add normalisation divided by a tiny
    # window sum at the ends and boosted the first/last few ms up to ~30x,
    # which clicked at block boundaries.
    length = len(y)
    y = np.pad(y, (n, n), mode="reflect" if length > n else "edge")
    win = np.hanning(n)
    pre = lfilter([1.0, -0.97], [1.0], y)
    starts = np.arange(0, len(y) - n + 1, h)
    frames = np.stack([pre[s : s + n] for s in starts]) * win
    # autocorrelation for all frames via FFT
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(frames, nfft, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, nfft, axis=1)[:, : order + 1]
    out = np.zeros(len(y) + n)
    norm = np.zeros(len(y) + n)
    for i, s in enumerate(starts):
        r = ac[i]
        if r[0] <= 1e-10:
            continue
        r = r * np.exp(-0.5 * (np.arange(order + 1) * 2 * np.pi * 40 / sr) ** 2)  # lag window
        r[0] *= 1.0001
        try:
            a = np.concatenate(([1.0], -solve_toeplitz(r[:order], r[1 : order + 1])))
        except Exception:
            continue
        seg = lfilter([1.0], a, rng.standard_normal(n)) * win
        if not np.all(np.isfinite(seg)):
            continue
        # match the frame's own energy (smooth across frames), not the LPC
        # residual, which jumps at waveform kinks such as the padded ends
        seg *= np.sqrt(np.mean(frames[i] ** 2) / (np.mean(seg**2) + 1e-20))
        out[s : s + n] += seg
        norm[s : s + n] += win**2
    out = out[: len(y)] / np.maximum(norm[: len(y)], 1e-3)
    return lfilter([1.0], [1.0, -deemph], out)[n : n + length]


class UnvoicedRenderer:
    MIN_RUN_FRAMES = 24  # 10 ms frames (240 ms); shorter unvoiced stretches are consonants
    MIN_RUN_FRAMES_ONGOING = 16  # still going on at the newest input
    MAX_GAP_FRAMES = 3
    AUDIBLE_DB = -50.0
    WEAK_VOICED_DB = -8.0  # weak "voiced" frames inside a breath are usually spurious pitch
    MAX_SPURIOUS_VOICED_FRAMES = 8  # ...but only short weak voiced blips (< 80 ms); real speech runs longer
    SIBILANT_HF = 0.3
    ONSET_GUARD_FRAMES = 8  # leave the 80 ms before voiced speech to normal conversion (consonant onsets like k/t)
    TAIL_LOOKBACK_FRAMES = 25  # a stretch is a phrase-end tail if voiced speech ends within 250 ms before it
    MIN_SPEECH_FRAMES = 10  # voiced runs shorter than 100 ms do not count as that speech
    CONTENT_LAG_FRAMES = 5  # HuBERT frames (20 ms): 100 ms
    TAIL_MAX_CONTENT_CHANGE = 0.55  # median after the hold: tails about 0.1-0.4, whisper after a voiced inhale 0.85
    TAIL_HOLD_FRAMES = 30  # 300 ms after a voiced end that are never rendered
    RAMP_MS = 15  # crossfade length, placed outside the rendered stretch (real converted audio side)
    LONG_FADE_MS = 150  # fade-in for a stretch that starts after a phrase end (inside the stretch)
    # Loudness following: very quiet input (room noise, mouth/chair noises after
    # speech) is pushed down instead of reproduced, like the normal synthesis does,
    # and 1-2 frame spikes are removed with a median filter before following.
    EXPAND_THRESHOLD_DB = -42.0  # relative to the speech level
    EXPAND_EXTRA = 2.0  # each dB below the threshold lowers the output by 1 + 2 dB
    EXPAND_MAX_DB = 40.0
    LEVEL_MEDIAN_FRAMES = 5  # 50 ms

    def __init__(self, sr):
        self.sr = int(sr)
        self.speech_db = None
        self.f0_ref = 140.0  # unshifted Hz, tracked from voiced speech
        self.gain_db = 15.0
        self.win_mask = None
        self.win_strong = None
        self.rng = np.random.default_rng(0)
        self.tail_active = False  # inside a phrase-end tail that is left to the normal synthesis
        self.tail_start_abs = None  # absolute 10 ms frame where that tail began (the voiced end)
        self.frames_seen = 0  # absolute index of the newest F0 frame
        self.shift_frames = None  # F0 frames per block, set by rtrvc
        self.last_ratio = 0.0

    # step 0 ---------------------------------------------------------------
    def set_content_features(self, feats):
        """HuBERT features (1, T, C) per 20 ms of the whole buffer. Stores how fast
        they change (cosine distance at a 100 ms lag) per 10 ms frame. Whisper and
        speech carry phonetic content and change fast; exhale/room tails do not."""
        h = torch.nn.functional.normalize(feats[0].float(), dim=1)
        d = torch.zeros(h.shape[0], device=h.device)
        lag = self.CONTENT_LAG_FRAMES
        if h.shape[0] > lag:
            d[lag:] = 1 - (h[lag:] * h[:-lag]).sum(1)
        self.content_change = np.repeat(d.cpu().numpy(), 2)

    # step 1 ---------------------------------------------------------------
    def mark(self, f0, x16):
        """f0: unshifted F0 per 10 ms frame for the window x16 (16 kHz). Sets self.win_mask."""
        x = x16.detach().float().cpu().numpy() if torch.is_tensor(x16) else np.asarray(x16, np.float32)
        W = len(f0)
        self.frames_seen += self.shift_frames if self.shift_frames else W
        window_start_abs = self.frames_seen - W + 1
        pad = np.pad(x, (160, 160 + 320))
        frames = np.stack([pad[k * 160 : k * 160 + 320] for k in range(W)])
        energy = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-12)
        spec = np.abs(np.fft.rfft(frames * np.hanning(320), axis=1)) ** 2
        freqs = np.fft.rfftfreq(320, 1 / 16000)
        hf = spec[:, freqs > 4000].sum(1) / (spec[:, freqs > 100].sum(1) + 1e-12)
        voiced = f0 > 0
        floor = self.speech_db - 25 if self.speech_db is not None else -70.0
        strong = voiced & (energy > floor)
        if strong.sum() >= 10:
            level = float(np.percentile(energy[strong], 75))
            self.speech_db = level if self.speech_db is None else 0.8 * self.speech_db + 0.2 * level
            self.f0_ref = 0.8 * self.f0_ref + 0.2 * float(np.median(f0[strong]))
        self.win_mask = np.zeros(W, bool)
        self.win_fadein = np.zeros(W, bool)
        self.win_strong = strong
        if self.speech_db is None:
            return
        rel = energy - self.speech_db
        spurious = np.zeros(W, bool)
        for s, e in _runs(voiced):
            if e - s < self.MAX_SPURIOUS_VOICED_FRAMES and np.all(rel[s:e] < self.WEAK_VOICED_DB):
                spurious[s:e] = True
        cand = (rel > self.AUDIBLE_DB) & (hf < self.SIBILANT_HF) & (~voiced | spurious)
        for s, e in _runs(~cand):
            if 0 < s and e < W and e - s <= self.MAX_GAP_FRAMES:
                cand[s:e] = True
        mask = np.zeros(W, bool)
        real_voiced = voiced & ~spurious
        # only voiced runs of at least 100 ms count as the speech a tail follows;
        # shorter voiced bits inside a breath must not stop it being rendered
        speech_frames = np.zeros(W, bool)
        for s, e in _runs(real_voiced):
            if e - s >= self.MIN_SPEECH_FRAMES:
                speech_frames[s:e] = True
        tail_active = False
        new_tail_start = None
        fadein = np.zeros(W, bool)
        cc = getattr(self, "content_change", None)
        change = None
        if cc is not None and len(cc) >= W:
            change = cc[len(cc) - W - 1 : len(cc) - 1] if len(cc) > W else cc[-W:]  # aligned to the buffer end
        for s, e in _runs(cand):
            # An unvoiced stretch that starts right where voiced speech ends is the
            # fading end of a phrase (exhale, room tail). Rendering it as a whisper
            # made a creak after phrases, so it is left to the normal synthesis,
            # which simply fades out. Look back over consonants such as a final
            # devoiced "su" (audible, not a candidate) to find the voiced end. The
            # state carries over to the next window so a tail is not picked up halfway.
            j, steps = s - 1, 0
            while j >= 0 and steps < self.TAIL_LOOKBACK_FRAMES and not speech_frames[j] and rel[j] > self.AUDIBLE_DB:
                j -= 1
                steps += 1
            post_speech = speech_frames[j] if j >= 0 else self.tail_active
            med = float("nan")
            if post_speech:
                # 1) the first TAIL_HOLD after the voiced end is always left to the
                #    normal synthesis (it fades out there; the creak was here). The
                #    voiced end is remembered in absolute frames across windows.
                if j >= 0:
                    tail_start = window_start_abs + j + 1
                else:
                    tail_start = self.tail_start_abs if self.tail_start_abs is not None else window_start_abs
                hold_end = max(s, tail_start + self.TAIL_HOLD_FRAMES - window_start_abs)
                # 2) after that, whisper (fast-changing content) is rendered and an
                #    exhale/room tail (slow) is left alone. The first ~0.3 s of a tail
                #    still wobbles, so it is not used for this decision.
                rest = change[hold_end:e] if change is not None else np.zeros(0)
                med = float(np.median(rest)) if len(rest) >= 10 else float("nan")
                if med >= self.TAIL_MAX_CONTENT_CHANGE:
                    s = hold_end
                    post_speech = False
                    if s < W:
                        fadein[s] = True  # start slowly, the normal synthesis is near silence here
            if getattr(self, "debug_log", None) is not None:  # tests only
                self.debug_log.append((s, e, W, j, bool(post_speech), med, bool(self.tail_active)))
            if post_speech:
                if e >= W - 2:
                    tail_active = True
                    new_tail_start = tail_start
                continue
            if e < W and real_voiced[e]:
                e -= self.ONSET_GUARD_FRAMES
            if e - s >= (self.MIN_RUN_FRAMES_ONGOING if e >= W - 2 else self.MIN_RUN_FRAMES):
                mask[s:e] = True
        self.tail_active = tail_active
        self.tail_start_abs = new_tail_start
        self.win_fadein = fadein & mask
        self.win_mask = mask

    # step 2 ---------------------------------------------------------------
    def render(self, infer_wav, pseudo_wav, input_wav, extra_frame, p_len, skip_head):
        """infer_wav: normal synthesis of the region (device rate), aligned to input_wav[extra_frame:].
        pseudo_wav: the pseudo-voiced synthesis of the same region (or None)."""
        self.last_ratio = 0.0
        self.last_m = None
        if self.win_mask is None or pseudo_wav is None:
            return infer_wav
        W = len(self.win_mask)
        L = int(infer_wav.shape[0])
        zc = self.sr // 100
        R = L // zc
        k = skip_head + np.arange(R) - p_len + W - 1  # window frame for each region frame
        ok = (k >= 0) & (k < W)
        m_fr = np.zeros(R, bool)
        m_fr[ok] = self.win_mask[k[ok]]
        strong_fr = np.zeros(R, bool)
        strong_fr[ok] = self.win_strong[k[ok]]
        fadein_fr = np.zeros(R, bool)
        fadein_fr[ok] = self.win_fadein[k[ok]]
        inp = input_wav[extra_frame : extra_frame + L].detach().float().cpu().numpy()
        if len(inp) < L:
            inp = np.pad(inp, (0, L - len(inp)))
        y = infer_wav.detach().float().cpu().numpy()
        yp = pseudo_wav.detach().float().cpu().numpy()
        if len(yp) < L:
            yp = np.pad(yp, (0, L - len(yp)))

        # follow the converted/input level ratio on clearly voiced frames
        if strong_fr.sum() >= 10:
            sel = np.repeat(strong_fr, zc)[:L]
            ratio_db = 20 * np.log10((np.sqrt(np.mean(y[: len(sel)][sel] ** 2)) + 1e-9) /
                                     (np.sqrt(np.mean(inp[: len(sel)][sel] ** 2)) + 1e-9))
            self.gain_db = float(np.clip(0.8 * self.gain_db + 0.2 * ratio_db, 0.0, 30.0))
        if not m_fr.any():
            return infer_wav

        # The pseudo-voiced synthesis is a buzz at speech level, so it never goes to
        # the output directly; only its whisperized version does. The crossfade is
        # placed just outside each stretch, on the normal-synthesis side.
        ramp = int(self.RAMP_MS / 1000 * self.sr)
        m = np.zeros(L, np.float32)
        out = y.copy()
        gain = 10 ** (self.gain_db / 20)
        # target loudness per 10 ms frame of the region: median-filtered input level,
        # pushed down below EXPAND_THRESHOLD_DB, scaled to the converted voice
        in_db = 20 * np.log10(np.sqrt(np.mean(inp[: R * zc].reshape(R, zc) ** 2, axis=1)) + 1e-9)
        half = self.LEVEL_MEDIAN_FRAMES // 2
        in_db = np.median(
            np.lib.stride_tricks.sliding_window_view(np.pad(in_db, (half, half), mode="edge"), self.LEVEL_MEDIAN_FRAMES),
            axis=1,
        )
        if self.speech_db is not None:
            below = np.clip(self.EXPAND_THRESHOLD_DB - (in_db - self.speech_db), 0.0, None)
            in_db = in_db - np.minimum(self.EXPAND_EXTRA * below, self.EXPAND_MAX_DB)
        target_fr = 10 ** (in_db / 20) * gain
        centers = np.arange(R) * zc + zc / 2
        long_ramp = int(self.LONG_FADE_MS / 1000 * self.sr)
        for s, e in _runs(m_fr):
            a, b = s * zc, min(e * zc, L)
            slow = bool(fadein_fr[s])  # starts after a phrase: fade in slowly, inside the stretch
            fa = a if (s == 0 or slow) else max(0, a - ramp)  # s == 0: continues from the previous block
            fb = b if e >= R else min(L, b + ramp)  # e == R: still going on
            pa, pb = max(0, fa - 1024), min(L, fb + 1024)  # LPC analysis context
            w = whisperize(yp[pa:pb], self.sr, rng=self.rng)[fa - pa : fb - pa]
            # follow the target loudness (10 ms frames)
            nf = int(np.ceil((fb - fa) / zc))
            seg_w = np.pad(w, (0, nf * zc - (fb - fa)))
            t_rms = np.interp(fa + np.arange(nf) * zc + zc / 2, centers, target_fr)
            w_rms = np.sqrt(np.mean(seg_w.reshape(nf, zc) ** 2, axis=1)) + 1e-9
            g = np.clip(t_rms / w_rms, 0, 20)
            g = np.convolve(np.pad(g, (2, 2), mode="edge"), np.ones(5) / 5, mode="valid")
            w = w * np.interp(np.arange(fb - fa), np.arange(nf) * zc + zc / 2, g)
            mm = np.ones(fb - fa, np.float32)
            if a - fa > 0:
                r = a - fa
                mm[:r] = 0.5 - 0.5 * np.cos(np.pi * np.arange(r) / r)
            if slow:
                r = min(long_ramp, (b - a) // 2)
                if r > 0:
                    mm[:r] = 0.5 - 0.5 * np.cos(np.pi * np.arange(r) / r)
            if fb - b > 0:
                r = fb - b
                mm[-r:] = 0.5 + 0.5 * np.cos(np.pi * (np.arange(r) + 1) / r)
            out[fa:fb] = y[fa:fb] * (1 - mm) + w * mm
            m[fa:fb] = np.maximum(m[fa:fb], mm)
        self.last_m = m
        self.last_ratio = float(m.mean())
        return torch.from_numpy(out.astype(np.float32)).to(infer_wav.device)
