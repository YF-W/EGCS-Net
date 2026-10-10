import math
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------- small utils -----------------
def _norm2(x: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(torch.clamp((x * x).sum(dim=dim, keepdim=True), min=eps))


def _spatial_softmax(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    return F.softmax(x.view(b, c, -1), dim=-1).view(b, c, h, w)


def _softargmax_2d(p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # p: [B,1,H,W] sum=1
    b, _, h, w = p.shape
    xs = torch.linspace(0, w - 1, w, device=p.device, dtype=p.dtype)
    ys = torch.linspace(0, h - 1, h, device=p.device, dtype=p.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xx = xx.view(1, 1, h, w)
    yy = yy.view(1, 1, h, w)
    cx = (p * xx).sum((2, 3))
    cy = (p * yy).sum((2, 3))
    c_xy = torch.cat([cx, cy], dim=1)  # [B,2]
    cxn = 2 * (cx / (w - 1 + 1e-6)) - 1
    cyn = 2 * (cy / (h - 1 + 1e-6)) - 1
    c = torch.cat([cxn, cyn], dim=1)   # [B,2]
    return c_xy, c


def _rc_from_entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # p: [B,1,H,W]
    b, _, h, w = p.shape
    q = p.view(b, -1).clamp_min(eps)
    ent = -(q * q.log()).sum(dim=1, keepdim=True)
    return (1 - ent / (math.log(h * w + 1e-6) + 1e-6)).clamp(0, 1)


def _make_base_grid(h: int, w: int, device, dtype) -> torch.Tensor:
    xs = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    ys = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).unsqueeze(0)  # [1,H,W,2]


def _grid_sample_offset(S: torch.Tensor, vec: torch.Tensor, delta: int, base_grid: torch.Tensor) -> torch.Tensor:
    # S: [B,C,H,W], vec: [B,2,H,W] unit (x,y)
    b, _, h, w = vec.shape
    dx = (2.0 * delta / (w - 1 + 1e-6)) * vec[:, 0]
    dy = (2.0 * delta / (h - 1 + 1e-6)) * vec[:, 1]
    grid = base_grid.expand(b, -1, -1, -1) + torch.stack([dx, dy], dim=-1)
    return F.grid_sample(S, grid, mode="bilinear", padding_mode="border", align_corners=True)


def _coh_4nbr(t: torch.Tensor) -> torch.Tensor:
    # t: [B,2,H,W] unit
    b, _, h, w = t.shape
    x = F.pad(t, (1, 1, 1, 1), mode="replicate")
    tR = x[:, :, 1:1+h, 2:2+w]
    tL = x[:, :, 1:1+h, 0:0+w]
    tU = x[:, :, 0:0+h, 1:1+w]
    tD = x[:, :, 2:2+h, 1:1+w]
    dot = lambda a, b: (a * b).sum(dim=1, keepdim=True).abs()
    return ((dot(t, tR) + dot(t, tL) + dot(t, tU) + dot(t, tD)) / 4.0).clamp(0, 1)


def _haar_dwt(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int]]:
    b, c, h, w = x.shape
    ph, pw = h & 1, w & 1
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    a = x[:, :, 0::2, 0::2]
    b2 = x[:, :, 0::2, 1::2]
    c2 = x[:, :, 1::2, 0::2]
    d = x[:, :, 1::2, 1::2]
    LL = (a + b2 + c2 + d) * 0.5
    LH = (a - b2 + c2 - d) * 0.5
    HL = (a + b2 - c2 - d) * 0.5
    HH = (a - b2 - c2 + d) * 0.5
    return LL, LH, HL, HH, (h, w)


def _haar_idwt(LL: torch.Tensor, LH: torch.Tensor, HL: torch.Tensor, HH: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    b, c, h, w = LL.shape
    a = (LL + LH + HL + HH) * 0.5
    b2 = (LL - LH + HL - HH) * 0.5
    c2 = (LL + LH - HL - HH) * 0.5
    d = (LL - LH - HL + HH) * 0.5
    x = torch.zeros(b, c, h * 2, w * 2, device=LL.device, dtype=LL.dtype)
    x[:, :, 0::2, 0::2] = a
    x[:, :, 0::2, 1::2] = b2
    x[:, :, 1::2, 0::2] = c2
    x[:, :, 1::2, 1::2] = d
    H, W = hw
    return x[:, :, :H, :W]


class _CBA(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, groups=1, act=True):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self, x): return self.act(self.bn(self.conv(x)))


# ----------------- 1) Center Finding (C³) -----------------
class CenterFindingModule(nn.Module):
    # x4: [B,C4,H4,W4], b: [B,Cb,Hb,Wb] -> dict(A,c_xy,c,V,R,rc)
    def __init__(self, c4: int, cb: int, hidden: int = 128):
        super().__init__()
        self.fuse = nn.Sequential(_CBA(c4 + cb, hidden, 3), _CBA(hidden, hidden, 3))
        self.hA = nn.Conv2d(hidden, 1, 1)
        self.hV = nn.Conv2d(hidden, 2, 1)
        self.hR = nn.Conv2d(hidden, 1, 1)

    def forward(self, x4: torch.Tensor, b: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, _, H, W = x4.shape
        b = F.interpolate(b, size=(H, W), mode="bilinear", align_corners=False)
        f = self.fuse(torch.cat([x4, b], dim=1))
        A = _spatial_softmax(self.hA(f))                 # [B,1,H,W]
        c_xy, c = _softargmax_2d(A)                      # [B,2], [B,2] (-1..1)
        V = self.hV(f)
        V = V / (_norm2(V, 1) + 1e-6)                    # [B,2,H,W]
        R = torch.sigmoid(self.hR(f))                    # [B,1,H,W]
        rc = _rc_from_entropy(A)                         # [B,1]
        return {"A": A, "c_xy": c_xy, "c": c, "V": V, "R": R, "rc": rc}


# ----------------- 2) Sector Enhance + Paint-back (OR) -----------------
def _wrap_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def _sector_masks(H: int, W: int, K: int, c: torch.Tensor,
                  r_in: float, r_out0: float, r_out1: float,
                  overlap: float, tau_th: float = 0.10, tau_r: float = 0.03) -> torch.Tensor:
    # c: [B,2] (-1..1) -> masks: [B,2K,H,W]
    device, dtype = c.device, c.dtype
    B = c.shape[0]
    xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xx = xx.unsqueeze(0).expand(B, -1, -1)
    yy = yy.unsqueeze(0).expand(B, -1, -1)
    dx = xx - c[:, 0].view(B, 1, 1)
    dy = yy - c[:, 1].view(B, 1, 1)
    r = torch.sqrt(dx * dx + dy * dy) / math.sqrt(2.0)
    th = torch.atan2(dy, dx)

    w = 2 * math.pi / K
    halfw = w * (0.5 + overlap)
    thk = (-math.pi + w * (torch.arange(K, device=device, dtype=dtype) + 0.5)).view(1, K, 1, 1)
    ang = torch.sigmoid((halfw - _wrap_pi(th.unsqueeze(1) - thk).abs()) / (tau_th + 1e-6))

    def rg(rmin, rmax):
        return torch.sigmoid((r - rmin) / (tau_r + 1e-6)) * torch.sigmoid((rmax - r) / (tau_r + 1e-6))

    inner = ang * rg(0.0, r_in).unsqueeze(1)
    outer = ang * rg(r_out0, r_out1).unsqueeze(1)
    return torch.cat([inner, outer], dim=1)  # [B,2K,H,W]


def _tokenize(X: torch.Tensor, M: torch.Tensor, R: Optional[torch.Tensor] = None, eps: float = 1e-6) -> torch.Tensor:
    # X:[B,C,H,W], M:[B,K,H,W] -> T:[B,K,C]
    Wm = M if R is None else (M * R)
    den = Wm.sum((2, 3), keepdim=True).clamp_min(eps)    # [B,K,1,1]
    num = torch.einsum("bkhw,bchw->bkc", Wm, X)
    return num / den.view(X.shape[0], M.shape[1], 1)


def _writeback(T: torch.Tensor, M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # T:[B,K,C], M:[B,K,H,W] -> Xrec:[B,C,H,W]
    B, K, H, W = M.shape
    C = T.shape[-1]
    num = (M.unsqueeze(2) * T.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # [B,C,H,W]
    den = M.sum(dim=1, keepdim=True).clamp_min(eps)
    return num / den


class SectorEnhancePaintBackModule(nn.Module):
    # X:[B,C,H,W], center_state:{c:[B,2], R:[B,1,*,*] opt, rc:[B,1] opt} -> [B,C,H,W]
    def __init__(self, C: int, K: int = 12, heads: int = 4,
                 r_in: float = 0.35, r_out0: float = 0.35, r_out1: float = 0.85, overlap: float = 0.10):
        super().__init__()
        self.K, self.r_in, self.r_out0, self.r_out1, self.overlap = K, r_in, r_out0, r_out1, overlap
        self.pos = nn.Parameter(torch.zeros(1, 2 * K, C))
        self.mha = nn.MultiheadAttention(C, heads, batch_first=True)
        self.dw = nn.Conv1d(C, C, 3, padding=1, groups=C, bias=False)
        self.pw = nn.Conv1d(C, C, 1, bias=False)
        self.gate = nn.Sequential(nn.Linear(3, 32), nn.SiLU(inplace=True), nn.Linear(32, 1))
        self.proj = nn.Conv2d(C, C, 1, bias=False)
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, X: torch.Tensor, center_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        B, C, H, W = X.shape
        c = center_state["c"]                                   # [B,2]
        R = center_state.get("R", None)
        if R is not None and R.shape[-2:] != (H, W):
            R = F.interpolate(R, size=(H, W), mode="bilinear", align_corners=False)
        rc = center_state.get("rc", None)
        if rc is None:
            rc = torch.ones(B, 1, device=X.device, dtype=X.dtype)

        M = _sector_masks(H, W, self.K, c, self.r_in, self.r_out0, self.r_out1, self.overlap)  # [B,2K,H,W]
        T = _tokenize(X, M, R=R) + self.pos                                                   # [B,2K,C]

        t1 = self.pw(self.dw(T.transpose(1, 2))).transpose(1, 2)
        T = T + t1

        Tout, attn = self.mha(T, T, T, need_weights=True, average_attn_weights=True)          # attn:[B,2K,2K]
        T = T + Tout

        # entropy confidence per token from attn row distribution
        p = attn.clamp_min(1e-8)
        ent = -(p * p.log()).sum(dim=-1)                                                      # [B,2K]
        conf = (1 - ent / (math.log(attn.shape[-1] + 1e-6) + 1e-6)).clamp(0, 1)               # [B,2K]

        if R is None:
            Rk = torch.ones(B, 2 * self.K, device=X.device, dtype=X.dtype)
        else:
            num = (M * R).sum((2, 3))
            den = M.sum((2, 3)).clamp_min(1e-6)
            Rk = (num / den).clamp(0, 1)

        feat = torch.stack([rc.expand(-1, 2 * self.K), Rk, conf], dim=-1)                      # [B,2K,3]
        g = torch.sigmoid(self.gate(feat)).squeeze(-1)                                        # [B,2K]
        G = (M * g.view(B, 2 * self.K, 1, 1)).sum(dim=1, keepdim=True)                        # [B,1,H,W]

        Xrec = self.proj(_writeback(T, M))                                                     # [B,C,H,W]
        return X + self.gamma * G * (Xrec - X)


# ----------------- 3) Skip Connection Purification (SCP / DCWP-Vlocal) -----------------
class SkipConnectionPurificationModule(nn.Module):
    # S:[B,C,H,W], D:[B,Cd,H,W] -> PurifiedSkip:[B,C,H,W]
    def __init__(self, C: int, Cd: int, deltas: List[int] = (1, 2, 4),
                 tau_amp: float = 0.75, gamma_branch: float = 1.5, rho_inject: float = 0.5):
        super().__init__()
        self.deltas = list(deltas)
        self.tau_amp = tau_amp
        self.gamma_branch = gamma_branch
        self.rho_inject = rho_inject

        self.v_head = nn.Sequential(_CBA(C, C, 3), nn.Conv2d(C, 2, 1))
        self.need = nn.Sequential(_CBA(C + Cd, C, 3), nn.Conv2d(C, 1, 1))
        self.w_delta = nn.Sequential(_CBA(Cd, Cd, 3), nn.Conv2d(Cd, len(self.deltas), 1))

        self.gLH = nn.Sequential(_CBA(C + Cd, C, 3), nn.Conv2d(C, 1, 1))
        self.gHL = nn.Sequential(_CBA(C + Cd, C, 3), nn.Conv2d(C, 1, 1))
        self.gHH = nn.Sequential(_CBA(C + Cd, C, 3), nn.Conv2d(C, 1, 1))

        self.m = nn.Sequential(_CBA(Cd, Cd, 3), nn.Conv2d(Cd, 1, 1))
        self.proj = nn.Conv2d(C, C, 1, bias=False)

        self.beta = nn.Parameter(torch.tensor(3.0))
        self.kappa = nn.Parameter(torch.tensor(3.0))

    def forward(self, S: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        B, C, H, W = S.shape
        base = _make_base_grid(H, W, S.device, S.dtype)

        v = self.v_head(S)                                    # [B,2,H,W]
        s = _norm2(v, 1)                                      # [B,1,H,W]
        t = v / (s + 1e-6)
        n = torch.cat([-t[:, 1:2], t[:, 0:1]], dim=1)

        q_amp = (s / (s + self.tau_amp)).clamp(0, 1)
        q_coh = _coh_4nbr(t)
        q = (q_amp * q_coh).clamp(0, 1)

        g_need = torch.sigmoid(self.need(torch.cat([S, D], dim=1)))         # [B,1,H,W]
        E_c = torch.sqrt(torch.clamp((S * S).mean(1, keepdim=True), 1e-6))   # [B,1,H,W]
        branch = (q_coh.clamp(0, 1) ** self.gamma_branch)

        bridges, Gs = [], []
        for dlt in self.deltas:
            Spt = _grid_sample_offset(S, t, +dlt, base)
            Smt = _grid_sample_offset(S, t, -dlt, base)
            Spn = _grid_sample_offset(S, n, +dlt, base)
            Smn = _grid_sample_offset(S, n, -dlt, base)

            bridge = 0.5 * (Spt + Smt)
            bridges.append(bridge)

            E_t = torch.sqrt(torch.clamp((0.5 * (Spt * Spt + Smt * Smt)).mean(1, keepdim=True), 1e-6))
            E_n = torch.sqrt(torch.clamp((0.5 * (Spn * Spn + Smn * Smn)).mean(1, keepdim=True), 1e-6))

            g_gap = torch.sigmoid(self.beta * (E_t - E_c))
            g_anti = torch.sigmoid(self.kappa * (E_t - E_n))
            Gs.append((g_need * q * g_gap * g_anti * branch).clamp(0, 1))

        w = F.softmax(self.w_delta(D), dim=1)                                  # [B,|Δ|,H,W]
        if len(self.deltas) > 1:
            lam = 1.5
            dmin = min(self.deltas)
            scales = [torch.exp(-lam * float(d - dmin) * (1 - q_coh.detach())) for d in self.deltas]
            scale = torch.cat(scales, dim=1)
            w = w * scale
            w = w / w.sum(1, keepdim=True).clamp_min(1e-6)

        Bridge = torch.zeros_like(S)
        G = torch.zeros(B, 1, H, W, device=S.device, dtype=S.dtype)
        for i in range(len(self.deltas)):
            wi = w[:, i:i+1]
            Bridge = Bridge + wi * bridges[i]
            G = G + wi * Gs[i]
        G = G.clamp(0, 1)

        S_fill = (1 - G) * S + G * Bridge

        LL, LH, HL, HH, hw = _haar_dwt(S_fill)
        Dd = F.avg_pool2d(D, 2, 2)
        qd = F.avg_pool2d(q, 2, 2).clamp(0, 1)

        def _bg(band, gnet):
            g0 = torch.sigmoid(gnet(torch.cat([band, Dd], dim=1)))            # [B,1,h,w]
            g = 1 - qd + qd * g0
            return band * g

        LH = _bg(LH, self.gLH)
        HL = _bg(HL, self.gHL)
        HH = _bg(HH, self.gHH)
        S_wav = _haar_idwt(LL, LH, HL, HH, hw)

        m = torch.sigmoid(self.m(D))                                          # [B,1,H,W]
        m = (m * (1 - self.rho_inject * G.detach()) * q).clamp(0, 1)
        return S_wav + m * self.proj(S_wav)
