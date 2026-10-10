import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))

try:
    from thop import profile
except Exception:
    profile = None

# === Pluggable modules (ALL ON) ===
from C_polar import CenterFindingModule, SectorEnhancePaintBackModule, SkipConnectionPurificationModule


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _to_set(scales):
    if scales is None:
        return set()
    if isinstance(scales, (list, tuple, set)):
        return set(scales)
    return {scales}


class EGCSNet(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        features=(64, 128, 256, 512),

        # Cascade injection strength: A_out -> B_in
        inj_alpha: float = 0.5,

        # OR fusion strength (F1)
        alpha_s3: float = 1.0,
        alpha_s4: float = 1.0,

        # SCP placement (B self-skip)
        scp_scales=("S3",),  # {"S3","S4"}

        # OR params
        K: int = 12,
        heads: int = 4,
    ):
        super().__init__()
        f = list(features)
        self.pool = nn.MaxPool2d(2, 2)

        self.inj_alpha = float(inj_alpha)
        self.alpha_s3 = float(alpha_s3)
        self.alpha_s4 = float(alpha_s4)
        self.scp_scales = _to_set(scp_scales)

        # -------- A encoder --------
        self.A_down1 = DoubleConv(in_channels, f[0])
        self.A_down2 = DoubleConv(f[0], f[1])
        self.A_down3 = DoubleConv(f[1], f[2])
        self.A_down4 = DoubleConv(f[2], f[3])
        self.A_bot = DoubleConv(f[3], f[3] * 2)

        # -------- A decoder (aux, produces A_out for injection) --------
        self.A_up1_t = nn.ConvTranspose2d(f[3] * 2, f[3], 2, 2)
        self.A_up2_t = nn.ConvTranspose2d(f[3], f[2], 2, 2)
        self.A_up3_t = nn.ConvTranspose2d(f[2], f[1], 2, 2)
        self.A_up4_t = nn.ConvTranspose2d(f[1], f[0], 2, 2)

        self.A_up1_c = DoubleConv(f[3] * 2, f[3])
        self.A_up2_c = DoubleConv(f[2] * 2, f[2])
        self.A_up3_c = DoubleConv(f[1] * 2, f[1])
        self.A_up4_c = DoubleConv(f[0] * 2, f[0])

        self.outA = nn.Conv2d(f[0], out_channels, 1)

        # -------- cascade injection: A_out -> B_in --------
        self.inj_proj = nn.Conv2d(out_channels, in_channels, 1, bias=False)

        # -------- B encoder --------
        self.B_down1 = DoubleConv(in_channels, f[0])
        self.B_down2 = DoubleConv(f[0], f[1])
        self.B_down3 = DoubleConv(f[1], f[2])
        self.B_down4 = DoubleConv(f[2], f[3])
        self.B_bot = DoubleConv(f[3], f[3] * 2)

        # -------- B decoder (main) --------
        self.B_up1_t = nn.ConvTranspose2d(f[3] * 2, f[3], 2, 2)
        self.B_up2_t = nn.ConvTranspose2d(f[3], f[2], 2, 2)
        self.B_up3_t = nn.ConvTranspose2d(f[2], f[1], 2, 2)
        self.B_up4_t = nn.ConvTranspose2d(f[1], f[0], 2, 2)

        self.B_up1_c = DoubleConv(f[3] * 2, f[3])
        self.B_up2_c = DoubleConv(f[2] * 2, f[2])
        self.B_up3_c = DoubleConv(f[1] * 2, f[1])
        self.B_up4_c = DoubleConv(f[0] * 2, f[0])

        self.outB = nn.Conv2d(f[0], out_channels, 1)

        # =========================
        # Modules (ALL ON)
        # =========================
        self.use_center = True
        self.use_or = True
        self.use_scp = True

        # Center: fixed (a2 + ab)
        self.center = CenterFindingModule(c4=f[1], cb=f[3] * 2, hidden=128)

        # OR: fixed at S3 (C=f[1]) and S4 (C=f[0])
        self.or_s3 = SectorEnhancePaintBackModule(C=f[1], K=K, heads=heads)
        self.or_s4 = SectorEnhancePaintBackModule(C=f[0], K=K, heads=heads)

        # SCP: B self-skip entrances
        self.scp_b_s3 = SkipConnectionPurificationModule(C=f[1], Cd=f[1]) if "S3" in self.scp_scales else None
        self.scp_b_s4 = SkipConnectionPurificationModule(C=f[0], Cd=f[0]) if "S4" in self.scp_scales else None

    # =========================
    # A branch
    # =========================
    def _A_encoder(self, x: torch.Tensor):
        a1 = self.A_down1(x)
        a2 = self.A_down2(self.pool(a1))
        a3 = self.A_down3(self.pool(a2))
        a4 = self.A_down4(self.pool(a3))
        ab = self.A_bot(self.pool(a4))
        return a1, a2, a3, a4, ab

    def _A_decoder(self, ab, a1, a2, a3, a4):
        a_p1 = self.A_up1_t(ab)
        a_up1 = self.A_up1_c(torch.cat([a4, a_p1], dim=1))

        a_p2 = self.A_up2_t(a_up1)
        a_up2 = self.A_up2_c(torch.cat([a3, a_p2], dim=1))

        a_p3 = self.A_up3_t(a_up2)
        a_up3 = self.A_up3_c(torch.cat([a2, a_p3], dim=1))

        a_p4 = self.A_up4_t(a_up3)
        a_up4 = self.A_up4_c(torch.cat([a1, a_p4], dim=1))

        yA = self.outA(a_up4)
        return yA, (a_up3, a_up4)

    # =========================
    # B branch
    # =========================
    def _B_encoder(self, x: torch.Tensor):
        b1 = self.B_down1(x)
        b2 = self.B_down2(self.pool(b1))
        b3 = self.B_down3(self.pool(b2))
        b4 = self.B_down4(self.pool(b3))
        bb = self.B_bot(self.pool(b4))
        return b1, b2, b3, b4, bb

    def _B_decoder(self, bb, b1, b2, b3, b4, a_up3_or=None, a_up4_or=None):
        b_p1 = self.B_up1_t(bb)
        b_up1 = self.B_up1_c(torch.cat([b4, b_p1], dim=1))

        b_p2 = self.B_up2_t(b_up1)
        b_up2 = self.B_up2_c(torch.cat([b3, b_p2], dim=1))

        b_p3 = self.B_up3_t(b_up2)
        # SCP @S3
        if self.use_scp and (self.scp_b_s3 is not None):
            b2p = self.scp_b_s3(S=b2, D=b_p3)
        else:
            b2p = b2
        b_up3 = self.B_up3_c(torch.cat([b2p, b_p3], dim=1))

        # OR-F1 injection @S3
        if self.use_or and (a_up3_or is not None):
            b_up3 = b_up3 + self.alpha_s3 * a_up3_or

        b_p4 = self.B_up4_t(b_up3)
        # SCP @S4
        if self.use_scp and (self.scp_b_s4 is not None):
            b1p = self.scp_b_s4(S=b1, D=b_p4)
        else:
            b1p = b1
        b_up4 = self.B_up4_c(torch.cat([b1p, b_p4], dim=1))

        # OR-F1 injection @S4
        if self.use_or and (a_up4_or is not None):
            b_up4 = b_up4 + self.alpha_s4 * a_up4_or

        return self.outB(b_up4)

    # =========================
    # Forward
    # =========================
    def forward(self, x: torch.Tensor, return_aux: bool = False):
        # A encoder
        a1, a2, a3, a4, ab = self._A_encoder(x)
        center_state = self.center(a2, ab) if self.use_center else None

        # A decoder -> logits + features for OR
        yA, (a_up3, a_up4) = self._A_decoder(ab, a1, a2, a3, a4)

        # OR on A @S3/S4
        if self.use_or:
            a_up3_or = self.or_s3(a_up3, center_state)
            a_up4_or = self.or_s4(a_up4, center_state)
        else:
            a_up3_or = None
            a_up4_or = None

        # Cascade injection: A_out -> B_in
        xB = x + self.inj_alpha * self.inj_proj(torch.sigmoid(yA))

        # B encoder/decoder
        b1, b2, b3, b4, bb = self._B_encoder(xB)
        yB = self._B_decoder(bb, b1, b2, b3, b4, a_up3_or=a_up3_or, a_up4_or=a_up4_or)

        if return_aux:
            return yB, {"unetA": yA, "unetB": yB, "center_state": center_state}
        return yB


def test():
    m = EGCSNet(in_channels=3, out_channels=1, inj_alpha=0.5)
    x = torch.randn(2, 3, 224, 224)
    y, aux = m(x, return_aux=True)
    print("out:", y.shape, "A:", aux["unetA"].shape, "B:", aux["unetB"].shape)
    if profile is not None:
        flops, params = profile(m, inputs=(x,))
        print("FLOPs:", flops, "Params:", params)


if __name__ == "__main__":
    test()
