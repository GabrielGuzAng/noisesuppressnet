"""
tests/test_causality.py

Verifica las propiedades causales del sistema, en los DOS niveles en que se
manifiestan, que no son el mismo y que durante mucho tiempo se reportaron como
uno solo.

1. CAUSALIDAD DE FRAME (exacta). La red no mira ningún frame futuro: perturbar
   la magnitud de entrada desde el frame t no cambia ni un bit de la salida en
   los frames anteriores. Es lo que consiguen el truncamiento temporal del
   encoder y el padding asimétrico del decoder, y es lo que el proyecto venía
   reportando como "diff = 0.00e+00".

2. LATENCIA ALGORÍTMICA A NIVEL DE SEÑAL (10 ms). Sobre la forma de onda el
   sistema SÍ tiene lookahead: la salida en la muestra n depende de la entrada
   hasta n + (n_fft − hop). Con n_fft=320 y hop=160 son 160 muestras = 10.0 ms.

   No viene del `center=True` de la STFT. Se verificó que `center=False` con
   padding causal da exactamente el mismo retardo: es la latencia inherente al
   overlap-add de la síntesis, que reparte la contribución de cada frame hacia
   atrás una ventana entera aunque el análisis sea estrictamente causal.

   Consecuencia: los 10 ms son propios de la configuración STFT que especifica
   el paper base (ventana 20 ms, hop 10 ms), no una desviación nuestra. Tan &
   Wang 2018, sección 1, cita ese mismo valor como límite de lo aceptable.
   Reducirlo exigiría ventanas asimétricas de análisis/síntesis.

Uso:
    python -m pytest tests/test_causality.py -v
    python -m tests.test_causality
"""
import pytest
import torch

from models.crn import CRN
from stft import STFTHelper

SAMPLE_RATE = 16000
SEED = 42


def _model():
    torch.manual_seed(SEED)
    return CRN().eval()


def _measure_signal_lookahead(helper, transform, n_samples=16000, split=None):
    """Samples of future the output depends on, measured, not derived.

    Perturbs the input from `split` onwards and finds the earliest output sample
    that changes. `transform` is any non-linear operation on the magnitude: a
    linear pass-through would round-trip to the identity and hide the effect.
    """
    split = split if split is not None else n_samples // 2
    torch.manual_seed(SEED)
    x = torch.randn(1, n_samples) * 0.1
    x_mod = x.clone()
    x_mod[0, split:] += torch.randn(n_samples - split) * 0.5

    outs = []
    with torch.no_grad():
        for sig in (x, x_mod):
            mag, phase = helper.to_spec(sig)
            outs.append(helper.from_spec(transform(mag), phase, length=n_samples))

    diff = (outs[0] - outs[1]).abs()
    changed = (diff > 1e-6).nonzero()
    first_changed = changed[0, 1].item() if len(changed) else n_samples
    return split - first_changed


# ── Nivel 1: causalidad de frame ───────────────────────────────────────────

@pytest.mark.parametrize("n_frames,split", [(101, 50), (101, 25), (200, 137), (60, 1)])
def test_frame_level_causality_is_exact(n_frames, split):
    """Perturbar frames >= split no altera la salida anterior, bit a bit."""
    model = _model()
    torch.manual_seed(SEED)
    mag = torch.rand(1, n_frames, 161)
    mag_mod = mag.clone()
    mag_mod[:, split:, :] += 1.0

    with torch.no_grad():
        out, out_mod = model(mag), model(mag_mod)

    delta = (out[:, :split, :] - out_mod[:, :split, :]).abs().max().item()
    assert delta == 0.0, (
        f"El modelo filtra información del futuro: al perturbar desde el frame "
        f"{split} la salida previa cambió en {delta:.3e} (debe ser exactamente 0)"
    )


def test_frame_causality_first_changed_frame_is_the_perturbed_one():
    """El cambio arranca exactamente en el frame perturbado, ni antes ni después."""
    model = _model()
    torch.manual_seed(SEED)
    mag = torch.rand(1, 101, 161)
    mag_mod = mag.clone()
    mag_mod[:, 50:, :] += 1.0

    with torch.no_grad():
        diff = (model(mag) - model(mag_mod)).abs()

    changed = (diff > 1e-6).nonzero()
    assert len(changed) > 0, "La perturbación no produjo ningún cambio"
    assert changed[0, 1].item() == 50


def test_output_shape_matches_input_frames():
    """El truncamiento del encoder y el padding del decoder preservan T."""
    model = _model()
    for n_frames in (2, 7, 50, 101, 401):
        with torch.no_grad():
            out = model(torch.rand(1, n_frames, 161))
        assert out.shape == (1, n_frames, 161), f"T no se preservó con {n_frames} frames"


def test_minimum_chunk_on_cpu_is_two_frames():
    """En CPU el chunk mínimo procesable son 2 frames, no 1.

    Con T=1 el conv_transpose2d del decoder produce un intermedio de largo 0 y
    el backend de CPU lo rechaza (CUDA lo tolera). Condiciona cualquier
    implementación de streaming sobre el hardware objetivo: no se puede
    procesar frame a frame, hay que bufferear de a 2 (20 ms de audio).
    """
    model = _model()
    with pytest.raises(RuntimeError):
        with torch.no_grad():
            model(torch.rand(1, 1, 161))

    with torch.no_grad():
        assert model(torch.rand(1, 2, 161)).shape == (1, 2, 161)


# ── Nivel 2: latencia algorítmica a nivel de señal ─────────────────────────

@pytest.mark.parametrize("n_fft,hop", [(320, 160), (160, 80), (640, 160), (320, 80)])
@pytest.mark.parametrize("causal", [False, True])
def test_signal_lookahead_equals_nfft_minus_hop(n_fft, hop, causal):
    """El lookahead sigue n_fft − hop, y NO depende del modo de framing.

    Es la comprobación de que `center=False` con padding causal no reduce la
    latencia: el retardo lo impone el overlap-add de la síntesis.
    """
    helper = STFTHelper(n_fft=n_fft, hop_length=hop, causal=causal)
    helper._window = torch.hamming_window(n_fft)
    lookahead = _measure_signal_lookahead(helper, lambda m: m ** 0.8)
    assert lookahead == n_fft - hop, (
        f"n_fft={n_fft} hop={hop} causal={causal}: lookahead medido {lookahead}, "
        f"esperado {n_fft - hop}"
    )


def test_end_to_end_latency_is_10ms():
    """Con el modelo real y la configuración del proyecto: 160 muestras = 10 ms."""
    model = _model()
    helper = STFTHelper(n_fft=320, hop_length=160)
    lookahead = _measure_signal_lookahead(helper, lambda m: model(m))
    assert lookahead == 160
    assert lookahead / SAMPLE_RATE * 1000 == pytest.approx(10.0)


def test_causal_mode_does_not_reduce_latency():
    """Explícito, porque es contraintuitivo y motivó una decisión de diseño."""
    model = _model()
    default_mode = _measure_signal_lookahead(
        STFTHelper(n_fft=320, hop_length=160, causal=False), lambda m: model(m))
    causal_mode = _measure_signal_lookahead(
        STFTHelper(n_fft=320, hop_length=160, causal=True), lambda m: model(m))
    assert default_mode == causal_mode == 160


# ── Guarda de regresión: el framing por defecto no puede cambiar ───────────

def test_default_framing_unchanged():
    """V1..V3e se entrenaron con este framing; cambiarlo invalida su evaluación."""
    helper = STFTHelper()
    assert helper.causal is False, "El default DEBE seguir siendo center=True"

    torch.manual_seed(SEED)
    x = torch.randn(1, 64000) * 0.1
    mag, phase = helper.to_spec(x)
    assert mag.shape[1] == 401, f"401 frames para 4 s; salieron {mag.shape[1]}"

    recon = helper.from_spec(mag, phase, length=64000)
    assert (recon - x).abs().max().item() < 1e-5


def test_causal_mode_preserves_frame_count_and_round_trip():
    """El modo causal mantiene shapes y reconstruye, aunque no baje la latencia."""
    helper = STFTHelper(causal=True)
    torch.manual_seed(SEED)
    x = torch.randn(1, 64000) * 0.1
    mag, phase = helper.to_spec(x)
    assert mag.shape[1] == 401

    recon = helper.from_spec(mag, phase, length=64000)
    assert recon.shape[-1] == 64000
    assert (recon - x).abs().max().item() < 1e-5


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "--no-header", "-q"]))
