import torch
from torchaudio.pipelines import SQUIM_OBJECTIVE


def _run_case(nombre, forward_fn, device):
    """Corre un caso, imprime predicciones y gradientes, devuelve True/False."""
    print(f"\n--- {nombre} ---")

    audio = torch.randn(1, 64000, device=device, requires_grad=True)

    try:
        stoi_hat, pesq_hat, si_sdr_hat = forward_fn(audio)
    except RuntimeError as e:
        print(f"  ❌ ERROR en forward/backward: {e}")
        return False

    print(f"  Predicciones -> STOI={stoi_hat.item():.3f}, "
          f"PESQ={pesq_hat.item():.3f}, SI-SDR={si_sdr_hat.item():.3f}")

    loss = -pesq_hat.mean()
    print(f"  Loss: {loss.item():.4f}")

    try:
        loss.backward()
    except RuntimeError as e:
        print(f"  ❌ ERROR en backward: {e}")
        return False

    # --- Variables clave a inspeccionar ---
    if audio.grad is None:
        print("  ❌ FALLO: audio.grad es None -> el grafo no llegó hasta acá.")
        return False

    grad_norm = audio.grad.norm().item()   # magnitud global del gradiente
    grad_max = audio.grad.abs().max().item()  # pico máximo (detecta explosión)
    grad_mean = audio.grad.abs().mean().item()  # promedio (detecta gradiente "plano")

    print(f"  grad_norm  = {grad_norm:.6e}   (¿> 0? si es 0 el grafo se cortó)")
    print(f"  grad_max   = {grad_max:.6e}   (si es enorme, ojo con exploding grad)")
    print(f"  grad_mean  = {grad_mean:.6e}   (si es ~0 pero norm>0, gradiente disperso)")

    if grad_norm == 0:
        print("  ❌ FALLO: gradiente todo cero.")
        return False
    if not torch.isfinite(audio.grad).all():
        print("  ❌ FALLO: gradiente con NaN/Inf.")
        return False

    print("  ✅ PASS: gradiente finito y no nulo.")
    return True


def caso_1_sin_cudnn(device):
    """Desactiva cuDNN ANTES del forward (clave: no después)."""
    squim_model = SQUIM_OBJECTIVE.get_model().to(device)
    squim_model.eval()
    for p in squim_model.parameters():
        p.requires_grad_(False)

    def forward_fn(audio):
        with torch.backends.cudnn.flags(enabled=False):
            return squim_model(audio)

    return forward_fn


def caso_2_train_mode(device):
    """Modelo en .train() (reserve buffer de cuDNN), pesos congelados."""
    squim_model = SQUIM_OBJECTIVE.get_model().to(device)
    squim_model.train()
    for p in squim_model.parameters():
        p.requires_grad_(False)

    def forward_fn(audio):
        return squim_model(audio)

    return forward_fn


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    resultados = {}

    resultados["cuDNN desactivado (forward+backward)"] = _run_case(
        "Solución 1: cudnn.flags(enabled=False) antes del forward",
        caso_1_sin_cudnn(device),
        device,
    )

    resultados["Modelo en .train() con pesos congelados"] = _run_case(
        "Solución 2: squim_model.train() + requires_grad_(False)",
        caso_2_train_mode(device),
        device,
    )

    print("\n=== RESUMEN ===")
    for nombre, ok in resultados.items():
        estado = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {estado}  -  {nombre}")

    return all(resultados.values())


if __name__ == "__main__":
    exit(0 if main() else 1)
