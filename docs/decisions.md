## Simplificaciones para el preproyecto

1. STFT: torch.stft estándar en vez de implementación custom con F.relu sobre Hamming.
   Justificación: equivalente numéricamente (Hamming no tiene valores negativos), torch.stft es C++ optimizado.

2. Padding temporal: simétrico (no estrictamente causal).
   Justificación: el repo de referencia (CRN-causal) también usa padding simétrico en la STFT.
   Esta limitación se documenta y se evalúa estrictamente en el informe final.

3. LSTM no agrupada (G=1) en vez del GCRN del repo (G=2).
   Justificación: el paper original usa LSTM estándar; el repo agrega grouping como extensión posterior.


## Causalidad temporal — detalles de implementación

El paper especifica convoluciones causales (kernel 2x3, sin lookahead temporal) 
pero no detalla cómo implementarlas en PyTorch. El repo de referencia 
(JupiterEthan/CRN-causal) revela el patrón:

**Encoder:** después de cada conv2d con padding=(1,0), se trunca el último 
frame temporal con [:, :, :-1, :]. Esto elimina el lookahead que PyTorch 
introduce por su padding simétrico.

**Decoder:** después de cada conv_transpose2d con padding=(1,0), se aplica 
F.pad(x, [0,0,1,0]) que agrega 1 frame de padding solo al pasado.

Esto mantiene la dimensión T constante a lo largo de la red y garantiza 
causalidad estricta a nivel de frame.


# Tests de validación del modelo:
- "Causalidad estricta: diff = 0.00e+00 con input modificado en t≥50"
- "Reconstrucción STFT: error round-trip = 9.54e-07"
- "Pipeline integrado audio→STFT→CRN→audio: funcional con gradientes finitos"
