"""asr/wer.py

WER of the hypotheses written by `asr.transcribe`.

Runs in `.venv-asr`, next to the transcriber: jiwer and the normalizer
(transformers) are pinned there, and `.venv` / `requirements.txt` stay untouched.

Estimand: aggregate WER = total errors / total reference words, i.e.
sum(S+D+I) / sum(n_words). It is NOT the mean of per-utterance WERs, which
over-weights short utterances. S, D, I, H and n_words are kept per utterance,
which is what makes the aggregate and a clustered bootstrap computable later.

Normalization is identical for reference and hypothesis: Whisper's
BasicTextNormalizer (as vendored in transformers) with remove_diacritics=True —
lowercase, symbols/punctuation to spaces, diacritics dropped, whitespace
collapsed. Its default is remove_diacritics=False, so the flag is explicit.

Usage::

    .venv-asr/bin/python -m asr.wer        # hand-checked case + smoke WER(clean)
"""
import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import jiwer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SMOKE_HYP = PROJECT_ROOT / "data" / "asr_hyp" / "asr_smoke" / "hypotheses.json"
SMOKE_REPORT = PROJECT_ROOT / "results" / "asr_smoke_wer_clean.json"

logger = logging.getLogger("asr.wer")

_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)


def normalize(text: str) -> str:
    """Normalization applied identically to reference and hypothesis."""
    return _normalizer(text).strip()


def utterance_counts(reference: str, hypothesis: str) -> dict:
    """Word-level alignment counts for one utterance, after normalization.

    Returns:
        Dict with S, D, I, H, n_words (reference words) and both normalized texts.
    """
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        raise ValueError(f"Referencia vacía tras normalizar: {reference!r}")
    n_words = len(ref.split())
    if not hyp:
        # Hipótesis vacía: todo borrado. Se cuenta explícito en vez de depender
        # de cómo trate jiwer la cadena vacía.
        return {"S": 0, "D": n_words, "I": 0, "H": 0, "n_words": n_words,
                "ref_norm": ref, "hyp_norm": hyp}
    out = jiwer.process_words(ref, hyp)
    counts = {"S": out.substitutions, "D": out.deletions, "I": out.insertions,
              "H": out.hits, "n_words": n_words, "ref_norm": ref, "hyp_norm": hyp}
    if counts["S"] + counts["D"] + counts["H"] != n_words:
        raise RuntimeError(f"Alineación inconsistente: {counts}")
    return counts


def aggregate_wer(rows: list[dict]) -> float:
    """Total errors over total reference words (NOT the mean of per-utterance WERs)."""
    errors = sum(r["S"] + r["D"] + r["I"] for r in rows)
    return errors / sum(r["n_words"] for r in rows)


def self_check() -> None:
    """Hand-built case where the aggregate and the mean of WERs differ.

    utt A: 1 reference word, wrong      -> 1 error / 1 word,  WER_A = 1.0
    utt B: 9 reference words, 1 inserted -> 1 error / 9 words, WER_B = 0.111
    aggregate = 2 / 10 = 0.200 ; mean of WERs = (1.0 + 0.111) / 2 = 0.556
    """
    a = utterance_counts("Hola.", "chau")
    b = utterance_counts("uno dos tres cuatro cinco seis siete ocho nueve",
                         "uno dos tres cuatro cinco seis siete ocho nueve diez")
    assert (a["S"], a["D"], a["I"], a["n_words"]) == (1, 0, 0, 1), a
    assert (b["S"], b["D"], b["I"], b["n_words"]) == (0, 0, 1, 9), b
    agg = aggregate_wer([a, b])
    mean = (1 / 1 + 1 / 9) / 2
    assert abs(agg - 0.2) < 1e-12, agg
    assert abs(mean - 0.5556) < 1e-4
    # jiwer sobre la lista entera calcula el mismo estimando (errores/palabras):
    # control cruzado de que la suma por utterance no pierde nada.
    assert abs(jiwer.wer([a["ref_norm"], b["ref_norm"]],
                         [a["hyp_norm"], b["hyp_norm"]]) - agg) < 1e-12
    # Hipótesis vacía: todo borrado.
    e = utterance_counts("dos palabras", "")
    assert (e["D"], e["n_words"]) == (2, 2), e
    # Normalización simétrica: mayúsculas, puntuación y diacríticos no son errores.
    n = utterance_counts("¿Qué pasó, Martín?", "que paso martin")
    assert n["S"] + n["D"] + n["I"] == 0, n
    logger.info("self_check OK: agregado %.3f vs media de WERs %.3f", agg, mean)


def score(hyp_path: Path) -> dict:
    """WER per corpus and model for the governing pass, plus pass-to-pass changes.

    Args:
        hyp_path: `hypotheses.json` written by `asr.transcribe`.

    Returns:
        Report dict (also carries the per-utterance counts).
    """
    data = json.loads(hyp_path.read_text())
    utts = {u["utt_id"]: u for u in data["utterances"]}
    governing = data["governing_pass"]
    report = {"hypotheses": str(hyp_path.relative_to(PROJECT_ROOT)),
              "governing_pass": governing, "models": {}}

    for model_name, passes in data["runs"].items():
        by_pass = [{h["utt_id"]: h["hypothesis"] for h in p["hypotheses"]} for p in passes]
        per_utt = []
        for uid, u in utts.items():
            c = utterance_counts(u["reference"], by_pass[governing][uid])
            per_utt.append({"utt_id": uid, "corpus": u["corpus"],
                            "hypothesis": by_pass[governing][uid], **c})

        by_corpus = defaultdict(list)
        for r in per_utt:
            by_corpus[r["corpus"]].append(r)

        changes = {}
        other = [i for i in range(len(by_pass)) if i != governing]
        for i in other:
            raw = [uid for uid in utts if by_pass[i][uid] != by_pass[governing][uid]]
            norm = [uid for uid in utts
                    if normalize(by_pass[i][uid]) != normalize(by_pass[governing][uid])]
            changes[f"pass_{i}_vs_{governing}"] = {
                "n_utterances": len(utts),
                "frac_changed_raw": len(raw) / len(utts),
                "frac_changed_normalized": len(norm) / len(utts),
                "changed_utt_ids": raw,
            }

        report["models"][model_name] = {
            "wer_by_corpus": {
                corpus: {"wer": aggregate_wer(rows),
                         "n_utterances": len(rows),
                         "n_words": sum(r["n_words"] for r in rows),
                         "S": sum(r["S"] for r in rows), "D": sum(r["D"] for r in rows),
                         "I": sum(r["I"] for r in rows)}
                for corpus, rows in sorted(by_corpus.items())},
            "pass_changes": changes,
            "timing": [{"pass_index": p["pass_index"], "elapsed_s": p["elapsed_s"],
                        "sec_per_audio_min": p["sec_per_audio_min"]} for p in passes],
            "per_utterance": per_utt,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="WER of asr.transcribe hypotheses")
    parser.add_argument("--hyp", type=str, default=str(SMOKE_HYP))
    parser.add_argument("--out", type=str, default=str(SMOKE_REPORT))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    self_check()
    report = score(Path(args.hyp).resolve())
    report["purpose"] = ("Caracterización del instrumento sobre voz limpia fuera de los "
                         "sellados. No es un resultado; no hay brazos ni contrastes.")
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for model_name, m in report["models"].items():
        for corpus, w in m["wer_by_corpus"].items():
            logger.info("%-9s %-15s WER %.4f  (n=%d utt, %d palabras; S=%d D=%d I=%d)",
                        model_name, corpus, w["wer"], w["n_utterances"], w["n_words"],
                        w["S"], w["D"], w["I"])
        for k, ch in m["pass_changes"].items():
            logger.info("%-9s %s: cambian %.3f crudo / %.3f normalizado",
                        model_name, k, ch["frac_changed_raw"], ch["frac_changed_normalized"])
    logger.info("Reporte: %s", args.out)


if __name__ == "__main__":
    main()
