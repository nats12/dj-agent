"""Gradio web UI for the DJ Agent mastering workflow.

Launch with::

    python -m dj_agent.gui

Opens at http://localhost:7860
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np

from .master import TEMPLATES, MasterTemplate, format_comparison, master_track

_temp_dirs: list[Path] = []


def _cleanup_temp_dirs() -> None:
    for d in _temp_dirs:
        shutil.rmtree(d, ignore_errors=True)


atexit.register(_cleanup_temp_dirs)


# ---------------------------------------------------------------------------
# Processing wrapper
# ---------------------------------------------------------------------------

def _process_single(
    audio_path: str,
    template_name: str,
) -> tuple[str | None, str | None, str, str]:
    """Process one track and return (original_audio, processed_audio, metrics, status).

    Gradio audio components expect file paths.
    """
    if not audio_path:
        return None, None, "", "No file selected."

    input_path = Path(audio_path)
    output_dir = Path(tempfile.mkdtemp(prefix="dj_agent_"))
    _temp_dirs.append(output_dir)
    output_path = output_dir / f"{input_path.stem}_mastered{input_path.suffix}"

    template_key = template_name.lower().replace(" ", "_")
    if template_key not in TEMPLATES:
        template_key = "official"

    try:
        result = master_track(input_path, output_path, template=template_key)
        metrics = format_comparison(result)
        status = f"Processed with {result['template']} template."
        return str(input_path), str(output_path), metrics, status
    except Exception as e:
        return str(input_path), None, "", f"Error: {e}"


def _process_batch(
    files: list,
    template_name: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str]:
    """Process multiple tracks. Returns (summary_text, detail_text)."""
    if not files:
        return "No files selected.", ""

    # Gradio File components return File objects, not strings
    files = [f.name if hasattr(f, "name") else str(f) for f in files]

    template_key = template_name.lower().replace(" ", "_")
    if template_key not in TEMPLATES:
        template_key = "official"

    output_dir = Path(tempfile.mkdtemp(prefix="dj_agent_batch_"))
    _temp_dirs.append(output_dir)
    results: list[dict[str, Any]] = []
    errors: list[str] = []

    for i, fpath in enumerate(progress.tqdm(files, desc="Mastering")):
        input_path = Path(fpath)
        output_path = output_dir / f"{input_path.stem}_mastered{input_path.suffix}"

        try:
            result = master_track(input_path, output_path, template=template_key)
            results.append(result)
        except Exception as e:
            errors.append(f"{input_path.name}: {e}")

    # Summary
    summary_lines = [
        f"Processed {len(results)}/{len(files)} tracks with {template_name}",
        f"Output directory: {output_dir}",
    ]
    if errors:
        summary_lines.append(f"Errors: {len(errors)}")

    # Detail
    detail_lines: list[str] = []
    for r in results:
        b = r["before"]
        a = r["after"]
        name = Path(r["input_path"]).name
        delta_lufs = a["lufs"] - b["lufs"]
        detail_lines.append(
            f"{name:<40} {b['lufs']:>6.1f} -> {a['lufs']:>6.1f} LUFS ({delta_lufs:>+.1f})"
        )
    for err in errors:
        detail_lines.append(f"ERROR: {err}")

    return "\n".join(summary_lines), "\n".join(detail_lines)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def create_app() -> gr.Blocks:
    """Build the Gradio interface."""

    template_names = [t.name for t in TEMPLATES.values()]

    with gr.Blocks(title="DJ Agent — Master") as app:
        gr.Markdown("# DJ Agent — Audio Mastering")
        gr.Markdown(
            "Platinum Notes-style processing: clip repair, multiband dynamics, "
            "EQ, limiting. **Creates new files — never modifies originals.**"
        )

        with gr.Tabs():
            # ---------------------------------------------------------------
            # Tab 1: Single track (A/B comparison)
            # ---------------------------------------------------------------
            with gr.Tab("Single Track"):
                with gr.Row():
                    with gr.Column(scale=2):
                        input_audio = gr.Audio(
                            label="Input Track",
                            type="filepath",
                        )
                        template_select = gr.Dropdown(
                            choices=template_names,
                            value="Official",
                            label="Template",
                        )
                        process_btn = gr.Button(
                            "Process", variant="primary", size="lg",
                        )

                    with gr.Column(scale=3):
                        status_text = gr.Textbox(
                            label="Status", interactive=False, lines=1,
                        )
                        with gr.Row():
                            original_player = gr.Audio(
                                label="Before (Original)", type="filepath",
                            )
                            processed_player = gr.Audio(
                                label="After (Mastered)", type="filepath",
                            )
                        metrics_display = gr.Textbox(
                            label="Before / After Comparison",
                            interactive=False,
                            lines=10,
                                                    )

                process_btn.click(
                    fn=_process_single,
                    inputs=[input_audio, template_select],
                    outputs=[original_player, processed_player, metrics_display, status_text],
                )

            # ---------------------------------------------------------------
            # Tab 2: Batch processing
            # ---------------------------------------------------------------
            with gr.Tab("Batch"):
                batch_files = gr.File(
                    label="Upload tracks",
                    file_count="multiple",
                    file_types=["audio"],
                )
                batch_template = gr.Dropdown(
                    choices=template_names,
                    value="Official",
                    label="Template",
                )
                batch_btn = gr.Button(
                    "Process All", variant="primary", size="lg",
                )
                batch_summary = gr.Textbox(
                    label="Summary", interactive=False, lines=3,
                )
                batch_detail = gr.Textbox(
                    label="Results", interactive=False, lines=15,
                                    )

                batch_btn.click(
                    fn=_process_batch,
                    inputs=[batch_files, batch_template],
                    outputs=[batch_summary, batch_detail],
                )

            # ---------------------------------------------------------------
            # Tab 3: Key Detection
            # ---------------------------------------------------------------
            with gr.Tab("Key Detection"):
                with gr.Row():
                    with gr.Column(scale=2):
                        key_input = gr.Audio(label="Upload Track", type="filepath")
                        detect_btn = gr.Button("Detect Key", variant="primary")
                    with gr.Column(scale=3):
                        key_result = gr.Textbox(label="Detected Key", interactive=False, lines=4)
                        verify_audio = gr.Audio(label="Verification Chord (play to verify by ear)", type="numpy")

                def _detect_key_handler(audio_path):
                    if not audio_path:
                        return "No file selected.", None
                    try:
                        from .keydetect import detect_key, generate_key_verification_audio
                        result = detect_key(audio_path)
                        chord = generate_key_verification_audio(result.key)
                        text = (
                            f"Key: {result.key}\n"
                            f"Camelot: {result.camelot}\n"
                            f"Confidence: {result.confidence:.0%}\n"
                            f"Method: {result.method}"
                        )
                        return text, (44100, chord)
                    except Exception as e:
                        return f"Error: {e}", None

                detect_btn.click(
                    fn=_detect_key_handler,
                    inputs=[key_input],
                    outputs=[key_result, verify_audio],
                )

            # ---------------------------------------------------------------
            # Tab 4: Stem Export
            # ---------------------------------------------------------------
            with gr.Tab("Stems"):
                gr.Markdown("### Stem Separation — Vocals, Drums, Bass, Other")
                gr.Markdown("*Requires `demucs` (`pip install demucs`). GPU recommended.*")

                stem_input = gr.Audio(label="Upload Track", type="filepath")
                stem_btn = gr.Button("Separate Stems", variant="primary")
                stem_status = gr.Textbox(label="Status", interactive=False, lines=2)

                with gr.Row():
                    stem_vocals = gr.Audio(label="Vocals", type="filepath")
                    stem_drums = gr.Audio(label="Drums", type="filepath")
                with gr.Row():
                    stem_bass = gr.Audio(label="Bass", type="filepath")
                    stem_other = gr.Audio(label="Other", type="filepath")

                def _separate_stems_handler(audio_path):
                    if not audio_path:
                        return "No file selected.", None, None, None, None
                    try:
                        from .stems import export_stems
                        import tempfile
                        output_dir = Path(tempfile.mkdtemp(prefix="dj_stems_"))
                        _temp_dirs.append(output_dir)
                        paths = export_stems(audio_path, output_dir)
                        path_map = {p.stem: str(p) for p in paths}
                        # Handle variant stem names across Demucs models
                        return (
                            f"Separated into {len(paths)} stems at {output_dir}",
                            path_map.get("vocals") or path_map.get("voice"),
                            path_map.get("drums"),
                            path_map.get("bass"),
                            path_map.get("other"),
                        )
                    except ImportError:
                        return "Demucs not installed. Run: pip install demucs", None, None, None, None
                    except Exception as e:
                        return f"Error: {e}", None, None, None, None

                stem_btn.click(
                    fn=_separate_stems_handler,
                    inputs=[stem_input],
                    outputs=[stem_status, stem_vocals, stem_drums, stem_bass, stem_other],
                )

            # ---------------------------------------------------------------
            # Tab 5: Mashup Deck
            # ---------------------------------------------------------------
            with gr.Tab("Mashup Deck"):
                gr.Markdown("### Find Mashup-Compatible Tracks")
                with gr.Row():
                    mashup_a = gr.Audio(label="Track A", type="filepath")
                    mashup_b = gr.Audio(label="Track B", type="filepath")
                with gr.Row():
                    mashup_key_a = gr.Textbox(label="Key A", interactive=False)
                    mashup_bpm_a = gr.Number(label="BPM A", value=128)
                    mashup_key_b = gr.Textbox(label="Key B", interactive=False)
                    mashup_bpm_b = gr.Number(label="BPM B", value=128)
                mashup_score = gr.Textbox(label="Compatibility", interactive=False, lines=5)

                def _mashup_score_handler(path_a, bpm_a, path_b, bpm_b):
                    if not path_a or not path_b:
                        return "", "", ""
                    try:
                        from .keydetect import detect_key
                        from .mashups import score_mashup
                        from .types import TrackInfo

                        key_a = detect_key(path_a)
                        key_b = detect_key(path_b)

                        ta = TrackInfo("a", path_a, "", "", "", float(bpm_a), key_a.camelot, 0)
                        tb = TrackInfo("b", path_b, "", "", "", float(bpm_b), key_b.camelot, 0)
                        ms = score_mashup(ta, tb)

                        score_text = (
                            f"Overall: {ms.total:.0%}\n"
                            f"Harmonic: {ms.harmonic:.0%} (key distance: {ms.key_distance})\n"
                            f"BPM: {ms.bpm:.0%} (diff: {ms.bpm_diff:.1f})\n"
                            f"Vocal complement: {ms.vocal_complement:.0%}\n"
                            f"Energy match: {ms.energy_match:.0%}"
                        )
                        return key_a.camelot, key_b.camelot, score_text
                    except Exception as e:
                        return "", "", f"Error: {e}"

                gr.Button("Score Compatibility", variant="primary").click(
                    fn=_mashup_score_handler,
                    inputs=[mashup_a, mashup_bpm_a, mashup_b, mashup_bpm_b],
                    outputs=[mashup_key_a, mashup_key_b, mashup_score],
                )

            # ---------------------------------------------------------------
            # Tab 6: Pitch Shift
            # ---------------------------------------------------------------
            with gr.Tab("Pitch Shift"):
                gr.Markdown("### Change a Song's Key")
                with gr.Row():
                    ps_input = gr.Audio(label="Input Track", type="filepath")
                    ps_semitones = gr.Slider(label="Semitones", minimum=-6, maximum=6, step=1, value=0)
                ps_btn = gr.Button("Shift Pitch", variant="primary")
                ps_output = gr.Audio(label="Shifted Track", type="filepath")
                ps_status = gr.Textbox(label="Status", interactive=False)

                def _pitch_shift_handler(audio_path, semitones):
                    if not audio_path or semitones == 0:
                        return audio_path, "No shift needed."
                    try:
                        from .pitchshift import pitch_shift
                        result = pitch_shift(audio_path, semitones)
                        return str(result), f"Shifted {semitones:+d} semitones → {result.name}"
                    except Exception as e:
                        return None, f"Error: {e}"

                ps_btn.click(
                    fn=_pitch_shift_handler,
                    inputs=[ps_input, ps_semitones],
                    outputs=[ps_output, ps_status],
                )

            # ---------------------------------------------------------------
            # Tab 7: AI Reasoning (Vibe Analysis)
            # ---------------------------------------------------------------
            with gr.Tab("AI Reasoning"):
                gr.Markdown(
                    "### Deep Vibe Analysis\n"
                    "Powered by local **Ollama** (free, private) or cloud **Gemini** (fast, cheap).\n\n"
                    "Get a free Gemini API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)"
                )

                # API key setup
                with gr.Accordion("Gemini API Setup", open=False):
                    api_key_input = gr.Textbox(
                        label="Gemini API Key",
                        placeholder="Paste your key here (or set GOOGLE_API_KEY env var)",
                        type="password",
                    )
                    api_key_btn = gr.Button("Save Key")
                    api_key_status = gr.Textbox(label="Status", interactive=False, lines=1)

                    def _save_api_key(key):
                        if not key:
                            return "No key provided."
                        try:
                            from .reasoning import setup_gemini
                            if setup_gemini(key.strip()):
                                return "Gemini API key saved and verified."
                            return "Key saved but verification failed. Check the key."
                        except Exception as e:
                            return f"Error: {e}"

                    api_key_btn.click(fn=_save_api_key, inputs=[api_key_input], outputs=[api_key_status])

                # Analysis
                reason_input = gr.Audio(label="Upload Track", type="filepath")
                with gr.Row():
                    reason_backend = gr.Dropdown(
                        choices=["auto", "ollama", "gemini-lite", "gemini-flash", "gemini-pro"],
                        value="auto",
                        label="Backend",
                        info="auto=best available | ollama=local | gemini-lite=cheapest | flash=balanced | pro=best",
                    )
                    reason_action = gr.Dropdown(
                        choices=["Vibe Analysis", "Energy Arc", "Nuance Tags"],
                        value="Vibe Analysis", label="Analysis Type",
                    )
                reason_btn = gr.Button("Analyze", variant="primary")
                reason_output = gr.Textbox(label="AI Analysis", interactive=False, lines=8)

                def _reason_handler(audio_path, backend, action):
                    if not audio_path:
                        return "No file selected."
                    try:
                        from .reasoning import analyze_vibe, get_energy_arc, classify_nuance

                        # Map dropdown values to backend strings
                        backend_map = {
                            "gemini-lite": "gemini-lite",
                            "gemini-flash": "gemini",  # default flash tier
                            "gemini-pro": "gemini-pro",
                        }
                        be = backend_map.get(backend, backend)

                        if action == "Vibe Analysis":
                            return analyze_vibe(audio_path, backend=be)
                        elif action == "Energy Arc":
                            return get_energy_arc(audio_path, backend=be)
                        elif action == "Nuance Tags":
                            import json
                            tags = classify_nuance(audio_path, backend=be)
                            return json.dumps(tags, indent=2)
                        return "Unknown action."
                    except Exception as e:
                        return f"Error: {e}"

                reason_btn.click(
                    fn=_reason_handler,
                    inputs=[reason_input, reason_backend, reason_action],
                    outputs=[reason_output],
                )

            # ---------------------------------------------------------------
            # Tab 8: Template info
            # ---------------------------------------------------------------
            with gr.Tab("Templates"):
                gr.Markdown(_template_info_markdown())

    return app


def _template_info_markdown() -> str:
    """Generate markdown table describing all templates."""
    lines = [
        "## Processing Templates",
        "",
        "| Template | Target LUFS | Bass | Treble | Peak Ceiling | Character |",
        "|----------|-------------|------|--------|-------------|-----------|",
    ]
    descriptions = {
        "Official": "Gentle, preserves dynamics",
        "Festival": "Punchy bass, bright presence",
        "Big Boost": "Maximum loudness, aggressive",
        "Gentle": "Minimal processing, consistency only",
    }
    for name, tmpl in TEMPLATES.items():
        desc = descriptions.get(tmpl.name, "")
        lines.append(
            f"| **{tmpl.name}** | {tmpl.target_lufs} | "
            f"{tmpl.bass_shelf_db:+.0f} dB | {tmpl.treble_shelf_db:+.0f} dB | "
            f"{tmpl.peak_ceiling_db} dBFS | {desc} |"
        )

    lines.extend([
        "",
        "### Processing Chain",
        "",
        "1. **Clip repair** — cubic spline reconstruction of clipped peaks",
        "2. **Multiband compression** — 4 bands (0-200Hz, 200-2kHz, 2k-8kHz, 8kHz+)",
        "3. **Shelving EQ** — bass/treble shaping",
        "4. **LUFS gain** — adjust to target loudness",
        "5. **Brick-wall limiter** — final peak safety",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def launch(**kwargs: Any) -> None:
    """Launch the Gradio UI."""
    app = create_app()
    app.launch(**kwargs)


if __name__ == "__main__":
    launch()
