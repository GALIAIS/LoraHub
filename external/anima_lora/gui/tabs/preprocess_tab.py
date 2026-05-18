"""PreprocessingTab — caption shuffle/dropout + SAM3/MIT mask config.

Layout mirrors ConfigTab:
- Top action bar (refresh + per-step Run buttons + Save + Stop)
- Horizontal split: form on left (clickable labels show help on the right),
  explanation panel on the right
- Log panel below in a vertical splitter

Surfaces the knobs that the bare ``make preprocess`` / ``make mask`` paths
hardcode: caption shuffle variant count, per-tag dropout rate, SAM prompt
list / threshold / dilate, MIT text-threshold / dilate.

Settings persist to:
- ``configs/sam_mask.yaml`` — SAM prompts / threshold / dilate (existing
  canonical location, read directly by ``preprocess/generate_masks.py``).
- ``gui/gui_settings.json`` — TE-cache and MIT knobs, picked up by this
  tab on launch and forwarded to subprocesses via env vars
  (``CAPTION_SHUFFLE_VARIANTS``, ``CAPTION_TAG_DROPOUT_RATE``,
  ``MIT_TEXT_THRESHOLD``, ``MIT_DILATE``) consumed by
  ``scripts/tasks/preprocess.py`` and ``scripts/tasks/masking.py``.
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

import yaml
from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from gui import IMAGE_EXTS, ROOT, count_preprocess_caches
from gui.explanations import preprocess_field_help, preprocess_guide
from gui.i18n import t
from gui.process import kill_process_tree, make_subprocess_env, setup_kill_safe
from gui.progress import TQDM_RE, TqdmProgressTracker, make_progress_bar
from gui.tabs.config_tab import ClickableLabel

SAM_YAML = ROOT / "configs" / "sam_mask.yaml"
SETTINGS_FILE = Path(__file__).resolve().parent.parent / "gui_settings.json"

# Defaults match the historical hardcoded values in scripts/tasks/preprocess.py
# and preprocess/generate_masks_mit.py so a freshly installed GUI runs the
# same pipeline as the bare CLI.
DEFAULT_TE_SHUFFLE_VARIANTS = 4
DEFAULT_TE_TAG_DROPOUT = 0.1
DEFAULT_SAM_PROMPTS = ("speech bubble", "text bubble")
DEFAULT_SAM_THRESHOLD = 0.5
DEFAULT_SAM_DILATE = 5
DEFAULT_MIT_TEXT_THRESHOLD = 0.8
DEFAULT_MIT_DILATE = 5
DEFAULT_RUN_SAM_MASK = True
DEFAULT_RUN_MIT_MASK = True

RESIZED_DIR = ROOT / "post_image_dataset" / "resized"
LORA_CACHE_DIR = ROOT / "post_image_dataset" / "lora"
# Merged masks now live under the cache root alongside the resized tree
# (the SAM/MIT intermediates run through a tempdir during `make mask`).
MASK_DIR = ROOT / "post_image_dataset" / "masks"


def _load_settings() -> dict:
    if not SETTINGS_FILE.exists():
        return {}
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_settings(updates: dict) -> None:
    """Merge ``updates`` into the existing settings JSON, preserving other keys."""
    data = _load_settings()
    data.update(updates)
    SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_sam_yaml() -> dict:
    if not SAM_YAML.exists():
        return {}
    try:
        return yaml.safe_load(SAM_YAML.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


class _IndentedListDumper(yaml.SafeDumper):
    """SafeDumper that indents list items under mapping keys.

    PyYAML's default dumper writes list items flush with the parent key,
    which is valid YAML but doesn't match the canonical sam_mask.yaml
    formatting (2-space indent on the dash). Overriding ``increase_indent``
    to disable ``indentless`` mode gives us the indented form so saving
    from the GUI doesn't churn the file's whitespace.
    """

    def increase_indent(self, flow=False, indentless=False):  # noqa: D401
        return super().increase_indent(flow, False)


def _save_sam_yaml(prompts: list[str], threshold: float, dilate: int) -> None:
    SAM_YAML.parent.mkdir(parents=True, exist_ok=True)
    payload = {"prompts": prompts, "threshold": threshold, "dilate": dilate}
    text = yaml.dump(
        payload,
        Dumper=_IndentedListDumper,
        default_flow_style=False,
        sort_keys=False,
    )
    # Match the canonical layout's blank line between the prompts list and
    # the scalar settings.
    text = text.replace("\nthreshold:", "\n\nthreshold:", 1)
    SAM_YAML.write_text(text, encoding="utf-8")


def _count_masks(mask_dir: Path) -> int:
    if not mask_dir.is_dir():
        return 0
    # rglob picks up the nested `<rel>/` subtrees produced by `make mask`
    # under the consolidated layout; legacy flat trees still count correctly.
    return sum(1 for _ in mask_dir.rglob("*_mask.png"))


def _count_resized() -> int:
    if not RESIZED_DIR.is_dir():
        return 0
    # rglob picks up the nested `<rel>/` subtrees produced by recursive
    # resize_images.py; flat trees still count correctly.
    return sum(
        1
        for p in RESIZED_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


class PreprocessingTab(QWidget):
    def __init__(self):
        super().__init__()
        self._proc = QProcess(self)
        self._proc.setWorkingDirectory(str(ROOT))
        self._proc.setProcessChannelMode(QProcess.MergedChannels)
        setup_kill_safe(self._proc)
        self._proc.readyReadStandardOutput.connect(self._read_stdout)
        self._proc.finished.connect(self._on_finished)
        self._buf = ""
        self._run_buttons: list[QPushButton] = []

        outer = QVBoxLayout(self)

        # ── Top action bar ────────────────────────────────────────
        # Mirrors ConfigTab: Save + per-step Run buttons + Stop, all under
        # the tab strip on a single row. No manual refresh — the status
        # one-liner is rebuilt automatically on every _on_finished.
        top = QHBoxLayout()

        # Color semantics (matches ConfigTab):
        #   Save           → neutral (default styling, no background tint)
        #   Cache / mask   → blue   (#2980b9) — run a specific preprocess step
        #   Stop           → red    (#c0392b) — abort the running subprocess
        run_step_style = (
            "background:#2980b9;color:white;font-weight:bold;padding:4px 16px;"
        )

        self.save_btn = QPushButton(t("preprocess_save_settings"))
        self.save_btn.setToolTip(t("preprocess_save_settings_tip"))
        self.save_btn.clicked.connect(self._save_all_clicked)
        top.addWidget(self.save_btn)

        # Per-step Run buttons. Save is implicit on each Run (same pattern
        # as ConfigTab's auto-save before Train/Preprocess).
        self.run_te_btn = QPushButton(t("preprocess_run_te"))
        self.run_te_btn.setStyleSheet(run_step_style)
        self.run_te_btn.clicked.connect(self._run_te)
        self._run_buttons.append(self.run_te_btn)
        top.addWidget(self.run_te_btn)

        self.run_mask_btn = QPushButton(t("preprocess_run_mask"))
        self.run_mask_btn.setStyleSheet(run_step_style)
        self.run_mask_btn.clicked.connect(self._run_mask)
        self._run_buttons.append(self.run_mask_btn)
        top.addWidget(self.run_mask_btn)

        top.addStretch()
        self.stop_btn = QPushButton(t("stop"))
        self.stop_btn.setStyleSheet(
            "background:#c0392b;color:white;font-weight:bold;padding:4px 16px;"
        )
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        top.addWidget(self.stop_btn)
        outer.addLayout(top)

        # tqdm bar (same look as ConfigTab — shared QSS in gui/progress.py).
        # Shown when a child process emits a parseable tqdm line, hidden again
        # on _on_finished.
        self.progress = make_progress_bar()
        self._progress_tracker = TqdmProgressTracker(self.progress)
        outer.addWidget(self.progress)

        # Status one-liner stays directly under the progress bar.
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("color:#dcdcdc; padding: 2px 0;")
        outer.addWidget(self.status_lbl)

        # ── Body: vertical splitter (form+explain top, log bottom) ──
        vsplit = QSplitter(Qt.Vertical)

        # Horizontal splitter: form on left, explanation panel on right.
        hsplit = QSplitter(Qt.Horizontal)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_host = QWidget()
        form_layout = QVBoxLayout(form_host)
        form_layout.setContentsMargins(0, 0, 0, 0)

        settings = _load_settings()
        sam_yaml = _load_sam_yaml()
        sam_prompts = sam_yaml.get("prompts") or list(DEFAULT_SAM_PROMPTS)
        sam_threshold = float(sam_yaml.get("threshold", DEFAULT_SAM_THRESHOLD))
        sam_dilate = int(sam_yaml.get("dilate", DEFAULT_SAM_DILATE))

        # Text caching group
        text_box = QGroupBox(t("preprocess_text_caching"))
        text_form = QFormLayout()
        self.shuffle_spin = QSpinBox()
        self.shuffle_spin.setRange(0, 64)
        self.shuffle_spin.setValue(
            int(settings.get("caption_shuffle_variants", DEFAULT_TE_SHUFFLE_VARIANTS))
        )
        # Block scroll-wheel changes (matches gui/__init__.py::_widget convention).
        self.shuffle_spin.wheelEvent = lambda e: e.ignore()
        text_form.addRow(
            self._field_label(
                "caption_shuffle_variants",
                t("preprocess_caption_shuffle_variants"),
            ),
            self.shuffle_spin,
        )

        self.dropout_edit = QLineEdit(
            f"{float(settings.get('caption_tag_dropout_rate', DEFAULT_TE_TAG_DROPOUT)):g}"
        )
        text_form.addRow(
            self._field_label(
                "caption_tag_dropout_rate",
                t("preprocess_caption_tag_dropout_rate"),
            ),
            self.dropout_edit,
        )
        text_box.setLayout(text_form)
        form_layout.addWidget(text_box)

        # SAM masking group
        sam_box = QGroupBox(t("preprocess_masking_sam"))
        sam_form = QFormLayout()
        self.run_sam_mask_chk = QCheckBox(t("preprocess_run_sam_mask"))
        self.run_sam_mask_chk.setToolTip(t("preprocess_run_sam_mask_tip"))
        self.run_sam_mask_chk.setChecked(
            bool(settings.get("run_sam_mask", DEFAULT_RUN_SAM_MASK))
        )
        sam_form.addRow(
            self._field_label("run_sam_mask", t("preprocess_run_sam_mask")),
            self.run_sam_mask_chk,
        )

        self.sam_prompts_edit = QPlainTextEdit("\n".join(sam_prompts))
        self.sam_prompts_edit.setMaximumHeight(80)
        self.sam_prompts_edit.setStyleSheet("font-family:monospace;")
        sam_form.addRow(
            self._field_label("sam_prompts", t("preprocess_sam_prompts")),
            self.sam_prompts_edit,
        )

        self.sam_threshold_edit = QLineEdit(f"{sam_threshold:g}")
        sam_form.addRow(
            self._field_label("sam_threshold", t("preprocess_sam_threshold")),
            self.sam_threshold_edit,
        )

        self.sam_dilate_spin = QSpinBox()
        self.sam_dilate_spin.setRange(0, 64)
        self.sam_dilate_spin.setValue(sam_dilate)
        self.sam_dilate_spin.wheelEvent = lambda e: e.ignore()
        sam_form.addRow(
            self._field_label("sam_dilate", t("preprocess_dilate")),
            self.sam_dilate_spin,
        )
        sam_box.setLayout(sam_form)
        form_layout.addWidget(sam_box)

        # MIT masking group
        mit_box = QGroupBox(t("preprocess_masking_mit"))
        mit_form = QFormLayout()
        self.run_mit_mask_chk = QCheckBox(t("preprocess_run_mit_mask"))
        self.run_mit_mask_chk.setToolTip(t("preprocess_run_mit_mask_tip"))
        self.run_mit_mask_chk.setChecked(
            bool(settings.get("run_mit_mask", DEFAULT_RUN_MIT_MASK))
        )
        mit_form.addRow(
            self._field_label("run_mit_mask", t("preprocess_run_mit_mask")),
            self.run_mit_mask_chk,
        )

        self.mit_threshold_edit = QLineEdit(
            f"{float(settings.get('mit_text_threshold', DEFAULT_MIT_TEXT_THRESHOLD)):g}"
        )
        mit_form.addRow(
            self._field_label("mit_text_threshold", t("preprocess_mit_threshold")),
            self.mit_threshold_edit,
        )

        self.mit_dilate_spin = QSpinBox()
        self.mit_dilate_spin.setRange(0, 64)
        self.mit_dilate_spin.setValue(
            int(settings.get("mit_dilate", DEFAULT_MIT_DILATE))
        )
        self.mit_dilate_spin.wheelEvent = lambda e: e.ignore()
        mit_form.addRow(
            self._field_label("mit_dilate", t("preprocess_dilate")),
            self.mit_dilate_spin,
        )
        mit_box.setLayout(mit_form)
        form_layout.addWidget(mit_box)

        form_layout.addStretch()
        scroll.setWidget(form_host)
        hsplit.addWidget(scroll)

        # Right panel — same QTextBrowser style as ConfigTab's explain panel
        # so the look matches across tabs.
        self._explain = QTextBrowser()
        self._explain.setOpenExternalLinks(True)
        self._explain.setStyleSheet(
            "QTextBrowser { font-size: 13px; padding: 12px; "
            "background: #2b2b2b; color: #e0e0e0; }"
        )
        self._explain.setMinimumWidth(320)
        self._show_default_explain()
        hsplit.addWidget(self._explain)
        hsplit.setStretchFactor(0, 3)
        hsplit.setStretchFactor(1, 2)
        hsplit.setSizes([720, 420])
        vsplit.addWidget(hsplit)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("font-family:monospace;font-size:11px;")
        self.log.setPlaceholderText(t("preprocess_log_placeholder"))
        vsplit.addWidget(self.log)
        vsplit.setSizes([520, 200])
        outer.addWidget(vsplit, 1)

        self._refresh_status()

    # ── Field labels & explain panel ───────────────────────────────

    def _field_label(self, key: str, text_str: str) -> ClickableLabel:
        """Build a ClickableLabel that shows this field's help when clicked."""
        lbl = ClickableLabel(text_str)
        lbl.setStyleSheet("color:#f0f0f0; text-decoration: underline dotted;")
        help_text = preprocess_field_help(key)
        lbl.clicked.connect(
            lambda _k=key, _h=help_text, _t=text_str: self._show_field_help(_t, _h)
        )
        return lbl

    def _show_default_explain(self) -> None:
        self._explain.setHtml(preprocess_guide())

    def _show_field_help(self, field_label: str, help_text: str | None) -> None:
        parts = [
            f"<h2 style='margin:0 0 10px 0; font-size:18px;'>"
            f"{html.escape(field_label)}</h2>"
        ]
        if help_text:
            parts.append(
                f"<p style='font-size:14px; line-height:1.6;'>"
                f"{html.escape(help_text)}</p>"
            )
        else:
            parts.append(
                f"<p style='color:#888; font-style:italic;'>"
                f"{html.escape(t('no_help_available'))}</p>"
            )
        self._explain.setHtml("".join(parts))

    # ── Status panel ───────────────────────────────────────────────

    def _refresh_status(self) -> None:
        n_resized = _count_resized()
        caches = count_preprocess_caches(LORA_CACHE_DIR)
        mask_n = _count_masks(MASK_DIR)
        if n_resized == 0:
            self.status_lbl.setText(t("preprocess_status_no_resized"))
            return
        lines = [
            t("preprocess_status_resized", n=n_resized),
            t(
                "preprocess_status_caches",
                lat=caches["latents"],
                te=caches["te"],
                pe=caches["pe"],
            ),
            t("preprocess_status_masks", masks=mask_n),
        ]
        self.status_lbl.setText("  |  ".join(lines))

    # ── Settings persistence ───────────────────────────────────────

    def _parse_float(self, text: str, field_label: str) -> float | None:
        try:
            return float(text)
        except ValueError:
            QMessageBox.warning(
                self,
                t("error"),
                t("preprocess_invalid_float", field=field_label, value=text),
            )
            return None

    def _save_all(self) -> bool:
        """Validate and persist every form value. Returns True on success."""
        dropout = self._parse_float(
            self.dropout_edit.text().strip(),
            t("preprocess_caption_tag_dropout_rate"),
        )
        if dropout is None:
            return False
        sam_threshold = self._parse_float(
            self.sam_threshold_edit.text().strip(),
            t("preprocess_sam_threshold"),
        )
        if sam_threshold is None:
            return False
        mit_threshold = self._parse_float(
            self.mit_threshold_edit.text().strip(),
            t("preprocess_mit_threshold"),
        )
        if mit_threshold is None:
            return False

        prompts = [
            line.strip()
            for line in self.sam_prompts_edit.toPlainText().splitlines()
            if line.strip()
        ]
        if not prompts:
            prompts = list(DEFAULT_SAM_PROMPTS)

        _save_sam_yaml(prompts, sam_threshold, int(self.sam_dilate_spin.value()))
        _save_settings(
            {
                "caption_shuffle_variants": int(self.shuffle_spin.value()),
                "caption_tag_dropout_rate": dropout,
                "mit_text_threshold": mit_threshold,
                "mit_dilate": int(self.mit_dilate_spin.value()),
                "run_sam_mask": self.run_sam_mask_chk.isChecked(),
                "run_mit_mask": self.run_mit_mask_chk.isChecked(),
            }
        )
        return True

    def _save_all_clicked(self) -> None:
        if self._save_all():
            QMessageBox.information(self, t("saved"), t("preprocess_settings_saved"))

    # ── Subprocess actions ─────────────────────────────────────────

    def _is_running(self) -> bool:
        return self._proc.state() != QProcess.NotRunning

    def _run_te(self) -> None:
        # Unified "caching" step — runs `tasks.py preprocess`, which chains
        # resize → VAE-latent cache → text-embedding cache. Replaces the old
        # text-only path now that the ConfigTab's standalone Preprocess
        # button is gone and this tab owns the cache-build UI. The TE knobs
        # (shuffle / dropout) are still surfaced as env vars; resize and VAE
        # currently have no GUI-tunable parameters, so the form stays TE-only.
        if not self._save_all():
            return
        env = make_subprocess_env(
            CAPTION_SHUFFLE_VARIANTS=str(int(self.shuffle_spin.value())),
            CAPTION_TAG_DROPOUT_RATE=self.dropout_edit.text().strip(),
        )
        self._launch(["tasks.py", "preprocess"], env=env)

    def _run_mask(self) -> None:
        # Single-shot pipeline. ``tasks.py mask`` runs SAM and/or MIT into
        # a tempdir, merges the produced sources, and writes only the
        # merged result to ``post_image_dataset/masks/<rel>/``. SAM reads
        # ``configs/sam_mask.yaml`` directly; MIT picks up the
        # ``MIT_TEXT_THRESHOLD`` / ``MIT_DILATE`` env vars set below.
        # ``RUN_SAM_MASK`` / ``RUN_MIT_MASK`` gate each backend.
        if not self._save_all():
            return
        run_sam = self.run_sam_mask_chk.isChecked()
        run_mit = self.run_mit_mask_chk.isChecked()
        if not (run_sam or run_mit):
            QMessageBox.warning(
                self, t("error"), t("preprocess_mask_nothing_enabled")
            )
            return
        env = make_subprocess_env(
            MIT_TEXT_THRESHOLD=self.mit_threshold_edit.text().strip(),
            MIT_DILATE=str(int(self.mit_dilate_spin.value())),
            RUN_SAM_MASK="1" if run_sam else "0",
            RUN_MIT_MASK="1" if run_mit else "0",
        )
        self._launch(["tasks.py", "mask"], env=env)

    def _launch(self, argv: list[str], env=None) -> None:
        if self._is_running():
            QMessageBox.information(self, "", t("preprocess_already_running"))
            return
        if env is not None:
            self._proc.setProcessEnvironment(env)
        self.log.clear()
        self._buf = ""
        self._progress_tracker.reset()
        self.log.appendPlainText("> " + " ".join([sys.executable, *argv]))
        self._proc.start(sys.executable, argv)
        for btn in self._run_buttons:
            btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _stop(self) -> None:
        kill_process_tree(self._proc)

    def cleanup_subprocess(self) -> None:
        """Hook for app shutdown — kill any running preprocess subtree."""
        kill_process_tree(self._proc)

    def _read_stdout(self) -> None:
        # Split on \n and \r so tqdm's carriage-return updates feed the bar
        # without spamming the log. Mirrors ConfigTab._handle_stream.
        data = bytes(self._proc.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        self._buf += data
        parts = re.split(r"[\r\n]", self._buf)
        self._buf = parts[-1]  # incomplete trailing fragment stays buffered
        for line in parts[:-1]:
            if self._progress_tracker.feed(line):
                continue
            if line:
                self.log.appendPlainText(line)

    def _on_finished(self, code: int, _status) -> None:
        # Flush any leftover non-tqdm tail before the finish banner. A
        # trailing tqdm fragment is dropped — the bar already reflected its
        # state and a half-written progress line in the log is just noise.
        if self._buf and not TQDM_RE.search(self._buf):
            self.log.appendPlainText(self._buf)
        self._buf = ""
        self.log.appendPlainText(t("finished", code=code))
        self._progress_tracker.reset()
        for btn in self._run_buttons:
            btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._refresh_status()
