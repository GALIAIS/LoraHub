from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORM = ROOT / "web" / "src" / "components" / "config-form" / "index.tsx"


def _source() -> str:
    return FORM.read_text(encoding="utf-8")


def test_backend_forms_are_explicit_routes() -> None:
    src = _source()
    assert "SECTION_BACKENDS" not in src
    assert "function KohyaForm" in src
    assert "function DiffusionPipeForm" in src
    assert "function AnimaLoraForm" in src
    assert "function AiToolkitForm" in src
