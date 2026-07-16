import csv
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, PngImagePlugin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extract_prompts import (
    extract_final_positive_prompt_from_png,
    extract_from_api_prompt,
    extract_from_parameters,
    extract_from_workflow,
    write_csv,
    write_txt,
)

MODULE_PATH = Path(__file__).resolve().parent.parent / "extract_prompts.py"


def _make_png(path, chunks=None):
    img = Image.new("RGB", (1, 1))
    info = PngImagePlugin.PngInfo()
    for key, value in (chunks or {}).items():
        info.add_text(key, value)
    img.save(path, pnginfo=info)


# A minimal workflow graph: CLIPTextEncode(3) -> KSampler(4) -> VAEDecode(5) -> SaveImage(6)
# Node 7 is a CLIPTextEncode for the negative prompt.
# Link record format: [link_id, src_id, src_slot, dst_id, dst_slot, type]
SAMPLE_WORKFLOW = {
    "nodes": [
        {
            "id": 3,
            "type": "CLIPTextEncode",
            "order": 0,
            "inputs": [{"name": "clip", "type": "CLIP", "link": None},
                       {"name": "text", "type": "STRING", "link": None}],
            "widgets_values": ["a beautiful sunset over mountains"],
        },
        {
            "id": 7,
            "type": "CLIPTextEncode",
            "order": 1,
            "inputs": [{"name": "clip", "type": "CLIP", "link": None},
                       {"name": "text", "type": "STRING", "link": None}],
            "widgets_values": ["ugly, blurry"],
        },
        {
            "id": 4,
            "type": "KSampler",
            "order": 2,
            "inputs": [
                {"name": "model", "type": "MODEL", "link": None},
                {"name": "positive", "type": "CONDITIONING", "link": 1},
                {"name": "negative", "type": "CONDITIONING", "link": 2},
                {"name": "latent_image", "type": "LATENT", "link": None},
            ],
        },
        {
            "id": 5,
            "type": "VAEDecode",
            "order": 3,
            "inputs": [{"name": "samples", "type": "LATENT", "link": 3},
                       {"name": "vae", "type": "VAE", "link": None}],
        },
        {
            "id": 6,
            "type": "SaveImage",
            "order": 4,
            "inputs": [{"name": "images", "type": "IMAGE", "link": 4}],
        },
    ],
    "links": [
        [1, 3, 0, 4, 1, "CONDITIONING"],
        [2, 7, 0, 4, 2, "CONDITIONING"],
        [3, 4, 0, 5, 0, "LATENT"],
        [4, 5, 0, 6, 0, "IMAGE"],
    ],
}

# Same pipeline in flattened API format (as executed / after subgraph expansion).
SAMPLE_API_PROMPT = {
    "3": {"class_type": "CLIPTextEncode",
          "inputs": {"clip": ["1", 1], "text": "a beautiful sunset over mountains"}},
    "7": {"class_type": "CLIPTextEncode",
          "inputs": {"clip": ["1", 1], "text": "ugly, blurry"}},
    "4": {"class_type": "KSampler",
          "inputs": {"model": ["1", 0], "positive": ["3", 0], "negative": ["7", 0],
                     "latent_image": ["2", 0]}},
    "5": {"class_type": "VAEDecode", "inputs": {"samples": ["4", 0], "vae": ["1", 2]}},
    "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0]}},
}

SAMPLE_PARAMETERS = (
    "a beautiful sunset over mountains\n"
    "Negative prompt: ugly, blurry\n"
    "Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1"
)


def test_workflow_extraction():
    prompt, err = extract_from_workflow(SAMPLE_WORKFLOW)
    assert err is None
    assert prompt == "a beautiful sunset over mountains"


def test_workflow_null_links_do_not_crash():
    # Regression: modern ComfyUI serializes unconnected inputs as "link": null.
    prompt, err = extract_from_workflow(SAMPLE_WORKFLOW)
    assert err is None


def test_workflow_link_id_zero():
    # Regression: link id 0 must not be treated as "no link".
    wf = json.loads(json.dumps(SAMPLE_WORKFLOW))
    remap = {1: 0}
    for rec in wf["links"]:
        rec[0] = remap.get(rec[0], rec[0])
    for node in wf["nodes"]:
        for inp in node["inputs"]:
            if inp.get("link") in remap:
                inp["link"] = remap[inp["link"]]
    prompt, err = extract_from_workflow(wf)
    assert err is None
    assert prompt == "a beautiful sunset over mountains"


def test_workflow_dangling_link_id_no_crash():
    wf = json.loads(json.dumps(SAMPLE_WORKFLOW))
    wf["links"] = [rec for rec in wf["links"] if rec[0] != 4]  # SaveImage link dangles
    prompt, err = extract_from_workflow(wf)
    assert prompt is None
    assert err == "no-start-from-saveimage"


def test_api_prompt_extraction():
    prompt, err = extract_from_api_prompt(SAMPLE_API_PROMPT)
    assert err is None
    assert prompt == "a beautiful sunset over mountains"


def test_api_prompt_preview_only_workflow():
    # No SaveImage at all: fall back to PreviewImage as the terminal node.
    graph = json.loads(json.dumps(SAMPLE_API_PROMPT))
    graph["6"] = {"class_type": "PreviewImage", "inputs": {"images": ["5", 0]}}
    prompt, err = extract_from_api_prompt(graph)
    assert err is None
    assert prompt == "a beautiful sunset over mountains"


def test_api_prompt_linked_text():
    graph = json.loads(json.dumps(SAMPLE_API_PROMPT))
    graph["9"] = {"class_type": "ShowText|pysssss",
                  "inputs": {"text": "prompt from a text node"}}
    graph["3"]["inputs"]["text"] = ["9", 0]
    prompt, err = extract_from_api_prompt(graph)
    assert err is None
    assert prompt == "prompt from a text node"


def test_parameters_extraction():
    prompt, err = extract_from_parameters(SAMPLE_PARAMETERS)
    assert err is None
    assert prompt == "a beautiful sunset over mountains"


def test_parameters_no_negative():
    prompt, err = extract_from_parameters(
        "just a prompt\nSteps: 20, Sampler: Euler a, CFG scale: 7")
    assert err is None
    assert prompt == "just a prompt"


def test_png_fallback_workflow_to_api(tmp_path):
    # workflow chunk is unusable (subgraph-style: no top-level SaveImage),
    # so extraction must fall back to the API prompt chunk.
    broken_workflow = {"nodes": [], "links": [], "definitions": {"subgraphs": []}}
    png = tmp_path / "subgraph.png"
    _make_png(png, {
        "workflow": json.dumps(broken_workflow),
        "prompt": json.dumps(SAMPLE_API_PROMPT),
    })
    prompt, err = extract_final_positive_prompt_from_png(str(png))
    assert err is None
    assert prompt == "a beautiful sunset over mountains"


def test_png_a1111_parameters(tmp_path):
    png = tmp_path / "a1111.png"
    _make_png(png, {"parameters": SAMPLE_PARAMETERS})
    prompt, err = extract_final_positive_prompt_from_png(str(png))
    assert err is None
    assert prompt == "a beautiful sunset over mountains"


def test_png_no_metadata(tmp_path):
    png = tmp_path / "plain.png"
    _make_png(png)
    prompt, err = extract_final_positive_prompt_from_png(str(png))
    assert prompt is None
    assert err == "no-metadata"


def test_write_csv_sanitizes_formula_injection(tmp_path):
    out = tmp_path / "out.csv"
    write_csv(str(out), [("=cmd|' /C calc'!A0.png", "+SUM(1,1) prompt")])
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[1][0] == "'=cmd|' /C calc'!A0.png"
    assert rows[1][1] == "'+SUM(1,1) prompt"


def test_write_txt_flattens_newlines(tmp_path):
    out = tmp_path / "out.txt"
    write_txt(str(out), [("a.png", "line one\nline two\r\nline three")])
    assert out.read_text(encoding="utf-8") == "line one line two line three\n"


def test_cli_end_to_end(tmp_path):
    png_dir = tmp_path / "images"
    png_dir.mkdir()
    _make_png(png_dir / "a.png", {"workflow": json.dumps(SAMPLE_WORKFLOW)})
    _make_png(png_dir / "b.png", {"prompt": json.dumps(SAMPLE_API_PROMPT)})
    _make_png(png_dir / "c.png")  # no metadata -> skipped

    txt_out = tmp_path / "result.txt"
    proc = subprocess.run(
        [sys.executable, str(MODULE_PATH), str(png_dir), str(txt_out)],
        check=True, capture_output=True, text=True,
    )
    assert "Extracted 2 prompts (1 skipped)" in proc.stdout
    lines = txt_out.read_text(encoding="utf-8").splitlines()
    assert lines == ["a beautiful sunset over mountains"] * 2

    csv_out = tmp_path / "result.csv"
    subprocess.run(
        [sys.executable, str(MODULE_PATH), str(png_dir), str(csv_out), "--csv"],
        check=True, capture_output=True, text=True,
    )
    with open(csv_out, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["filename", "prompt"]
    assert [r[0] for r in rows[1:]] == ["a.png", "b.png"]


def test_cli_recursive(tmp_path):
    png_dir = tmp_path / "images"
    (png_dir / "sub").mkdir(parents=True)
    _make_png(png_dir / "sub" / "nested.png", {"workflow": json.dumps(SAMPLE_WORKFLOW)})

    csv_out = tmp_path / "result.csv"
    subprocess.run(
        [sys.executable, str(MODULE_PATH), str(png_dir), str(csv_out), "--csv", "--recursive"],
        check=True, capture_output=True, text=True,
    )
    with open(csv_out, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2
    assert rows[1][0].replace("\\", "/") == "sub/nested.png"


def test_cli_bad_folder():
    proc = subprocess.run(
        [sys.executable, str(MODULE_PATH), "does_not_exist_xyz"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1
    assert "not a folder" in proc.stdout
