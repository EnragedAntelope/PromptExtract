#!/usr/bin/env python3
# extract_prompts.py
#
# Extract the final positive prompt from ComfyUI (and A1111-style) PNG images.
#
# Usage:
#   python extract_prompts.py
#   python extract_prompts.py --csv
#   python extract_prompts.py "C:\path\to\images"
#   python extract_prompts.py "C:\path\to\images" "C:\out\my_prompts.csv" --csv
#   python extract_prompts.py "C:\path\to\images" --recursive

import argparse
import csv
import json
import os
import sys
from collections import deque

from PIL import Image

SAVE_NODE_MARKERS = ("SaveImage", "Save Image", "Image Save")
# Used only when no save node exists (e.g. a preview-only workflow was embedded).
FALLBACK_NODE_MARKERS = ("PreviewImage",)


def _matching_nodes(candidates, get_type):
    for markers in (SAVE_NODE_MARKERS, FALLBACK_NODE_MARKERS):
        found = [c for c in candidates
                 if any(marker in get_type(c) for marker in markers)]
        if found:
            return found
    return []


# ---------------------------------------------------------------------------
# Method 1: workflow graph traversal (the "workflow" PNG chunk)
# ---------------------------------------------------------------------------

def build_link_map(workflow):
    link_map = {}
    for rec in workflow.get("links", []):
        if len(rec) >= 6:
            link_id, src_id, src_slot, dst_id, dst_slot, ltype = rec[:6]
            link_map[link_id] = (src_id, src_slot, dst_id, dst_slot, ltype)
    return link_map


def get_node_by_id(workflow, node_id):
    for n in workflow.get("nodes", []):
        if n.get("id") == node_id:
            return n
    return None


def find_highest_order_saveimage(workflow):
    nodes = workflow.get("nodes", [])
    cands = _matching_nodes(nodes, lambda n: n.get("type", ""))
    if not cands:
        return None
    return max(cands, key=lambda n: n.get("order", 0))


def find_input_link_id(node, name):
    for inp in node.get("inputs", []):
        if inp.get("name") == name and inp.get("link") is not None:
            return inp["link"]
    return None


def _link_source(link_map, link_id):
    entry = link_map.get(link_id)
    return entry[0] if entry else None


def bfs_upstream_to_positive_source(workflow, start_node):
    """
    Starting at the node that feeds SaveImage (e.g., VAEDecode, FaceDetailer),
    walk upstream until we find a node with a 'positive' input. Return the node
    that *produces* that 'positive' (its source node).
    """
    link_map = build_link_map(workflow)
    q = deque([start_node])
    seen = set()
    while q:
        node = q.popleft()
        if node["id"] in seen:
            continue
        seen.add(node["id"])

        for inp in node.get("inputs", []):
            if inp.get("name") == "positive" and inp.get("link") is not None:
                src_id = _link_source(link_map, inp["link"])
                if src_id is not None:
                    return get_node_by_id(workflow, src_id)

        # keep walking upstream through all inputs
        for inp in node.get("inputs", []):
            if inp.get("link") is not None:
                src_id = _link_source(link_map, inp["link"])
                pred = get_node_by_id(workflow, src_id) if src_id is not None else None
                if pred:
                    q.append(pred)
    return None


def find_upstream_clip_encode(workflow, start_node):
    """
    From a CONDITIONING-producing node, walk upstream until a CLIPTextEncode* node is found.
    """
    link_map = build_link_map(workflow)
    q = deque([start_node])
    seen = set()
    while q:
        node = q.popleft()
        if node["id"] in seen:
            continue
        seen.add(node["id"])

        if "CLIPTextEncode" in node.get("type", ""):
            return node

        for inp in node.get("inputs", []):
            if inp.get("link") is not None and inp.get("type") in ("CONDITIONING", "CLIP", "STRING", "any"):
                src_id = _link_source(link_map, inp["link"])
                pred = get_node_by_id(workflow, src_id) if src_id is not None else None
                if pred:
                    q.append(pred)
    return None


def flatten_strings(x):
    if isinstance(x, str):
        return [x]
    if isinstance(x, list):
        out = []
        for e in x:
            out.extend(flatten_strings(e))
        return out
    return []


def extract_string_value_recursive(workflow, node):
    """
    Resolve the actual prompt text. Priority:
      1) If CLIP node has no 'text' link, take its widget value (usually widgets_values[0]).
      2) Else follow its 'text' input link back. For generic nodes, try any STRING-typed input.
      3) When landing on nodes with widgets_values containing strings, pick the longest.
    """
    link_map = build_link_map(workflow)
    visited = set()

    def helper(n):
        if n["id"] in visited:
            return None
        visited.add(n["id"])

        # Case: CLIP node with inline text
        if "CLIPTextEncode" in n.get("type", ""):
            # If 'text' is not linked, prefer its own widget value
            text_input = next((i for i in n.get("inputs", []) if i.get("name") == "text"), None)
            linked = bool(text_input and text_input.get("link") is not None)
            wv = n.get("widgets_values", [])
            if not linked and wv and isinstance(wv[0], str):
                return wv[0].strip()

        # Generic string from widgets_values (ShowText etc.)
        wv = n.get("widgets_values", None)
        if wv:
            strs = [s for s in flatten_strings(wv) if isinstance(s, str) and s.strip()]
            if strs:
                # assume the real prompt is the longest string present
                return max(strs, key=len).strip()

        # Follow 'text' input first if present
        for inp in n.get("inputs", []):
            if inp.get("name") == "text" and inp.get("link") is not None:
                src_id = _link_source(link_map, inp["link"])
                pred = get_node_by_id(workflow, src_id) if src_id is not None else None
                if pred:
                    res = helper(pred)
                    if res:
                        return res

        # Otherwise follow any STRING input
        for inp in n.get("inputs", []):
            if inp.get("type") == "STRING" and inp.get("link") is not None:
                src_id = _link_source(link_map, inp["link"])
                pred = get_node_by_id(workflow, src_id) if src_id is not None else None
                if pred:
                    res = helper(pred)
                    if res:
                        return res
        return None

    return helper(node)


def extract_from_workflow(workflow):
    save = find_highest_order_saveimage(workflow)
    if not save:
        return None, "no-saveimage"

    images_link = find_input_link_id(save, "images")
    if images_link is None:
        return None, "no-saveimage-images-link"

    link_map = build_link_map(workflow)
    src_id = _link_source(link_map, images_link)
    start_node = get_node_by_id(workflow, src_id) if src_id is not None else None
    if not start_node:
        return None, "no-start-from-saveimage"

    pos_src = bfs_upstream_to_positive_source(workflow, start_node)
    if not pos_src:
        return None, "no-positive-found"

    clip_node = find_upstream_clip_encode(workflow, pos_src)
    if not clip_node:
        return None, "no-clip-encode-upstream"

    prompt = extract_string_value_recursive(workflow, clip_node)
    if not prompt:
        return None, "no-prompt-resolved"

    return prompt, None


# ---------------------------------------------------------------------------
# Method 2: API prompt traversal (the "prompt" PNG chunk)
#
# The API format is the flattened graph ComfyUI actually executed: subgraphs,
# reroutes, and muted nodes are already resolved, which makes it the reliable
# source for workflows the visual-graph walk above can't handle (e.g. the
# subgraph feature added to ComfyUI in 2025).
# ---------------------------------------------------------------------------

def _api_link(value):
    # In API format a linked input is [source_node_id, output_slot]
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int)):
        return str(value[0])
    return None


def _api_resolve_text(prompt_graph, node_id, visited=None):
    if visited is None:
        visited = set()
    if node_id in visited:
        return None
    visited.add(node_id)

    node = prompt_graph.get(node_id)
    if not isinstance(node, dict):
        return None
    inputs = node.get("inputs", {})
    if not isinstance(inputs, dict):
        return None

    text = inputs.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    src = _api_link(text)
    if src:
        res = _api_resolve_text(prompt_graph, src, visited)
        if res:
            return res

    # Generic fallback: longest inline string among this node's inputs
    strs = [v for v in inputs.values() if isinstance(v, str) and v.strip()]
    if strs:
        return max(strs, key=len).strip()

    # Follow any remaining links
    for value in inputs.values():
        src = _api_link(value)
        if src:
            res = _api_resolve_text(prompt_graph, src, visited)
            if res:
                return res
    return None


def _api_find_positive_source(prompt_graph, start_id):
    q = deque([start_id])
    seen = set()
    while q:
        node_id = q.popleft()
        if node_id in seen:
            continue
        seen.add(node_id)

        node = prompt_graph.get(node_id)
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue

        src = _api_link(inputs.get("positive"))
        if src:
            return src

        for value in inputs.values():
            link = _api_link(value)
            if link:
                q.append(link)
    return None


def extract_from_api_prompt(prompt_graph):
    candidates = [(nid, node) for nid, node in prompt_graph.items()
                  if isinstance(node, dict)]
    save_ids = [nid for nid, _ in
                _matching_nodes(candidates, lambda c: c[1].get("class_type", ""))]
    if not save_ids:
        return None, "no-saveimage"

    # No execution-order field in API format; try highest node id first.
    def _sort_key(nid):
        try:
            return (1, int(nid))
        except ValueError:
            return (0, 0)

    for save_id in sorted(save_ids, key=_sort_key, reverse=True):
        pos_id = _api_find_positive_source(prompt_graph, save_id)
        if not pos_id:
            continue
        # Prefer a CLIPTextEncode upstream of the positive source; otherwise
        # resolve text from the positive source itself.
        clip_id = _api_find_clip_encode(prompt_graph, pos_id)
        text = _api_resolve_text(prompt_graph, clip_id or pos_id)
        if text:
            return text, None
    return None, "no-prompt-resolved"


def _api_find_clip_encode(prompt_graph, start_id):
    q = deque([start_id])
    seen = set()
    while q:
        node_id = q.popleft()
        if node_id in seen:
            continue
        seen.add(node_id)

        node = prompt_graph.get(node_id)
        if not isinstance(node, dict):
            continue
        if "CLIPTextEncode" in node.get("class_type", ""):
            return node_id

        inputs = node.get("inputs", {})
        if isinstance(inputs, dict):
            for value in inputs.values():
                link = _api_link(value)
                if link:
                    q.append(link)
    return None


# ---------------------------------------------------------------------------
# Method 3: A1111-style "parameters" text chunk
# ---------------------------------------------------------------------------

def extract_from_parameters(parameters):
    """
    A1111/Forge/SD.Next store plain text: positive prompt, then an optional
    'Negative prompt:' line, then a settings line ('Steps: ..., Sampler: ...').
    """
    text = parameters.strip()
    if not text:
        return None, "empty-parameters"

    lines = text.split("\n")
    positive_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Negative prompt:"):
            break
        if stripped.startswith("Steps:") and "Sampler:" in stripped:
            break
        positive_lines.append(line)

    positive = "\n".join(positive_lines).strip()
    if not positive:
        return None, "no-positive-in-parameters"
    return positive, None


# ---------------------------------------------------------------------------
# Per-file extraction: try each metadata source in order
# ---------------------------------------------------------------------------

def extract_final_positive_prompt_from_png(png_path):
    try:
        with Image.open(png_path) as img:
            meta = img.info
    except Exception as e:
        return None, f"error:{type(e).__name__}: {e}"

    errors = []

    wf_raw = meta.get("workflow")
    if wf_raw:
        try:
            workflow = json.loads(wf_raw)
            if isinstance(workflow, dict):
                prompt, err = extract_from_workflow(workflow)
                if prompt:
                    return prompt, None
                errors.append(f"workflow:{err}")
            else:
                errors.append("workflow:bad-json:not-an-object")
        except json.JSONDecodeError:
            errors.append("workflow:bad-json")
        except Exception as e:
            errors.append(f"workflow:error:{type(e).__name__}: {e}")

    api_raw = meta.get("prompt")
    if api_raw:
        try:
            prompt_graph = json.loads(api_raw)
            if isinstance(prompt_graph, dict):
                prompt, err = extract_from_api_prompt(prompt_graph)
                if prompt:
                    return prompt, None
                errors.append(f"prompt:{err}")
            else:
                errors.append("prompt:bad-json:not-an-object")
        except json.JSONDecodeError:
            errors.append("prompt:bad-json")
        except Exception as e:
            errors.append(f"prompt:error:{type(e).__name__}: {e}")

    params_raw = meta.get("parameters")
    if params_raw:
        prompt, err = extract_from_parameters(params_raw)
        if prompt:
            return prompt, None
        errors.append(f"parameters:{err}")

    if not errors:
        return None, "no-metadata"
    return None, "; ".join(errors)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_txt(output_path, rows):
    # rows: list of (filename, prompt)
    with open(output_path, "w", encoding="utf-8") as f:
        for _, prompt in rows:
            one_line = " ".join(prompt.replace("\r", " ").replace("\n", " ").split())
            f.write(one_line + "\n")


def _csv_safe_cell(value):
    # Guard against spreadsheet formula injection when the CSV is opened in
    # Excel/LibreOffice: prompts (or filenames) starting with = + - @ tab or CR
    # would otherwise be evaluated as formulas.
    text = str(value)
    if text and text[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + text
    return text


def write_csv(output_path, rows):
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "prompt"])
        for fn, prompt in rows:
            writer.writerow([_csv_safe_cell(fn), _csv_safe_cell(prompt)])


def iter_png_files(input_folder, recursive=False):
    if recursive:
        for root, _, files in os.walk(input_folder):
            for name in sorted(files):
                if name.lower().endswith(".png"):
                    full = os.path.join(root, name)
                    yield os.path.relpath(full, input_folder), full
    else:
        for name in sorted(os.listdir(input_folder)):
            if name.lower().endswith(".png"):
                yield name, os.path.join(input_folder, name)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract the final positive prompt from ComfyUI/A1111 PNG images."
    )
    parser.add_argument("input_folder", nargs="?", default=".",
                        help="Folder to scan for .png files (default: current folder)")
    parser.add_argument("output_file", nargs="?", default=None,
                        help="Output file (default: prompts.txt, or prompts.csv with --csv)")
    parser.add_argument("--csv", action="store_true",
                        help="Write CSV (filename,prompt) instead of plain text")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="Scan subfolders too")
    args = parser.parse_args(argv)

    output_file = args.output_file or ("prompts.csv" if args.csv else "prompts.txt")

    if not os.path.isdir(args.input_folder):
        print(f"ERROR: not a folder: {args.input_folder}")
        return 1

    results = []
    skipped = 0
    for name, path in iter_png_files(args.input_folder, recursive=args.recursive):
        prompt, err = extract_final_positive_prompt_from_png(path)
        if prompt:
            results.append((name, prompt))
        else:
            skipped += 1
            # Be explicit in the console; skip bad files silently in outputs.
            print(f"[skip] {name}: {err}")

    if args.csv or output_file.lower().endswith(".csv"):
        write_csv(output_file, results)
    else:
        write_txt(output_file, results)

    print(f"Extracted {len(results)} prompts ({skipped} skipped) -> {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
