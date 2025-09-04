#!/usr/bin/env python3
# extract_prompts.py
# Usage:
#   python extract_prompts.py
#   python extract_prompts.py --csv
#   python extract_prompts.py "C:\path\to\images"
#   python extract_prompts.py "C:\path\to\images" "C:\out\my_prompts.csv" --csv

import os
import sys
import json
import csv
from PIL import Image

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
    cands = [n for n in nodes if ("SaveImage" in n.get("type", "") or "Save Image" in n.get("type",""))]
    if not cands:
        return None
    return max(cands, key=lambda n: n.get("order", 0))

def find_input_link_id(node, name):
    for inp in node.get("inputs", []):
        if inp.get("name") == name and "link" in inp:
            return inp["link"]
    return None

def bfs_upstream_to_positive_source(workflow, start_node):
    """
    Starting at the node that feeds SaveImage (e.g., VAEDecode, FaceDetailer),
    walk upstream until we find a node with a 'positive' input. Return the node
    that *produces* that 'positive' (its source node).
    """
    from collections import deque
    link_map = build_link_map(workflow)
    q = deque([start_node])
    seen = set()
    while q:
        node = q.popleft()
        if node["id"] in seen:
            continue
        seen.add(node["id"])

        for inp in node.get("inputs", []):
            if inp.get("name") == "positive" and "link" in inp:
                src_id = link_map[inp["link"]][0]
                return get_node_by_id(workflow, src_id)

        # keep walking upstream through all inputs
        for inp in node.get("inputs", []):
            if "link" in inp:
                src_id = link_map[inp["link"]][0]
                pred = get_node_by_id(workflow, src_id)
                if pred:
                    q.append(pred)
    return None

def find_upstream_clip_encode(workflow, start_node):
    """
    From a CONDITIONING-producing node, walk upstream until a CLIPTextEncode* node is found.
    """
    from collections import deque
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
            if "link" in inp and inp.get("type") in ("CONDITIONING", "CLIP", "STRING", "any"):
                src_id = link_map[inp["link"]][0]
                pred = get_node_by_id(workflow, src_id)
                if pred:
                    q.append(pred)
    return None

def extract_string_value_recursive(workflow, node):
    """
    Resolve the actual prompt text. Priority:
      1) If CLIP node has no 'text' link, take its widget value (usually widgets_values[0]).
      2) Else follow its 'text' input link back. For generic nodes, try any STRING-typed input.
      3) When landing on nodes with widgets_values containing strings, pick the longest.
    """
    link_map = build_link_map(workflow)
    visited = set()

    def flatten_strings(x):
        if isinstance(x, str):
            return [x]
        if isinstance(x, list):
            out = []
            for e in x:
                out.extend(flatten_strings(e))
            return out
        return []

    def helper(n):
        if n["id"] in visited:
            return None
        visited.add(n["id"])

        # Case: CLIP node with inline text
        if "CLIPTextEncode" in n.get("type", ""):
            # If 'text' is not linked, prefer its own widget value
            text_input = next((i for i in n.get("inputs", []) if i.get("name") == "text"), None)
            linked = bool(text_input and "link" in text_input)
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
            if inp.get("name") == "text" and "link" in inp:
                src_id = link_map[inp["link"]][0]
                pred = get_node_by_id(workflow, src_id)
                if pred:
                    res = helper(pred)
                    if res:
                        return res

        # Otherwise follow any STRING input
        for inp in n.get("inputs", []):
            if inp.get("type") == "STRING" and "link" in inp:
                src_id = link_map[inp["link"]][0]
                pred = get_node_by_id(workflow, src_id)
                if pred:
                    res = helper(pred)
                    if res:
                        return res
        return None

    return helper(node)

def extract_final_positive_prompt_from_png(png_path):
    try:
        with Image.open(png_path) as img:
            meta = img.info
            wf_raw = meta.get("workflow")
            if not wf_raw:
                return None, "no-workflow"
            workflow = json.loads(wf_raw)

        save = find_highest_order_saveimage(workflow)
        if not save:
            return None, "no-saveimage"

        images_link = find_input_link_id(save, "images")
        if not images_link:
            return None, "no-saveimage-images-link"

        link_map = build_link_map(workflow)
        src_id = link_map[images_link][0]
        start_node = get_node_by_id(workflow, src_id)
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
    except Exception as e:
        return None, f"error:{e}"

def write_txt(output_path, rows):
    # rows: list of (filename, prompt)
    with open(output_path, "w", encoding="utf-8") as f:
        for _, prompt in rows:
            one_line = " ".join(prompt.replace("\r", " ").replace("\n", " ").split())
            f.write(one_line + "\n")

def write_csv(output_path, rows):
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "prompt"])
        for fn, prompt in rows:
            writer.writerow([fn, prompt])

def main():
    # Defaults
    use_csv = "--csv" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--csv"]

    input_folder = args[0] if len(args) >= 1 else "."
    output_file  = args[1] if len(args) >= 2 else ("prompts.csv" if use_csv else "prompts.txt")

    # Collect prompts
    results = []
    for name in os.listdir(input_folder):
        if name.lower().endswith(".png"):
            path = os.path.join(input_folder, name)
            prompt, err = extract_final_positive_prompt_from_png(path)
            if prompt:
                results.append((name, prompt))
            else:
                # Be explicit in the console; skip bad files silently in outputs.
                print(f"[skip] {name}: {err}")

    # Write
    if use_csv or output_file.lower().endswith(".csv"):
        write_csv(output_file, results)
    else:
        write_txt(output_file, results)

    print(f"Extracted {len(results)} prompts -> {output_file}")

if __name__ == "__main__":
    main()