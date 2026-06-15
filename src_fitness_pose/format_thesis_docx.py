from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import xml.etree.ElementTree as ET


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NAMESPACES = {
    "w": W_NS,
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Format thesis docx font family and sizes.")
    parser.add_argument("docx", type=Path)
    return parser.parse_args()


def w_tag(name: str) -> str:
    return f"{{{W_NS}}}{name}"


def ensure_child(parent: ET.Element, tag: str, index: int | None = None) -> ET.Element:
    child = parent.find(tag)
    if child is not None:
        return child
    child = ET.Element(tag)
    if index is None:
        parent.append(child)
    else:
        parent.insert(index, child)
    return child


def paragraph_text(paragraph: ET.Element) -> str:
    return "".join(text.text or "" for text in paragraph.findall(".//w:t", NAMESPACES)).strip()


def remove_children(parent: ET.Element, tag: str) -> None:
    for child in list(parent.findall(tag)):
        parent.remove(child)


def set_run_font(run: ET.Element, half_points: int) -> None:
    rpr = ensure_child(run, w_tag("rPr"), 0)

    remove_children(rpr, w_tag("rFonts"))
    fonts = ET.Element(w_tag("rFonts"))
    for key in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(w_tag(key), "Times New Roman")
    rpr.insert(0, fonts)

    remove_children(rpr, w_tag("sz"))
    size = ET.Element(w_tag("sz"))
    size.set(w_tag("val"), str(half_points))
    rpr.append(size)

    remove_children(rpr, w_tag("szCs"))
    size_cs = ET.Element(w_tag("szCs"))
    size_cs.set(w_tag("val"), str(half_points))
    rpr.append(size_cs)


def target_size_for_paragraph(index_with_text: int, text: str, in_references: bool) -> int:
    if index_with_text == 1:
        return 36  # 18 pt
    if in_references or "논문 키워드" in text or "keyword" in text.lower() or text.startswith(": 운동 동작 분류"):
        return 18  # 9 pt
    return 20  # 10 pt


def style_document(document_xml: bytes) -> tuple[bytes, dict[str, int]]:
    for prefix, uri in NAMESPACES.items():
        ET.register_namespace(prefix, uri)
    root = ET.fromstring(document_xml)

    stats = {"paragraphs": 0, "title_18": 0, "small_9": 0, "body_10": 0, "runs": 0}
    text_paragraph_index = 0
    in_references = False

    for paragraph in root.findall(".//w:p", NAMESPACES):
        text = paragraph_text(paragraph)
        if not text:
            continue
        text_paragraph_index += 1
        if text.startswith("5. 참고문헌") or text.lower().startswith("references"):
            in_references = True

        size = target_size_for_paragraph(text_paragraph_index, text, in_references)
        if size == 36:
            stats["title_18"] += 1
        elif size == 18:
            stats["small_9"] += 1
        else:
            stats["body_10"] += 1

        for run in paragraph.findall(".//w:r", NAMESPACES):
            set_run_font(run, size)
            stats["runs"] += 1
        stats["paragraphs"] += 1

    return ET.tostring(root, encoding="utf-8", xml_declaration=True), stats


def style_docx(docx_path: Path) -> dict[str, int]:
    if not docx_path.exists():
        raise FileNotFoundError(docx_path)
    temp_output = docx_path.with_suffix(".tmp.docx")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_input = Path(temp_dir) / docx_path.name
        shutil.copy2(docx_path, temp_input)
        with ZipFile(temp_input, "r") as zin, ZipFile(temp_output, "w", ZIP_DEFLATED) as zout:
            stats = {"paragraphs": 0, "title_18": 0, "small_9": 0, "body_10": 0, "runs": 0}
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "word/document.xml":
                    data, stats = style_document(data)
                zout.writestr(item, data)
    shutil.move(str(temp_output), docx_path)
    return stats


def main() -> None:
    args = parse_args()
    stats = style_docx(args.docx)
    print(f"Formatted thesis docx: {args.docx.resolve()}")
    print(stats)


if __name__ == "__main__":
    main()
