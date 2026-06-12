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
HEADER_FILL = "D9EAF7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply visible borders and header shading to every table in a docx file.")
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


def set_table_borders(tbl: ET.Element) -> None:
    tbl_pr = tbl.find(w_tag("tblPr"))
    if tbl_pr is None:
        tbl_pr = ET.Element(w_tag("tblPr"))
        tbl.insert(0, tbl_pr)

    for existing in list(tbl_pr.findall(w_tag("tblBorders"))):
        tbl_pr.remove(existing)

    borders = ET.Element(w_tag("tblBorders"))
    for border_name in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = ET.SubElement(borders, w_tag(border_name))
        border.set(w_tag("val"), "single")
        border.set(w_tag("sz"), "8")
        border.set(w_tag("space"), "0")
        border.set(w_tag("color"), "000000")
    tbl_pr.append(borders)


def shade_header_row(tbl: ET.Element) -> None:
    rows = tbl.findall(w_tag("tr"))
    if not rows:
        return
    header = rows[0]
    for cell in header.findall(w_tag("tc")):
        tc_pr = ensure_child(cell, w_tag("tcPr"), 0)
        for existing in list(tc_pr.findall(w_tag("shd"))):
            tc_pr.remove(existing)
        shading = ET.Element(w_tag("shd"))
        shading.set(w_tag("val"), "clear")
        shading.set(w_tag("color"), "auto")
        shading.set(w_tag("fill"), HEADER_FILL)
        tc_pr.append(shading)


def style_document_xml(document_xml: bytes) -> tuple[bytes, int]:
    for prefix, uri in NAMESPACES.items():
        ET.register_namespace(prefix, uri)
    root = ET.fromstring(document_xml)
    tables = root.findall(".//w:tbl", NAMESPACES)
    for table in tables:
        set_table_borders(table)
        shade_header_row(table)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True), len(tables)


def style_docx_tables(docx_path: Path) -> int:
    if not docx_path.exists():
        raise FileNotFoundError(docx_path)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / docx_path.name
        shutil.copy2(docx_path, temp_path)
        with ZipFile(temp_path, "r") as zin, ZipFile(docx_path.with_suffix(".tmp.docx"), "w", ZIP_DEFLATED) as zout:
            table_count = 0
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "word/document.xml":
                    data, table_count = style_document_xml(data)
                zout.writestr(item, data)
        shutil.move(str(docx_path.with_suffix(".tmp.docx")), docx_path)
    return table_count


def main() -> None:
    args = parse_args()
    count = style_docx_tables(args.docx)
    print(f"Styled {count} tables in {args.docx.resolve()}")


if __name__ == "__main__":
    main()
