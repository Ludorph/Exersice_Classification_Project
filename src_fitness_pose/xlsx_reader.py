from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _column_name(cell_reference: str) -> str:
    return re.sub(r"\d", "", cell_reference)


def _shared_strings(archive: ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    values: list[str] = []
    for item in root.findall(f"{{{MAIN_NS}}}si"):
        values.append("".join(node.text or "" for node in item.iter(f"{{{MAIN_NS}}}t")))
    return values


def _sheet_target(archive: ZipFile, sheet_name: str) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    relationship_targets = {
        node.attrib["Id"]: node.attrib["Target"]
        for node in relationships.findall(f"{{{PACKAGE_REL_NS}}}Relationship")
    }
    for sheet in workbook.findall(f".//{{{MAIN_NS}}}sheet"):
        if sheet.attrib["name"] == sheet_name:
            relation_id = sheet.attrib[f"{{{REL_NS}}}id"]
            target = relationship_targets[relation_id].replace("\\", "/")
            return target if target.startswith("xl/") else f"xl/{target}"
    raise KeyError(f"Sheet not found: {sheet_name}")


def read_sheet_rows(path: Path, sheet_name: str) -> list[dict[str, str]]:
    """Read an xlsx sheet into rows keyed by Excel column letters.

    This intentionally supports the small subset of xlsx needed by the
    AI-Hub mapping workbooks and avoids requiring Excel or openpyxl.
    """

    with ZipFile(path) as archive:
        shared = _shared_strings(archive)
        root = ET.fromstring(archive.read(_sheet_target(archive, sheet_name)))

    rows: list[dict[str, str]] = []
    for row in root.findall(f".//{{{MAIN_NS}}}row"):
        values: dict[str, str] = {}
        for cell in row.findall(f"{{{MAIN_NS}}}c"):
            reference = cell.attrib["r"]
            cell_type = cell.attrib.get("t")
            value_node = cell.find(f"{{{MAIN_NS}}}v")
            if cell_type == "inlineStr":
                value = "".join(node.text or "" for node in cell.iter(f"{{{MAIN_NS}}}t"))
            elif value_node is None:
                value = ""
            elif cell_type == "s":
                value = shared[int(value_node.text or "0")]
            else:
                value = value_node.text or ""
            values[_column_name(reference)] = value.strip()
        rows.append(values)
    return rows

