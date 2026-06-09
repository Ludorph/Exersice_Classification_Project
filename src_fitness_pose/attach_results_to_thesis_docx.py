from __future__ import annotations

import argparse
import csv
import html
import re
import shutil
import struct
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import xml.etree.ElementTree as ET


PLACEHOLDER = "[실험 결과 표 및 그림 첨부 예정]"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_TYPES_NS = "http://schemas.openxmlformats.org/package/2006/content-types"


TABLES = [
    ("표 1. 데이터셋 라벨 구성", "table_01_dataset_label_summary.csv"),
    ("표 2. Train/Validation/Test 전체 분할", "table_02_split_overall.csv"),
    ("표 3. 라벨별 데이터 분할 상세", "table_03_split_by_label.csv"),
    ("표 4. 최종 모델 구조 및 하이퍼파라미터", "table_04_final_model_hyperparameters.csv"),
    ("표 5. 최종 5-seed 반복 실험 성능 비교", "table_05_final_repeated_seed_performance.csv"),
    ("표 6. 하이퍼파라미터 튜닝 전후 Macro-F1 비교", "table_06_tuning_before_after_macro_f1.csv"),
    ("표 7. 모델 공통 주요 오분류 조합", "table_07_common_misclassification_patterns.csv"),
    ("표 8. 자세 그룹별 최고 모델", "table_08_pose_group_best_models.csv"),
    ("표 9. 자세 그룹별 모델 Macro-F1 상세", "table_09_pose_group_macro_f1_by_model.csv"),
]


FIGURES = [
    ("그림 1. 최종 5-seed 모델 성능 비교", "figure_01_final_repeated_seed_model_comparison.png"),
    ("그림 2. SVM 대표 혼동행렬(seed 42)", "figure_04_representative_confusion_matrix_svm_seed42.png"),
    ("그림 3. XGBoost 대표 혼동행렬(seed 42)", "figure_05_representative_confusion_matrix_xgboost_seed42.png"),
    ("그림 4. 라벨별 F1-score 비교(seed 42)", "figure_06_representative_per_class_f1_seed42.png"),
    ("그림 5. 자세 그룹별 모델 Macro-F1 비교", "figure_02_pose_group_macro_f1_by_model.png"),
    ("그림 6. 같은 운동군 예측 비율 Heatmap", "figure_03_same_group_prediction_rate_heatmap.png"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach thesis result tables and figures to the revised docx draft.")
    parser.add_argument("--source", type=Path, default=Path("공학논문연구_실험결과_AIHub_본문수정본.docx"))
    parser.add_argument("--output", type=Path, default=Path("공학논문연구_실험결과_AIHub_표그림첨부본.docx"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("outputs_fitness_pose") / "thesis_tables_figures")
    return parser.parse_args()


def esc(text: object) -> str:
    value = "" if text is None else str(text)
    return html.escape(value, quote=False)


def paragraph(text: str = "", *, bold: bool = False, center: bool = False) -> str:
    if not text:
        return "<w:p/>"
    ppr = '<w:pPr><w:jc w:val="center"/></w:pPr>' if center else ""
    rpr = "<w:rPr><w:b/></w:rPr>" if bold else ""
    return f"<w:p>{ppr}<w:r>{rpr}<w:t xml:space=\"preserve\">{esc(text)}</w:t></w:r></w:p>"


def table_xml(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    column_count = max(len(row) for row in rows)
    grid = "".join('<w:gridCol w:w="1800"/>' for _ in range(column_count))
    table_rows = []
    for row_index, row in enumerate(rows):
        cells = []
        padded = row + [""] * (column_count - len(row))
        for value in padded:
            bold = "<w:rPr><w:b/></w:rPr>" if row_index == 0 else ""
            cells.append(
                "<w:tc>"
                "<w:tcPr><w:tcW w:w=\"0\" w:type=\"auto\"/></w:tcPr>"
                f"<w:p><w:r>{bold}<w:t xml:space=\"preserve\">{esc(value)}</w:t></w:r></w:p>"
                "</w:tc>"
            )
        table_rows.append("<w:tr>" + "".join(cells) + "</w:tr>")
    return (
        "<w:tbl>"
        "<w:tblPr>"
        "<w:tblStyle w:val=\"TableGrid\"/>"
        "<w:tblW w:w=\"0\" w:type=\"auto\"/>"
        "<w:tblLook w:val=\"04A0\" w:firstRow=\"1\" w:lastRow=\"0\" w:firstColumn=\"1\" w:lastColumn=\"0\" w:noHBand=\"0\" w:noVBand=\"1\"/>"
        "</w:tblPr>"
        f"<w:tblGrid>{grid}</w:tblGrid>"
        + "".join(table_rows)
        + "</w:tbl>"
    )


def read_csv_table(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [row for row in csv.reader(file)]


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as file:
        header = file.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {path}")
    return struct.unpack(">II", header[16:24])


def image_xml(rel_id: str, image_path: Path, doc_pr_id: int) -> str:
    width_px, height_px = png_size(image_path)
    max_width = 5_900_000
    width_emu = max_width
    height_emu = int(max_width * height_px / width_px)
    max_height = 4_900_000
    if height_emu > max_height:
        height_emu = max_height
        width_emu = int(max_height * width_px / height_px)
    return f"""
<w:p>
  <w:pPr><w:jc w:val="center"/></w:pPr>
  <w:r>
    <w:drawing>
      <wp:inline distT="0" distB="0" distL="0" distR="0">
        <wp:extent cx="{width_emu}" cy="{height_emu}"/>
        <wp:effectExtent l="0" t="0" r="0" b="0"/>
        <wp:docPr id="{doc_pr_id}" name="{esc(image_path.stem)}"/>
        <wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>
        <a:graphic>
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic>
              <pic:nvPicPr>
                <pic:cNvPr id="0" name="{esc(image_path.name)}"/>
                <pic:cNvPicPr/>
              </pic:nvPicPr>
              <pic:blipFill>
                <a:blip r:embed="{rel_id}"/>
                <a:stretch><a:fillRect/></a:stretch>
              </pic:blipFill>
              <pic:spPr>
                <a:xfrm>
                  <a:off x="0" y="0"/>
                  <a:ext cx="{width_emu}" cy="{height_emu}"/>
                </a:xfrm>
                <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
              </pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>
"""


def result_blocks(table_dir: Path, figure_rel_ids: dict[str, str], figure_dir: Path) -> str:
    blocks: list[str] = [
        paragraph("본 절에서는 AI-Hub 피트니스 자세 이미지 데이터셋을 기반으로 수행한 최종 실험 결과를 제시한다. 결과는 데이터 구성, 모델 설정, 최종 성능, 오분류 패턴, 자세 그룹별 분석 순서로 정리하였다."),
        paragraph("최종 성능은 단일 실행 결과가 아니라 5개 random seed 반복 실험의 평균과 표준편차를 기준으로 제시한다. 혼동행렬과 라벨별 F1-score 그림은 대표 seed 42 결과를 사용하므로, 성능 수치의 최종 기준은 반복 seed 평균 표이다."),
        paragraph("3.3.1) 데이터셋 및 분할 결과", bold=True),
    ]
    for caption, filename in TABLES[:3]:
        blocks.append(paragraph(caption, bold=True))
        blocks.append(table_xml(read_csv_table(table_dir / filename)))
        blocks.append(paragraph())
    blocks.append(paragraph("표 1부터 표 3까지는 빈 frame을 가진 JSON 파일을 제외한 뒤 실제 모델 학습과 평가에 사용한 유효 샘플을 기준으로 정리한 결과이다. 전체 데이터는 17개 맨몸운동 라벨로 구성되며, 같은 촬영 세션의 샘플이 서로 다른 분할에 동시에 포함되지 않도록 Train, Validation, Test를 촬영 세션 단위로 나누었다. 이를 통해 동일 인물 또는 동일 촬영 환경의 영향이 학습과 평가에 중복 반영되는 문제를 줄이고자 하였다."))
    blocks.append(paragraph())

    blocks.append(paragraph("3.3.2) 모델 구조 및 하이퍼파라미터", bold=True))
    blocks.append(paragraph(TABLES[3][0], bold=True))
    blocks.append(table_xml(read_csv_table(table_dir / TABLES[3][1])))
    blocks.append(paragraph())
    blocks.append(paragraph("표 4는 최종 실험에 사용한 모델 구조와 주요 하이퍼파라미터를 나타낸다. SVM과 XGBoost는 관절 좌표로부터 생성한 정형 특징 벡터를 입력으로 사용하였고, GNN과 Transformer는 관절별 특징 행렬을 입력으로 사용하여 관절 간 구조적 관계를 학습하도록 구성하였다. 하이퍼파라미터는 Validation set 성능을 기준으로 소규모 탐색을 수행한 뒤 최종 반복 실험에 적용하였다."))
    blocks.append(paragraph())

    blocks.append(paragraph("3.3.3) 최종 성능 비교", bold=True))
    blocks.append(paragraph(TABLES[4][0], bold=True))
    blocks.append(table_xml(read_csv_table(table_dir / TABLES[4][1])))
    blocks.append(paragraph())
    blocks.append(paragraph("표 5의 최종 5-seed 반복 실험 결과에서 SVM은 Macro-F1 0.9044로 가장 높은 평균 성능을 보였고, XGBoost는 Macro-F1 0.9006으로 SVM과 매우 근접한 성능을 보였다. Transformer와 GNN은 각각 0.8770, 0.8462의 Macro-F1을 기록하여 전통적 머신러닝 모델보다 낮은 성능을 보였다. 이는 본 실험에서 사용한 관절 좌표 통계 특징과 데이터 규모에서는 SVM과 XGBoost가 비교적 안정적으로 동작했음을 의미한다."))
    blocks.append(paragraph())
    blocks.append(paragraph(TABLES[5][0], bold=True))
    blocks.append(table_xml(read_csv_table(table_dir / TABLES[5][1])))
    blocks.append(paragraph())
    blocks.append(paragraph("표 6은 하이퍼파라미터 튜닝 전후의 Macro-F1 변화를 비교한 것이다. XGBoost와 GNN은 튜닝 이후 소폭 향상되었고, SVM은 기존 설정과 동일한 수준을 유지하였다. Transformer는 Validation 기준 탐색에서는 개선 가능성이 있었으나 최종 5-seed Test 평균에서는 오히려 낮아졌다. 이는 Validation set에서 좋은 설정이 항상 Test set 반복 평균에서도 더 좋은 결과로 이어지지는 않음을 보여준다."))
    blocks.append(paragraph())
    blocks.append(paragraph(FIGURES[0][0], bold=True, center=True))
    blocks.append(image_xml(figure_rel_ids[FIGURES[0][1]], figure_dir / FIGURES[0][1], 1))
    blocks.append(paragraph())

    blocks.append(paragraph("3.3.4) 오분류 패턴 분석", bold=True))
    blocks.append(paragraph(TABLES[6][0], bold=True))
    blocks.append(table_xml(read_csv_table(table_dir / TABLES[6][1])))
    blocks.append(paragraph())
    blocks.append(paragraph("표 7의 주요 오분류 조합을 보면, 대부분의 오류는 운동 형태가 비슷한 라벨 사이에서 반복적으로 발생하였다. 특히 스텝 백워드 다이나믹 런지와 스텝 포워드 다이나믹 런지, 니푸쉬업과 푸시업, 바이시클 크런치와 크런치처럼 같은 자세군 또는 같은 운동 계열에 속하는 조합에서 혼동이 두드러졌다. 이는 관절 위치 정보만으로도 전체 운동 분류는 가능하지만, 방향 차이나 지지 방식처럼 세밀한 차이를 구분하는 데에는 추가 특징이 필요할 수 있음을 시사한다."))
    blocks.append(paragraph())
    for doc_id, (caption, filename) in enumerate(FIGURES[1:4], start=2):
        blocks.append(paragraph(caption, bold=True, center=True))
        blocks.append(image_xml(figure_rel_ids[filename], figure_dir / filename, doc_id))
        blocks.append(paragraph())

    blocks.append(paragraph("3.3.5) 자세 그룹별 분석", bold=True))
    for caption, filename in TABLES[7:]:
        blocks.append(paragraph(caption, bold=True))
        blocks.append(table_xml(read_csv_table(table_dir / filename)))
        blocks.append(paragraph())
    blocks.append(paragraph("표 8과 표 9는 운동 라벨을 자세 특성에 따라 묶어 모델 성능을 비교한 결과이다. 버피 테스트와 같이 전신 움직임이 뚜렷한 전신 복합 운동이나 굿모닝과 같은 고관절 힌지 운동은 비교적 높은 성능을 보였다. 반면 누운 자세 코어 운동, 상지 지지/엎드린 자세 운동처럼 관절 배치가 서로 유사한 그룹에서는 성능이 상대적으로 낮아졌다. 이러한 결과는 오분류가 단순한 모델 오류라기보다 운동 동작 자체의 구조적 유사성과 관련되어 있음을 보여준다."))
    blocks.append(paragraph())
    for doc_id, (caption, filename) in enumerate(FIGURES[4:6], start=5):
        blocks.append(paragraph(caption, bold=True, center=True))
        blocks.append(image_xml(figure_rel_ids[filename], figure_dir / filename, doc_id))
        blocks.append(paragraph())

    blocks.append(paragraph("이상의 표와 그림을 통해 SVM과 XGBoost가 본 실험 조건에서 가장 안정적인 성능을 보였으며, 주요 오분류는 유사한 자세 또는 같은 운동 계열 내부에서 집중적으로 발생함을 확인할 수 있다."))
    return "".join(blocks)


def next_rel_id(rels_xml: bytes) -> int:
    root = ET.fromstring(rels_xml)
    max_id = 0
    for rel in root:
        rel_id = rel.attrib.get("Id", "")
        if rel_id.startswith("rId") and rel_id[3:].isdigit():
            max_id = max(max_id, int(rel_id[3:]))
    return max_id + 1


def add_image_relationships(rels_xml: bytes, figures: list[tuple[str, str]]) -> tuple[bytes, dict[str, str]]:
    ET.register_namespace("", REL_NS)
    root = ET.fromstring(rels_xml)
    next_id = next_rel_id(rels_xml)
    mapping: dict[str, str] = {}
    for _, filename in figures:
        rel_id = f"rId{next_id}"
        next_id += 1
        mapping[filename] = rel_id
        rel = ET.Element(f"{{{REL_NS}}}Relationship")
        rel.set("Id", rel_id)
        rel.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")
        rel.set("Target", f"media/{filename}")
        root.append(rel)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True), mapping


def ensure_png_content_type(content_types_xml: bytes) -> bytes:
    ET.register_namespace("", CONTENT_TYPES_NS)
    root = ET.fromstring(content_types_xml)
    for child in root:
        if child.tag.endswith("Default") and child.attrib.get("Extension") == "png":
            return content_types_xml
    default = ET.Element(f"{{{CONTENT_TYPES_NS}}}Default")
    default.set("Extension", "png")
    default.set("ContentType", "image/png")
    root.insert(0, default)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def replace_placeholder(document_xml: str, insertion_xml: str) -> str:
    pattern = re.compile(r"<w:p\b(?:(?!</w:p>).)*" + re.escape(PLACEHOLDER) + r"(?:(?!</w:p>).)*</w:p>", re.DOTALL)
    updated, count = pattern.subn(insertion_xml, document_xml, count=1)
    if count != 1:
        raise ValueError(f"Could not find placeholder paragraph: {PLACEHOLDER}")
    return updated


def ensure_drawing_namespaces(document_xml: str) -> str:
    start = document_xml.find("<w:document")
    end = document_xml.find(">", start)
    if start == -1 or end == -1:
        raise ValueError("Could not find w:document root element.")
    root_open = document_xml[start:end]
    additions = []
    if "xmlns:r=" not in root_open:
        additions.append('xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"')
    if "xmlns:wp=" not in root_open:
        additions.append('xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"')
    if "xmlns:a=" not in root_open:
        additions.append('xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"')
    if "xmlns:pic=" not in root_open:
        additions.append('xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"')
    if not additions:
        return document_xml
    return document_xml[:end] + " " + " ".join(additions) + document_xml[end:]


def attach_results(source: Path, output: Path, artifact_dir: Path) -> None:
    table_dir = artifact_dir / "tables"
    figure_dir = artifact_dir / "figures"
    if not source.exists():
        raise FileNotFoundError(source)
    if not table_dir.exists() or not figure_dir.exists():
        raise FileNotFoundError(artifact_dir)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_docx = Path(temp_dir) / "working.docx"
        shutil.copy2(source, temp_docx)
        with ZipFile(temp_docx, "r") as zin:
            rels_xml, figure_rel_ids = add_image_relationships(zin.read("word/_rels/document.xml.rels"), FIGURES)
            insertion = result_blocks(table_dir, figure_rel_ids, figure_dir)
            document_xml = replace_placeholder(zin.read("word/document.xml").decode("utf-8"), insertion)
            document_xml = ensure_drawing_namespaces(document_xml)
            content_types_xml = ensure_png_content_type(zin.read("[Content_Types].xml"))
            with ZipFile(output, "w", ZIP_DEFLATED) as zout:
                written = set()
                for item in zin.infolist():
                    if item.filename == "word/document.xml":
                        zout.writestr(item, document_xml.encode("utf-8"))
                    elif item.filename == "word/_rels/document.xml.rels":
                        zout.writestr(item, rels_xml)
                    elif item.filename == "[Content_Types].xml":
                        zout.writestr(item, content_types_xml)
                    else:
                        zout.writestr(item, zin.read(item.filename))
                    written.add(item.filename)
                for _, filename in FIGURES:
                    media_name = f"word/media/{filename}"
                    if media_name not in written:
                        zout.write(figure_dir / filename, media_name)


def main() -> None:
    args = parse_args()
    attach_results(args.source, args.output, args.artifact_dir)
    print(f"Completed. Thesis with tables and figures: {args.output.resolve()}")


if __name__ == "__main__":
    main()
