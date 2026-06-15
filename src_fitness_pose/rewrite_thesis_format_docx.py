from __future__ import annotations

import argparse
import html
import shutil
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite the thesis draft into the provided paper format.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("관절 위치 정보 기반 운동 동작 분류 및 성능 및 오분류 패턴 비교 분석.docx"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("관절 위치 정보 기반 운동 동작 분류 및 성능 및 오분류 패턴 비교 분석_논문포맷적용본.docx"),
    )
    return parser.parse_args()


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def run_properties(*, bold: bool = False, size: int = 20) -> str:
    bold_xml = "<w:b/>" if bold else ""
    return (
        "<w:rPr>"
        '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
        'w:eastAsia="Times New Roman" w:cs="Times New Roman"/>'
        f"{bold_xml}"
        f'<w:sz w:val="{size}"/>'
        f'<w:szCs w:val="{size}"/>'
        "</w:rPr>"
    )


def paragraph(text: str = "", *, bold: bool = False, center: bool = False, size: int = 20) -> str:
    if not text:
        return "<w:p/>"
    align_xml = '<w:pPr><w:jc w:val="center"/></w:pPr>' if center else ""
    return (
        "<w:p>"
        f"{align_xml}"
        "<w:r>"
        f"{run_properties(bold=bold, size=size)}"
        f'<w:t xml:space="preserve">{esc(text)}</w:t>'
        "</w:r>"
        "</w:p>"
    )


def equation(text: str) -> str:
    return (
        '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
        "<m:oMathPara><m:oMath>"
        f"<m:r><m:t>{esc(text)}</m:t></m:r>"
        "</m:oMath></m:oMathPara></w:p>"
    )


def bullets(items: list[str], *, size: int = 20) -> list[str]:
    return [paragraph(f"- {item}", size=size) for item in items]


def doc_xml(body_blocks: list[str]) -> str:
    section = """
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>
      <w:cols w:space="425"/>
      <w:docGrid w:linePitch="360"/>
    </w:sectPr>
    """
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{W_NS}" xmlns:r="{R_NS}" xmlns:wp="{WP_NS}" '
        f'xmlns:a="{A_NS}" xmlns:pic="{PIC_NS}" xmlns:m="{M_NS}"><w:body>'
        + "".join(body_blocks)
        + section
        + "</w:body></w:document>"
    )


def formatted_blocks() -> list[str]:
    blocks: list[str] = []

    title = "관절 위치 정보 기반 운동 동작 분류 및 성능 및 오분류 패턴 비교 분석"
    keywords = (
        "Keyword : 운동 동작 분류, 관절 위치 정보, 3D 관절 좌표, "
        "AI-Hub 피트니스 자세 데이터셋, SVM, XGBoost, GNN, Transformer, 오분류 분석"
    )

    blocks.append(paragraph(title, center=True, bold=True, size=36))
    blocks.append(paragraph("김한서", center=True, size=20))
    blocks.append(paragraph("유한대학교 컴퓨터공학과", center=True, size=20))
    blocks.append(paragraph("손석기", center=True, size=20))
    blocks.append(paragraph("유한대학교 컴퓨터공학과", center=True, size=20))
    blocks.append(paragraph("최서혁", center=True, size=20))
    blocks.append(paragraph("유한대학교 컴퓨터공학과", center=True, size=20))
    blocks.append(paragraph())

    blocks.append(paragraph("Abstract", bold=True, size=18))
    blocks.append(
        paragraph(
            "본 연구는 AI-Hub 피트니스 자세 이미지 데이터셋의 3D 관절 좌표를 활용하여 "
            "맨몸운동 17개 동작을 분류하고, SVM, XGBoost, GNN, Transformer 모델의 성능과 "
            "오분류 패턴을 비교한다. 각 -3d.json 파일은 하나의 운동 수행 샘플로 정의하고, "
            "프레임별 24개 관절 좌표를 통계 특징으로 변환하였다. 5개 random seed 반복 실험에서 "
            "SVM과 XGBoost가 가장 안정적인 성능을 보였고, 주요 오분류는 푸시업-니푸쉬업, "
            "런지 계열, 크런치 계열처럼 자세와 관절 움직임이 유사한 라벨 사이에 집중되었다. "
            "이 결과는 관절 좌표 기반 운동 동작 분류에서 모델 성능뿐 아니라 라벨 간 구조적 유사성도 "
            "함께 고려해야 함을 보여준다.",
            size=18,
        )
    )
    blocks.append(paragraph(keywords, size=18))
    blocks.append(paragraph())

    blocks.append(paragraph("1. Introduction", bold=True))
    blocks.append(
        paragraph(
            "인공지능 기반 동작 인식 기술은 스포츠, 피트니스, 재활 운동, 스마트 트레이닝 시스템 등에서 "
            "사용자의 운동 동작을 자동으로 인식하고 분석하기 위한 핵심 기술로 활용된다. 특히 운동 동작 "
            "분류는 사용자가 수행한 동작의 종류를 판별하는 과정으로, 자세 피드백이나 운동 기록 자동화의 "
            "기초 단계가 된다."
        )
    )
    blocks.append(
        paragraph(
            "본 연구는 원본 영상이 아니라 관절 위치 정보에 주목하였다. 관절 좌표는 인체 구조와 움직임을 "
            "간결하게 표현할 수 있으며, 영상 전체를 직접 사용하는 방식보다 개인정보 노출 가능성이 낮고 "
            "모델 입력으로 구성하기 쉽다. AI-Hub 피트니스 자세 이미지 데이터셋은 운동 수행 과정에서 "
            "추출된 3차원 관절 좌표를 제공하므로, 관절 좌표 기반 운동 동작 분류 실험에 적합하다."
        )
    )
    blocks.append(
        paragraph(
            "기존 실험은 푸시업, 풀업, 스쿼트처럼 제한된 운동 라벨을 대상으로 수행되었으나, 본 연구에서는 "
            "데이터셋을 AI-Hub 피트니스 자세 이미지 데이터셋으로 변경하고 맨몸운동 17개 라벨로 범위를 "
            "확장하였다. 또한 단순히 전체 정확도만 비교하지 않고, 모델별 주요 오분류 조합과 자세 그룹별 "
            "성능 차이를 함께 분석하였다."
        )
    )
    blocks.append(
        paragraph(
            "본 연구의 목적은 첫째, 3D 관절 좌표 JSON으로부터 운동 동작 분류용 입력 특징을 구성하고, "
            "둘째, SVM, XGBoost, GNN, Transformer 네 모델의 성능을 동일한 데이터 분할 조건에서 비교하며, "
            "셋째, 모델별 오분류 패턴을 분석하여 관절 좌표 기반 분류의 가능성과 한계를 제시하는 것이다."
        )
    )
    blocks.append(paragraph())

    blocks.append(paragraph("2. Data", bold=True))
    blocks.append(
        paragraph(
            "본 연구에서 사용한 데이터는 AI-Hub 피트니스 자세 이미지 데이터셋이다. 해당 데이터셋은 운동 "
            "이미지 및 자세 주석 정보를 포함하며, 본 연구에서는 중복 입력을 줄이고 3차원 관절 정보를 "
            "활용하기 위해 -3d.json 파일만 사용하였다."
        )
    )
    blocks.append(
        paragraph(
            "각 -3d.json 파일은 하나의 운동 수행 샘플로 정의하였다. 파일 내부에는 여러 프레임이 포함되어 "
            "있고, 각 프레임에는 24개 관절의 x, y, z 좌표가 저장되어 있다. 즉, 본 연구의 평가 단위는 "
            "개별 프레임이 아니라 하나의 운동 수행을 나타내는 JSON 파일 단위이다."
        )
    )
    blocks.append(
        paragraph(
            "실험 대상은 맨몸운동 17개 라벨이며, 비어 있는 프레임만 포함한 JSON 파일은 제외하였다. "
            "최종적으로 16,408개의 유효 샘플을 사용하였다. Train, Validation, Test는 촬영 세션 단위로 "
            "분리하여 동일 세션의 샘플이 학습과 평가에 동시에 포함되는 문제를 줄이고자 하였다."
        )
    )
    blocks.append(
        paragraph(
            "사용 라벨은 Y - Exercise, 굿모닝, 니푸쉬업, 라잉 레그 레이즈, 바이시클 크런치, 버피 테스트, "
            "사이드 런지, 스탠딩 니업, 스탠딩 사이드 크런치, 스텝 백워드 다이나믹 런지, "
            "스텝 포워드 다이나믹 런지, 시저크로스, 크런치, 크로스 런지, 푸시업, 플랭크, "
            "힘쓰러스트이다. 이 중 Y - Exercise는 데이터셋에 명시된 원 라벨명을 그대로 사용하였다."
        )
    )
    blocks.append(
        paragraph(
            "Validation set은 모델 설정과 하이퍼파라미터를 선택하기 위한 중간 검증 데이터이며, Test set은 "
            "최종 성능을 확인하기 위한 평가 데이터이다. 따라서 Test 성능은 모델 선택이 끝난 뒤의 최종 "
            "비교 지표로 사용하였다."
        )
    )
    blocks.append(paragraph())

    blocks.append(paragraph("3. Methodology", bold=True))
    blocks.append(paragraph("3.1) 입력 데이터 및 특징 구성", bold=True))
    blocks.append(
        paragraph(
            "i번째 운동 수행 샘플의 관절 좌표 집합은 다음과 같이 정의한다. 여기서 T_i는 i번째 샘플의 "
            "프레임 수, J는 관절 수를 의미한다."
        )
    )
    blocks.append(equation("(1)  X_i = {p_{t,j} | t = 1,...,T_i, j = 1,...,J}"))
    blocks.append(equation("(2)  p_{t,j} = (x_{t,j}, y_{t,j}, z_{t,j})"))
    blocks.append(
        paragraph(
            "각 관절에 대해 평균, 표준편차, 최솟값, 최댓값, 움직임 범위, 시작-종료 변화량, 평균 이동량을 "
            "계산하였다. 대표적인 특징 수식은 다음과 같다."
        )
    )
    blocks.append(equation("(3)  mean_j = (1 / T_i) sum_{t=1}^{T_i} p_{t,j}"))
    blocks.append(equation("(4)  std_j = sqrt((1 / T_i) sum_{t=1}^{T_i} ||p_{t,j} - mean_j||^2)"))
    blocks.append(equation("(5)  range_j = max_t(p_{t,j}) - min_t(p_{t,j})"))
    blocks.append(equation("(6)  delta_j = p_{T_i,j} - p_{1,j}"))
    blocks.append(equation("(7)  move_j = (1 / (T_i - 1)) sum_{t=2}^{T_i} ||p_{t,j} - p_{t-1,j}||"))
    blocks.append(
        paragraph(
            "SVM과 XGBoost에는 관절별 통계 특징을 1차원 벡터로 펼친 입력을 사용하였다. GNN과 Transformer에는 "
            "관절을 개별 노드 또는 token으로 유지한 관절별 특징 행렬을 입력으로 사용하였다. 이러한 입력 "
            "형식의 차이는 결과 해석과도 연결된다. 즉, SVM과 XGBoost는 요약된 정형 특징 벡터에 강점을 "
            "보일 수 있고, GNN과 Transformer는 관절 간 구조적 관계를 학습하는 데 초점을 둔다."
        )
    )
    blocks.append(paragraph("[방법론 그림 첨부 예정]"))

    blocks.append(paragraph("3.2) 모델별 구조 및 수식", bold=True))
    blocks.append(
        paragraph(
            "SVM은 RBF kernel을 사용하여 비선형 분류 경계를 학습하였다. 다중 클래스 분류에서는 각 클래스에 "
            "대한 결정 함수 값을 계산하고, 가장 큰 값을 갖는 클래스를 최종 예측 라벨로 선택한다."
        )
    )
    blocks.append(equation("(8)  K(x_i, x_j) = exp(-gamma ||x_i - x_j||^2)"))
    blocks.append(
        paragraph(
            "XGBoost는 여러 개의 decision tree를 순차적으로 결합하는 gradient boosting 기반 앙상블 모델이다. "
            "각 샘플의 예측값은 여러 tree의 출력 합으로 표현된다."
        )
    )
    blocks.append(equation("(9)  y_hat_i = sum_{k=1}^{K} f_k(x_i),  f_k in F"))
    blocks.append(equation("(10)  Obj = sum_i l(y_i, y_hat_i) + sum_{k=1}^{K} Omega(f_k)"))
    blocks.append(
        paragraph(
            "GNN은 관절을 노드로, 인체 골격 연결 관계를 edge로 정의하여 관절 간 구조적 관계를 학습한다. "
            "정규화된 인접행렬과 그래프 합성곱 연산은 다음과 같다."
        )
    )
    blocks.append(equation("(11)  A_hat = D^(-1/2)(A + I)D^(-1/2)"))
    blocks.append(equation("(12)  H^(l+1) = sigma(A_hat H^(l) W^(l))"))
    blocks.append(
        paragraph(
            "Transformer는 각 관절을 token으로 보고 self-attention을 적용하여 관절 간 상호작용을 학습한다. "
            "self-attention의 기본 식은 다음과 같다."
        )
    )
    blocks.append(equation("(13)  Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V"))

    blocks.append(paragraph("3.3) 학습 및 평가 지표", bold=True))
    blocks.append(
        paragraph(
            "GNN과 Transformer는 다중 클래스 분류 문제로 학습하였으며, 손실 함수로 Cross-Entropy Loss를 사용하였다."
        )
    )
    blocks.append(equation("(14)  L = - sum_{c=1}^{C} y_c log(p_c)"))
    blocks.append(
        paragraph(
            "성능 평가는 Accuracy, Precision, Recall, F1-score, Macro-F1, Weighted-F1을 기준으로 수행하였다. "
            "라벨별 샘플 수가 균등하지 않기 때문에 본 연구에서는 모든 라벨을 동일한 비중으로 반영하는 "
            "Macro-F1을 주요 비교 지표로 사용하였다."
        )
    )
    blocks.append(
        paragraph(
            "다중 클래스 분류에서 Accuracy는 전체 N개 평가 샘플 중 실제 라벨 y_i와 예측 라벨 "
            "y_hat_i가 일치한 비율로 해석한다. Precision, Recall, F1-score는 각 클래스별로 계산한 뒤 "
            "Macro-F1 또는 Weighted-F1 방식으로 평균낸다."
        )
    )
    blocks.append(equation("(15)  Accuracy = (1 / N) sum_{i=1}^{N} I(y_i = y_hat_i)"))
    blocks.append(equation("(16)  Precision_c = TP_c / (TP_c + FP_c)"))
    blocks.append(equation("(17)  Recall_c = TP_c / (TP_c + FN_c)"))
    blocks.append(equation("(18)  F1_c = 2 * Precision_c * Recall_c / (Precision_c + Recall_c)"))
    blocks.append(equation("(19)  Macro-F1 = (1 / C) sum_{c=1}^{C} F1_c"))
    blocks.append(equation("(20)  Weighted-F1 = sum_{c=1}^{C} (n_c / N) F1_c"))
    blocks.append(paragraph())

    blocks.append(paragraph("4. Numerical results", bold=True))
    blocks.append(paragraph("[실험 결과 표 및 그림 첨부 예정]"))
    blocks.append(paragraph())

    blocks.append(paragraph("5. Conclusions", bold=True))
    blocks.append(
        paragraph(
            "본 연구는 AI-Hub 피트니스 자세 이미지 데이터셋의 3D 관절 좌표를 활용하여 맨몸운동 17개 라벨을 "
            "분류하고, SVM, XGBoost, GNN, Transformer 모델의 성능과 오분류 패턴을 비교하였다. 실험 결과 "
            "SVM과 XGBoost가 가장 안정적인 성능을 보였으며, GNN과 Transformer도 관절 간 관계를 반영하는 "
            "구조적 입력 방식의 가능성을 확인하였다."
        )
    )
    blocks.append(
        paragraph(
            "오분류는 무작위로 발생하기보다 자세와 관절 움직임이 유사한 운동 라벨 사이에서 반복적으로 나타났다. "
            "예를 들어 푸시업과 니푸쉬업, 런지 계열, 크런치 계열은 관절 배치와 움직임 패턴이 유사하여 "
            "모델이 혼동하기 쉬운 조합으로 확인되었다. 따라서 운동 동작 분류 연구에서는 전체 성능뿐 아니라 "
            "라벨 간 구조적 유사성에 따른 오분류 분석이 함께 필요하다."
        )
    )
    blocks.append(
        paragraph(
            "다만 본 연구는 AI-Hub 데이터셋에서 제공하는 관절 좌표를 기반으로 수행되었으므로, 직접 촬영한 "
            "웹캠 영상이나 MediaPipe로 추출한 좌표에 그대로 일반화된다고 단정하기는 어렵다. 향후 연구에서는 "
            "직접 촬영 데이터와 AI-Hub 기준 데이터의 좌표 형식, 관절 수, 정규화 방식, 샘플 단위를 일치시킨 뒤 "
            "외부 데이터에 대한 일반화 성능을 추가로 검증할 필요가 있다."
        )
    )
    blocks.append(paragraph())

    blocks.append(paragraph("6. References", bold=True, size=18))
    references = [
        "Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20, 273-297.",
        "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of KDD.",
        "Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR.",
        "Vaswani, A. et al. (2017). Attention is all you need. NeurIPS.",
        "AI-Hub. 피트니스 자세 이미지 데이터셋.",
    ]
    for ref in references:
        blocks.append(paragraph(ref, size=18))

    return blocks


def rewrite_docx(source: Path, output: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "paper_format.docx"
        shutil.copy2(source, tmp_path)
        with ZipFile(tmp_path, "r") as zin, ZipFile(output, "w", ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "word/document.xml":
                    data = doc_xml(formatted_blocks()).encode("utf-8")
                zout.writestr(item, data)


def main() -> None:
    args = parse_args()
    rewrite_docx(args.source, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
