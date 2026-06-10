from __future__ import annotations

import argparse
import html
import shutil
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite the old thesis draft for the AI-Hub fitness pose project.")
    parser.add_argument("--source", type=Path, default=Path("공학논문연구_실험결과.docx"))
    parser.add_argument("--output", type=Path, default=Path("공학논문연구_실험결과_AIHub_본문수정본.docx"))
    return parser.parse_args()


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def paragraph(text: str = "", *, bold: bool = False, center: bool = False) -> str:
    if not text:
        return "<w:p/>"
    bold_xml = "<w:rPr><w:b/></w:rPr>" if bold else ""
    align_xml = '<w:pPr><w:jc w:val="center"/></w:pPr>' if center else ""
    return (
        "<w:p>"
        f"{align_xml}"
        "<w:r>"
        f"{bold_xml}"
        f'<w:t xml:space="preserve">{esc(text)}</w:t>'
        "</w:r>"
        "</w:p>"
    )


def bullets(items: list[str]) -> list[str]:
    return [paragraph(f"• {item}") for item in items]


def numbered(items: list[str]) -> list[str]:
    return [paragraph(f"{index}. {item}") for index, item in enumerate(items, start=1)]


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
        f'<w:document xmlns:w="{W_NS}"><w:body>'
        + "".join(body_blocks)
        + section
        + "</w:body></w:document>"
    )


def revised_blocks() -> list[str]:
    blocks: list[str] = []

    blocks.append(paragraph("관절 위치 정보 기반 운동 동작 분류 및 SVM, XGBoost, GNN, Transformer 모델 성능 비교 분석", center=True))
    blocks.append(paragraph())

    blocks.append(paragraph("1. 제목 및 저자", bold=True))
    blocks.append(paragraph("1.1) 논문제목", bold=True))
    blocks.append(paragraph(": 관절 위치 정보 기반 운동 동작 분류에서 SVM, XGBoost, GNN, Transformer 모델 성능 및 오분류 패턴 비교 분석"))
    blocks.append(paragraph("1.2) 저자", bold=True))
    blocks.append(paragraph(": 김한서, 손석기, 최서혁"))
    blocks.append(paragraph("1.3) 논문 키워드", bold=True))
    blocks.append(paragraph(": 운동 동작 분류, 관절 위치 정보, 3D 관절 좌표, AI-Hub 피트니스 자세 데이터셋, SVM, XGBoost, GNN, Transformer, 오분류 분석"))
    blocks.append(paragraph())

    blocks.append(paragraph("2. 연구 배경 및 목적", bold=True))
    blocks.append(paragraph("2.1) 연구 배경 및 필요성", bold=True))
    blocks.append(paragraph(": 최근 인공지능 기술의 발전으로 스포츠 및 피트니스 분야에서도 사용자의 운동 동작을 자동으로 인식하고 분석하는 연구가 활발히 수행되고 있다. 홈트레이닝, 스마트 피트니스, 재활 운동, 운동 자세 피드백 시스템과 같은 응용 분야에서는 사용자가 수행한 운동 종류를 정확히 분류하는 기술이 중요한 기반 요소가 된다."))
    blocks.append(paragraph("운동 동작 분류에는 영상, 웨어러블 센서, 관절 위치 정보 등 다양한 입력 데이터가 사용될 수 있다. 이 중 관절 위치 정보는 사람의 신체 구조와 움직임을 비교적 간결하게 표현할 수 있으며, 원본 영상보다 개인정보 노출 위험이 낮고 모델 입력으로 사용하기 쉽다는 장점이 있다. 특히 3차원 관절 좌표는 2차원 좌표보다 신체의 공간적 배치와 움직임을 풍부하게 반영할 수 있어 운동 동작 분류 연구에 적합하다."))
    blocks.append(paragraph("다만 운동 동작은 서로 유사한 자세와 관절 움직임을 포함하는 경우가 많다. 예를 들어 푸시업과 니푸쉬업은 상체 지지와 팔 굽힘 동작이 유사하며, 크런치 계열 운동은 모두 누운 자세에서 몸통 또는 하체 움직임이 나타난다. 따라서 단순한 전체 정확도뿐 아니라 어떤 운동 조합에서 오분류가 발생하는지 분석하는 과정이 필요하다."))

    blocks.append(paragraph("2.2) 연구 목적", bold=True))
    blocks.append(paragraph(": 본 연구의 목적은 AI-Hub 피트니스 자세 이미지 데이터셋에서 제공되는 3D 관절 좌표 JSON 데이터를 활용하여 맨몸운동 동작을 분류하고, 여러 분류 모델의 성능과 오분류 패턴을 비교 분석하는 것이다."))
    blocks.extend(numbered([
        "AI-Hub 피트니스 자세 이미지 데이터셋의 관절 좌표 JSON 구조를 분석하고, 운동 동작 분류에 사용할 수 있는 입력 데이터를 구성한다.",
        "맨몸운동 17개 라벨을 대상으로 관절 위치 정보 기반 특징을 추출하고 Train, Validation, Test 데이터를 구성한다.",
        "SVM, XGBoost, GNN, Transformer 모델을 동일한 데이터 분할 조건에서 학습 및 평가한다.",
        "Accuracy, Macro-F1, Weighted-F1 등의 지표를 통해 모델별 분류 성능을 비교한다.",
        "주요 오분류 조합과 자세 그룹별 성능 차이를 분석하여 관절 좌표 기반 운동 동작 분류의 가능성과 한계를 제시한다.",
    ]))

    blocks.append(paragraph("2.3) 연구 동향", bold=True))
    blocks.append(paragraph(": 인간 동작 인식 연구는 크게 센서 데이터 기반 접근, 영상 기반 접근, 관절 위치 정보 기반 접근으로 나눌 수 있다. 센서 데이터 기반 연구는 웨어러블 장치의 가속도, 자이로스코프, IMU 신호를 이용하여 사용자의 활동을 분류한다. 영상 기반 연구는 원본 이미지 또는 영상을 직접 입력으로 사용하여 행동을 인식하지만, 촬영 환경과 개인정보 문제의 영향을 받을 수 있다."))
    blocks.append(paragraph("관절 위치 정보 기반 연구는 사람의 주요 관절 좌표를 추출한 뒤, 이를 이용하여 동작 종류를 분류하거나 자세의 정확성을 평가한다. 이 방식은 원본 영상보다 입력 차원이 낮고, 인체 구조를 명시적으로 반영할 수 있다는 장점이 있다. 관절 좌표를 정형 특징으로 요약하면 SVM이나 XGBoost와 같은 전통적 머신러닝 모델에 적용할 수 있으며, 관절을 노드로 보는 경우 GNN을 통해 신체 연결 구조를 반영할 수 있다. 또한 Transformer는 관절 간 관계를 attention 구조로 학습할 수 있어 관절 위치 정보 기반 분류 모델로 활용할 수 있다."))
    blocks.append(paragraph("기존 연구들이 모델의 전체 정확도에 집중하는 경우가 많았다면, 본 연구는 모델 성능 비교와 함께 오분류 패턴 및 자세 그룹별 분석을 수행한다. 이를 통해 단순히 어떤 모델이 높은 성능을 보였는지뿐 아니라, 어떤 운동군에서 관절 좌표 기반 분류가 어려운지를 함께 확인하고자 한다."))

    blocks.append(paragraph("2.4) 연구 범위 및 한계점", bold=True))
    blocks.append(paragraph(": 본 연구는 운동 자세의 정확성 평가가 아니라 운동 동작 분류(action classification)를 연구 문제로 설정한다. 입력 데이터는 원본 이미지나 영상이 아니라 AI-Hub 피트니스 자세 이미지 데이터셋에서 제공되는 3D 관절 좌표 JSON으로 한정한다."))
    blocks.extend(bullets([
        "분석 대상은 맨몸운동 17개 라벨로 한정한다.",
        "입력 파일은 중복 사용을 방지하기 위해 `-3d.json` 파일만 사용한다.",
        "각 JSON 파일은 하나의 운동 수행 샘플로 간주한다.",
        "모델 비교 대상은 SVM, XGBoost, GNN, Transformer로 설정한다.",
        "실시간 서비스 구현, 모바일 앱 개발, 사용자 인터페이스 구현은 연구 범위에 포함하지 않는다.",
        "관절 좌표 추출 자체의 정확성 평가는 본 연구의 주요 범위가 아니며, 제공된 JSON 관절 좌표를 기반으로 분류 실험을 수행한다.",
    ]))
    blocks.append(paragraph("본 연구의 한계는 다음과 같다. 첫째, 관절 좌표를 프레임별 통계 특징으로 요약하는 과정에서 운동의 세밀한 시간적 순서 정보가 일부 손실될 수 있다. 둘째, 같은 자세군 내부의 유사한 운동은 관절 좌표만으로 구분하기 어려울 수 있다. 셋째, 데이터셋의 촬영 조건과 라벨 구성에 따라 모델 성능이 달라질 수 있으므로, 다른 데이터셋에서 동일한 성능이 보장된다고 단정할 수 없다."))
    blocks.append(paragraph())

    blocks.append(paragraph("3. 연구 방법 및 내용", bold=True))
    blocks.append(paragraph("3.1) 연구 방법론", bold=True))
    blocks.append(paragraph(": 본 연구에서는 AI-Hub 피트니스 자세 이미지 데이터셋의 3D 관절 좌표 JSON을 활용하여 SVM, XGBoost, GNN, Transformer 모델의 운동 동작 분류 성능을 비교한다. 전체 연구 절차는 다음과 같다."))
    blocks.extend(numbered([
        "AI-Hub 피트니스 자세 이미지 데이터셋의 폴더 구조와 라벨 명명 규칙을 분석한다.",
        "맨몸운동 라벨에 해당하는 `-3d.json` 파일을 읽는다.",
        "각 `-3d.json` 파일의 frames 항목에서 24개 관절의 x, y, z 좌표를 추출한다.",
        "프레임 전체를 기준으로 관절별 위치와 움직임의 통계 특징을 계산한다.",
        "JSON 파일 하나를 하나의 운동 수행 샘플로 변환한다.",
        "생성된 샘플을 SVM, XGBoost, GNN, Transformer 모델의 입력 형식에 맞게 구성한다.",
        "촬영 세션 단위로 Train, Validation, Test 데이터를 분할한다.",
        "Validation set을 기준으로 소규모 하이퍼파라미터 튜닝을 수행한다.",
        "최종 설정으로 여러 random seed 반복 실험을 수행하여 결과의 안정성을 확인한다.",
        "성능 지표, 오분류 조합, 자세 그룹별 성능을 분석한다.",
    ]))
    blocks.append(paragraph("본 연구에서 프레임은 관절 좌표를 추출하기 위한 원천 단위이며, 모델 학습과 평가는 프레임 단위가 아니라 JSON 파일 단위의 운동 수행 샘플을 기준으로 수행한다. 즉, 하나의 `-3d.json` 파일 안에 포함된 여러 프레임의 관절 좌표를 독립 샘플로 분리하지 않고, 같은 수행 단위 안에서 요약하여 하나의 입력 샘플로 사용한다."))
    blocks.append(paragraph("3.1.1) 입력 데이터 및 특징 구성", bold=True))
    blocks.append(paragraph("i번째 운동 수행 샘플은 하나의 `-3d.json` 파일로 정의하며, 해당 파일에 포함된 프레임별 관절 좌표 집합을 다음과 같이 나타낸다."))
    blocks.append(paragraph("(1) X_i = {p_{t,j} | t = 1,...,T_i, j = 1,...,J}"))
    blocks.append(paragraph("(2) p_{t,j} = (x_{t,j}, y_{t,j}, z_{t,j})"))
    blocks.append(paragraph("여기서 T_i는 i번째 샘플의 전체 프레임 수, J는 관절 수를 의미한다. 본 연구에서는 24개 관절의 3차원 좌표를 사용하며, 각 프레임의 관절 좌표는 신체 중심과 크기 차이를 줄이기 위해 정규화한 뒤 특징 추출에 사용한다."))
    blocks.append(paragraph("관절 j에 대한 프레임별 좌표열을 기준으로 평균 위치, 표준편차, 최솟값, 최댓값, 움직임 범위, 시작-종료 변화량, 평균 이동량, 최대 이동량을 계산한다. 대표적인 특징 수식은 다음과 같다."))
    blocks.append(paragraph("(3) mean_j = (1 / T_i) sum_{t=1}^{T_i} p_{t,j}"))
    blocks.append(paragraph("(4) std_j = sqrt((1 / T_i) sum_{t=1}^{T_i} ||p_{t,j} - mean_j||^2)"))
    blocks.append(paragraph("(5) range_j = max_t(p_{t,j}) - min_t(p_{t,j})"))
    blocks.append(paragraph("(6) delta_j = p_{T_i,j} - p_{1,j}"))
    blocks.append(paragraph("(7) move_j = (1 / (T_i - 1)) sum_{t=2}^{T_i} ||p_{t,j} - p_{t-1,j}||"))
    blocks.append(paragraph("이 과정을 통해 각 운동 수행 샘플은 관절별 특징 행렬로 표현된다. SVM과 XGBoost에는 관절별 특징 행렬을 1차원 특징 벡터로 펼친 입력을 사용하고, GNN과 Transformer에는 관절을 개별 노드 또는 token으로 유지한 관절별 특징 행렬을 입력으로 사용한다."))
    blocks.append(paragraph("[방법론 그림 첨부 예정]"))

    blocks.append(paragraph("3.1.2) 모델별 구조 및 수식", bold=True))
    blocks.append(paragraph("분석에 사용되는 모델은 SVM, XGBoost, GNN, Transformer이며, 네 모델은 동일한 학습/검증/평가 분할에서 비교한다. SVM과 XGBoost는 정형 특징 벡터 기반 모델이고, GNN과 Transformer는 관절별 특징 행렬에서 관절 간 관계를 학습하는 모델이다."))
    blocks.append(paragraph("SVM은 관절 좌표로부터 생성한 특징 벡터를 입력으로 사용하는 전통적 머신러닝 분류 모델이다. 본 연구에서는 비선형 분류를 위해 RBF kernel을 사용하며, kernel 함수는 다음과 같다."))
    blocks.append(paragraph("(8) K(x_i, x_j) = exp(-gamma ||x_i - x_j||^2)"))
    blocks.append(paragraph("다중 클래스 분류에서는 각 클래스에 대한 결정 함수 값을 계산하고, 가장 큰 결정 함수 값을 갖는 클래스를 최종 예측 라벨로 선택한다. 클래스 불균형의 영향을 줄이기 위해 class_weight를 적용하였다."))
    blocks.append(paragraph("XGBoost는 여러 개의 decision tree를 순차적으로 결합하는 gradient boosting 기반 앙상블 모델이다. K개의 트리로 구성된 모델의 예측값은 다음과 같이 표현할 수 있다."))
    blocks.append(paragraph("(9) y_hat_i = sum_{k=1}^{K} f_k(x_i),  f_k in F"))
    blocks.append(paragraph("XGBoost는 예측 손실과 모델 복잡도 정규화 항을 함께 최소화한다."))
    blocks.append(paragraph("(10) Obj = sum_i l(y_i, y_hat_i) + sum_{k=1}^{K} Omega(f_k)"))
    blocks.append(paragraph("GNN은 관절을 그래프의 노드로, 인체 골격의 연결 관계를 edge로 정의하여 관절 간 구조적 관계를 학습하는 모델이다. 인접행렬 A에 자기 연결 I를 더한 뒤 정규화한 행렬은 다음과 같다."))
    blocks.append(paragraph("(11) A_hat = D^(-1/2)(A + I)D^(-1/2)"))
    blocks.append(paragraph("그래프 합성곱 계층은 이전 계층의 관절 특징 H^(l)에 정규화 인접행렬과 가중치 행렬을 적용하여 다음 계층의 표현을 계산한다."))
    blocks.append(paragraph("(12) H^(l+1) = sigma(A_hat H^(l) W^(l))"))
    blocks.append(paragraph("Transformer는 각 관절을 token처럼 보고 self-attention을 적용하는 모델이다. 관절별 특징은 선형 변환과 관절 embedding을 거쳐 Transformer encoder에 입력되며, attention 연산은 다음과 같다."))
    blocks.append(paragraph("(13) Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V"))
    blocks.append(paragraph("이를 통해 Transformer는 특정 관절 사이의 상호작용을 동적으로 반영할 수 있으며, 최종적으로 관절 token 표현을 평균 pooling한 뒤 분류 계층을 통해 운동 라벨을 예측한다."))

    blocks.append(paragraph("3.1.3) 학습 및 평가 지표", bold=True))
    blocks.append(paragraph("GNN과 Transformer는 다중 클래스 분류 문제로 학습되며, 손실 함수로 Cross-Entropy Loss를 사용한다."))
    blocks.append(paragraph("(14) L = - sum_{c=1}^{C} y_c log(p_c)"))
    blocks.append(paragraph("모델 성능 평가는 Accuracy, Precision, Recall, F1-score, Macro-F1, Weighted-F1을 중심으로 수행한다. 주요 평가 지표는 다음과 같다."))
    blocks.append(paragraph("(15) Accuracy = (TP + TN) / (TP + TN + FP + FN)"))
    blocks.append(paragraph("(16) Precision = TP / (TP + FP)"))
    blocks.append(paragraph("(17) Recall = TP / (TP + FN)"))
    blocks.append(paragraph("(18) F1 = 2 * Precision * Recall / (Precision + Recall)"))
    blocks.append(paragraph("(19) Macro-F1 = (1 / C) sum_{c=1}^{C} F1_c"))
    blocks.append(paragraph("(20) Weighted-F1 = sum_{c=1}^{C} (n_c / N) F1_c"))
    blocks.append(paragraph("여기서 C는 클래스 수, n_c는 클래스 c의 샘플 수, N은 전체 샘플 수를 의미한다. 본 연구에서는 라벨별 데이터 수 차이를 고려하여 Macro-F1을 주요 비교 지표로 사용하며, confusion matrix와 오분류 조합 분석을 통해 어떤 운동 동작 사이에서 혼동이 발생하는지 확인한다."))

    blocks.append(paragraph("3.2) 활용 데이터", bold=True))
    blocks.append(paragraph(": 본 연구에서는 AI-Hub 피트니스 자세 이미지 데이터셋을 활용한다. 해당 데이터셋은 다양한 피트니스 동작에 대한 이미지 및 라벨 정보를 포함하며, 본 연구에서는 이 중 운동 동작 분류에 사용할 수 있는 3D 관절 좌표 JSON 데이터를 대상으로 한다."))
    blocks.append(paragraph("데이터셋 내 JSON 파일은 일반 JSON과 `-3d.json` 파일이 함께 존재할 수 있다. 본 연구에서는 동일한 운동 수행 샘플이 중복으로 사용되는 것을 방지하고, 3차원 관절 좌표 정보를 일관되게 활용하기 위해 `-3d.json` 파일만 사용한다. 각 `-3d.json` 파일은 하나의 운동 수행 샘플로 간주하며, 파일 내부에는 해당 수행 과정에서 추출된 여러 프레임의 24개 관절 x, y, z 좌표가 포함된다."))
    blocks.append(paragraph("따라서 원본 데이터의 내부 구조는 프레임별 관절 좌표 데이터이지만, 본 연구의 실험 데이터 단위는 프레임이 아니라 `-3d.json` 파일 단위이다. 각 파일의 프레임 정보를 종합하여 관절별 통계 특징을 생성한 뒤, 이를 하나의 운동 동작 샘플로 모델에 입력하였다."))
    blocks.append(paragraph("라벨 정보는 데이터셋의 명명 규칙 문서와 source data list 문서를 함께 참조하여 구성하였다. 단순히 폴더명만 보고 라벨을 판단하지 않고, 데이터셋 문서에서 제공하는 serial 정보와 라벨 매핑을 기준으로 운동명을 정리하였다. 이 과정을 통해 맨몸운동 17개 라벨을 최종 실험 대상으로 선정하였다."))
    blocks.append(paragraph("본 연구에서 사용한 맨몸운동 라벨은 다음과 같다. 스탠딩 사이드 크런치, 스탠딩 니업, 버피 테스트, 스텝 포워드 다이나믹 런지, 스텝 백워드 다이나믹 런지, 사이드 런지, 크로스 런지, 굿모닝, 라잉 레그 레이즈, 크런치, 바이시클 크런치, 시저크로스, 힙쓰러스트, 플랭크, 푸시업, 니푸쉬업, Y - Exercise."))
    blocks.append(paragraph("데이터 분할은 단순한 파일 단위 무작위 분할이 아니라 촬영 세션 단위 분할을 사용한다. 같은 촬영 세션의 유사한 샘플이 학습 데이터와 평가 데이터에 동시에 포함될 경우 실제 일반화 성능보다 높은 결과가 나올 수 있기 때문이다. 따라서 Train, Validation, Test set은 서로 다른 촬영 세션을 기준으로 구성한다. Train set은 모델 학습에 사용하고, Validation set은 하이퍼파라미터 선택에 사용하며, Test set은 최종 성능 평가에 사용한다."))
    blocks.append(paragraph("전처리 과정에서 빈 frame을 가진 JSON 파일은 모델 입력으로 사용할 수 없으므로 제외한다. 최종적으로 유효한 `-3d.json` 샘플을 대상으로 관절 좌표 특징을 추출하고, 모든 모델이 동일한 입력 데이터와 동일한 분할 조건에서 비교되도록 구성한다."))

    blocks.append(paragraph("3.3) 실험 결과 및 분석", bold=True))
    blocks.append(paragraph("[실험 결과 표 및 그림 첨부 예정]"))
    blocks.append(paragraph())

    blocks.append(paragraph("4. 결과 요약 및 기대 효과", bold=True))
    blocks.append(paragraph(": 본 연구는 AI-Hub 피트니스 자세 이미지 데이터셋의 3D 관절 좌표를 활용하여 맨몸운동 동작을 분류하고, SVM, XGBoost, GNN, Transformer 모델의 성능을 비교하는 것을 목표로 한다. 17개 맨몸운동 라벨을 대상으로 모델별 분류 성능, 주요 오분류 조합, 자세 그룹별 성능 차이를 함께 분석함으로써 관절 위치 정보 기반 운동 동작 분류의 가능성과 한계를 확인하였다."))
    blocks.append(paragraph("본 연구를 통해 관절 좌표 기반 특징이 운동 동작 분류에 어느 정도 효과적인지 확인할 수 있으며, 전통적인 머신러닝 모델과 관절 관계 기반 딥러닝 모델의 차이를 비교할 수 있다. 또한 오분류 패턴 분석을 통해 모델이 어떤 운동을 혼동하는지 확인함으로써, 향후 운동 동작 분류 시스템에서 추가로 고려해야 할 특징이나 모델 구조를 제안할 수 있다."))
    blocks.append(paragraph("기대 효과는 다음과 같다. 첫째, 원본 영상이 아닌 관절 좌표만으로도 운동 동작 분류가 가능한지 확인할 수 있다. 둘째, 모델별 성능 비교를 통해 관절 좌표 기반 운동 분류에 적합한 모델 선택 기준을 제시할 수 있다. 셋째, 오분류 패턴과 자세 그룹별 분석을 통해 유사 운동군 내부의 세밀한 구분이 왜 어려운지 설명할 수 있다. 넷째, 향후 스마트 피트니스, 홈트레이닝 피드백, 재활 운동 보조 시스템의 기초 연구로 활용될 수 있다."))
    blocks.append(paragraph("다만 본 연구는 실시간 서비스 구현이나 운동 자세의 정확도 평가까지 포함하지 않으므로, 실제 응용 시스템으로 확장하기 위해서는 실시간 추론 속도, 사용자별 신체 차이, 카메라 각도 변화, 잘못된 자세와 올바른 자세의 구분 문제를 추가로 고려해야 한다."))

    blocks.append(paragraph("5. 참고문헌", bold=True))
    blocks.append(paragraph("[1] AI-Hub, 피트니스 자세 이미지 데이터셋, https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=231."))
    blocks.append(paragraph("[2] T. Chen and C. Guestrin, XGBoost: A Scalable Tree Boosting System, Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016."))
    blocks.append(paragraph("[3] C. Cortes and V. Vapnik, Support-Vector Networks, Machine Learning, 1995."))
    blocks.append(paragraph("[4] T. N. Kipf and M. Welling, Semi-Supervised Classification with Graph Convolutional Networks, International Conference on Learning Representations, 2017."))
    blocks.append(paragraph("[5] A. Vaswani et al., Attention Is All You Need, Advances in Neural Information Processing Systems, 2017."))
    blocks.append(paragraph("[6] Z. Cao et al., OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields, IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021."))
    return blocks


def rewrite_docx(source: Path, output: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "working.docx"
        shutil.copy2(source, temp_path)
        document_xml = doc_xml(revised_blocks())
        with ZipFile(temp_path, "r") as zin, ZipFile(output, "w", ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename == "word/document.xml":
                    zout.writestr(item, document_xml.encode("utf-8"))
                else:
                    zout.writestr(item, zin.read(item.filename))


def main() -> None:
    args = parse_args()
    rewrite_docx(args.source, args.output)
    print(f"Completed. Revised thesis draft: {args.output.resolve()}")


if __name__ == "__main__":
    main()
