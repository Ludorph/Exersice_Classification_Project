# 관절 위치 정보 기반 운동 동작 분류 모델 비교

이 프로젝트는 `Physical Exercise Recognition Time Series Dataset`을 사용해 `push_up`, `pull_up`, `squat` 세 동작을 분류하고, XGBoost, LSTM, ST-GCN의 성능을 동일한 데이터 분할에서 비교합니다.

## 설치

```powershell
pip install -r requirements.txt
```

## 전체 파이프라인 실행

```powershell
python src/train_pipeline.py --data-dir "C:\Users\Ludorph\Downloads\Physical Exercise RecognitionTime Series Dataset"
```

기본 실행 결과는 `outputs/`에 저장됩니다. 학습이 끝나면 모델별 성능 비교 및 오분류 패턴 분석도 자동으로 실행됩니다.

- `metrics_summary.csv`: 모델별 accuracy, macro/weighted precision, recall, f1
- `classification_report_*.csv`: 모델별 동작별 precision, recall, f1-score
- `confusion_matrix_*.png`: 모델별 confusion matrix 이미지
- `results.json`: 전체 평가 결과
- `test_predictions_*.csv`: test set의 비디오별 실제 라벨, 예측 라벨, 정오분류 여부
- `xgboost_feature_importance.csv`: XGBoost가 크게 사용한 집계 특징 순위
- `model_comparison_overall.png`: Accuracy, macro precision/recall/F1 비교 그래프
- `model_comparison_per_class_f1.png`: 동작별 F1-score 비교 그래프
- `misclassification_pairs.csv`: 실제 라벨과 예측 라벨 조합별 오분류 횟수
- `misclassification_pattern_*.png`: 모델별 오분류 방향 heatmap
- `hard_cases_by_video.csv`: 두 개 이상의 모델이 동시에 틀린 비디오 목록
- `analysis_summary.md`: 논문 결과 분석에 옮겨 쓸 수 있는 요약 문서
- `checkpoints/`: LSTM, ST-GCN 모델 가중치

## 분석만 다시 실행

학습 결과 파일이 이미 있는 경우에는 분석 코드만 다시 실행할 수 있습니다.

```powershell
python src/analyze_results.py --output-dir outputs
```

## 연구 파이프라인

1. `labels.csv`에서 `pull_up`, `push_up`, `squat` 라벨만 사용합니다.
2. `vid_id` 단위로 train/validation/test를 stratified split합니다.
3. XGBoost는 `angles.csv`, `calculated_3d_distances.csv`, `xyz_distances.csv`의 프레임별 특징을 비디오 단위 통계량으로 집계합니다.
4. LSTM은 `landmarks.csv`의 관절 좌표 시계열을 고정 길이로 리샘플링해 입력합니다.
5. ST-GCN은 MediaPipe 33개 관절을 그래프 노드로 보고, 관절 연결 구조와 시간 축을 함께 학습합니다.
6. 모든 모델을 같은 test set에서 평가해 성능표와 confusion matrix를 생성합니다.
