# CIRSense Python 期末项目

本项目用于完成“Wi-Fi 非接触式感知”期末选题：使用 CIRSense 提供的真实 802.11ax CSI 数据，设计并实现基于 CIR 与深度学习的非接触式感知系统，完成呼吸频率监测、单目标距离估计、NLOS 呼吸鲁棒性测试和多目标距离估计，并对比经典机器学习与深度学习方法。

项目不使用模拟数据。当前数据集应放在：

```text
cirsense_python/
├─ cirsense-dataset-real-world-80211ax-csi-measurements-wireless-sensing/
│  └─ CIRSense_dataset/
│     ├─ breathe/
│     ├─ distance/
│     ├─ multitarget/
│     └─ nlos_breathe/
```

## 环境依赖

首次运行前安装依赖：

```bash
pip install -r requirements.txt
```

依赖安装完成且数据集已经放好后，后续数据预处理、特征提取、模型训练、指标统计和图片生成均可离线运行，不需要联网。

## 项目结构

```text
cirsense_python/
├─ main.py                         # CIRSense 核心任务入口
├─ run_experiments.py              # 一键运行核心任务/学习对比实验
├─ learning_experiments.py         # PCA+SVM/KNN、CNN、融合网络、残差校正等实验
├─ learning_data.py                # 构建机器学习/深度学习样本
├─ models.py                       # PyTorch 深度学习模型
├─ train.py                        # 多任务训练：回归损失 + 分类损失
├─ baselines.py                    # PCA+SVM、PCA+KNN 等经典方法
├─ csi_to_cir.py                   # CSI 到 CIR 的转换
├─ preprocessing.py                # 去噪、归一化、主路径补偿
├─ cirsense_core.py                # Dylign、距离估计、呼吸频率估计
├─ multitarget_distance.py         # 多目标距离峰值检测
├─ feature_extraction.py           # 幅度、相位、频谱、时频图特征
├─ visualize.py                    # 结果可视化
├─ result_audit.py                 # 输出结果审计
├─ report_assets/                  # 已整理的报告用图片
├─ outputs/                        # 正式实验结果
└─ output_test/                    # 临时测试输出
```

## 输出目录规则

- 正式全量实验：默认输出到 `outputs/`。
- 带 `--max-files` 或 `--test-run` 的测试实验：默认输出到 `output_test/`。
- 每次实验使用独立 `--run-name`，便于复现和区分版本。

重要正式结果目录：

```text
outputs/
├─ core_full_final_core_bpm/                 # 呼吸频率核心算法结果
├─ core_full_final_core_distance/            # 单目标距离核心算法结果
├─ core_full_final_core_nlos_bpm/            # NLOS 呼吸结果
├─ core_full_final_core_multitarget_distance/# 多目标距离结果
├─ distance_strict_final/                    # 严格距离分类标准下的最终距离学习结果
├─ deep_stronger_v4/                         # 呼吸任务较优 DNN 集成结果
├─ physics_fusion_no_leak_final/             # 无标签泄漏的物理残差/多视图融合结果
└─ bpm_augmented_final/                      # 呼吸增强预处理消融实验
```

## 核心算法说明

本项目参考 CIRSense 的核心思想，将 CSI 先转换到 CIR 域，再围绕动态路径进行感知：

1. **CSI 到 CIR**：使用子载波投影矩阵将频域 CSI 恢复为 CIR。
2. **去噪与归一化**：对时间序列做滑动平均，利用主静态路径进行复数归一化，减弱硬件相位和幅度扰动。
3. **动态路径对齐**：参考 Dylign 思路，在动态路径附近搜索分数延迟，使目标引起的时间变化更集中。
4. **呼吸频率估计**：从动态路径处聚合 CSI/CIR，得到呼吸波形，经过去趋势、滤波、FFT/STFT 得到 BPM 和时频图。
5. **距离估计**：根据动态路径延迟与 CIR 峰值位置估计目标距离。
6. **多目标距离估计**：对时间平均 CIR profile 与动态方差信息进行多峰检测，输出多个候选目标距离，并生成 `distance_profile.png`。

## 特征与模型

项目满足任务要求中的预处理与特征提取：

- 去噪：滑动平均与频段筛选。
- 归一化：主路径复数归一化。
- 时频转换：FFT 和 STFT。
- 特征：CIR 幅度、CSI/CIR 相位、频谱峰值、时频图、统计特征、多视图特征。

模型包括：

- 经典机器学习：`PCA+SVM`、`PCA+KNN`。
- 深度学习：CNN、CNN-LSTM、增强 DNN 集成。
- 融合模型：多视图融合网络、物理算法 + 深度残差校正。

深度模型采用多任务输出：

```text
Total Loss = SmoothL1Loss(regression) + λ × CrossEntropyLoss(classification)
```

其中回归分支预测连续距离或 BPM，分类分支预测距离类别或呼吸频率区间。

## 运行方式

请先进入项目目录：

```powershell
cd D:\桌面\人工智能与通信工程的前沿实践\Wi-Fi非接触式感知\cirsense_python
```

### 1. 快速测试

```bash
python main.py --task bpm --max-files 3 --run-name quick_bpm_test --save-plots
python main.py --task distance --max-files 3 --run-name quick_distance_test --save-plots
```

### 2. 四个核心任务

```bash
python run_experiments.py --suite core_full --run-name core_full_final --resume
```

也可以分别运行：

```bash
python main.py --task bpm --run-name core_full_final_core_bpm --resume --save-plots --plot-limit 20
python main.py --task distance --run-name core_full_final_core_distance --resume --save-plots --plot-limit 20
python main.py --task bpm --subset nlos_breathe --run-name core_full_final_core_nlos_bpm --resume --save-plots --plot-limit 20
python main.py --task distance --subset multitarget_distance --run-name core_full_final_core_multitarget_distance --resume --save-plots --plot-limit 20
```

### 3. 严格距离分类最终实验

正式报告中的距离 Acc/F1 使用严格距离类别，而不是早期粗分类标准：

```bash
python learning_experiments.py --run-name distance_strict_final --source-run-name full_experiment_final_comparison --tasks distance --methods physics_baseline pca_svm pca_knn dnn_robust_ensemble physics_residual physics_residual_ensemble multiview_residual --epochs 120 --physics-run-name full_experiment_final --distance-class-mode strict
```

严格距离类别包括 11 个真实距离等级，例如 `2.0 m`、`2.25 m`、`3.0 m`、`3.175 m`、`5.0 m`、`10.0 m`、`20.0 m` 等。这样可以避免粗粒度分类导致 Acc/F1 虚高。

### 4. 呼吸频率学习实验

推荐保留两组结果：

```bash
python learning_experiments.py --run-name deep_stronger_v4 --source-run-name full_experiment_final_comparison --tasks bpm distance --methods dnn_robust_ensemble --epochs 120
```

```bash
python learning_experiments.py --run-name physics_fusion_no_leak_final --source-run-name full_experiment_final_comparison --tasks bpm distance --methods physics_baseline physics_residual physics_residual_ensemble multiview_fusion multiview_residual bpm_hybrid_fusion --epochs 120 --physics-run-name full_experiment_final
```

呼吸增强预处理消融实验：

```bash
python learning_experiments.py --run-name bpm_augmented_final --tasks bpm --methods pca_svm pca_knn dnn_robust_ensemble multiview_fusion bpm_dual_fusion physics_residual_ensemble --epochs 120 --rebuild-dataset --bpm-window-augment --bpm-signal-mode quality --physics-run-name full_experiment_final
```

说明：

- `--bpm-signal-mode legacy` 是默认值，使用原始 CIRSense 动态路径呼吸波形。当前报告中较好的 BPM 结果主要来自这一旧流程。
- `--bpm-signal-mode quality` 会在多天线、多动态 CIR tap 中选择呼吸频谱质量更高的候选信号，适合作为消融实验或后续改进方向。
- 如果要复现旧方法效果，重建 BPM 数据集时不要加 `--bpm-signal-mode quality`。

## 输出文件说明

每个学习方法目录通常包含：

```text
metrics.json              # 单方法指标
predictions.csv           # 测试集真实值、预测值、误差、分类结果
prediction_scatter.png    # 真实值-预测值散点图
confusion_matrix.csv      # 混淆矩阵数据
confusion_matrix.png      # 混淆矩阵图片
train_loss.png            # 深度学习训练曲线
```

汇总指标在：

```text
outputs/<run-name>/learning/metrics_summary.csv
```

审计输出：

```bash
python result_audit.py --run-name core_full_final
python result_audit.py --run-name distance_strict_final
python result_audit.py --run-name deep_stronger_v4
```

## 当前推荐报告结果

距离任务建议使用 `distance_strict_final`：

| 方法 | MAE / m | Acc | Macro-F1 |
| --- | ---: | ---: | ---: |
| PCA+SVM | 0.192 | 1.000 | 1.000 |
| PCA+KNN | 0.170 | 0.889 | 0.863 |
| DNN robust ensemble | 0.086 | 1.000 | 1.000 |
| Multiview residual | 0.024 | 0.981 | 0.988 |
| Physics residual ensemble | 0.022 | 0.981 | 0.988 |

呼吸任务建议综合使用：

| 方法 | MAE / bpm | Acc | Macro-F1 |
| --- | ---: | ---: | ---: |
| DNN robust ensemble | 0.442 | 0.825 | 0.838 |
| Multiview fusion | 0.521 | 0.850 | 0.875 |
| Augmented multiview | 0.526 | 0.900 | 0.699 |

报告主结论建议：

- 距离估计：`physics_residual_ensemble` 连续误差最低，MAE 为 `0.022 m`。
- 呼吸监测：`DNN robust ensemble` 的 BPM 回归误差最低，MAE 为 `0.442 bpm`；`multiview_fusion` 的 Macro-F1 更高，为 `0.875`。
- 呼吸增强预处理提高了 Accuracy，但 Macro-F1 和 MAE 没有同步改善，可作为局限性和消融讨论。

## 报告图片

报告用图片已整理到：

```text
report_assets/
├─ fig1_system_pipeline.png
├─ fig2_bpm_feature_examples.png
├─ fig3_multitarget_distance_profile.png
├─ fig4_distance_method_comparison.png
├─ fig5_bpm_method_comparison.png
└─ fig6_confusion_matrices.png
```

这些图片来自正式输出或由正式指标汇总生成，适合放入 6-10 页课程报告。由于当前 Matplotlib 环境缺少中文字体，组合图中文字采用英文；Word 报告中可使用中文图题和正文解释。

## 打包建议

提交给老师时建议保留：

- 所有 `.py` 文件；
- `README.md`；
- `requirements.txt`；
- `report_assets/`；
- `outputs/distance_strict_final/`；
- `outputs/deep_stronger_v4/`；
- `outputs/physics_fusion_no_leak_final/`；
- `outputs/core_full_final_core_bpm/`、`outputs/core_full_final_core_distance/`、`outputs/core_full_final_core_nlos_bpm/`、`outputs/core_full_final_core_multitarget_distance/` 中的结果 CSV 和少量图片。

如果数据集体积过大，可按课程要求单独说明数据集来源与本地路径，不建议把全部原始数据和全部预处理缓存都打包进 Word 附件。
