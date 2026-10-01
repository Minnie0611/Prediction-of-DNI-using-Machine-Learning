"""Recompute published metrics and check saved state without retraining."""
import json
from pathlib import Path

import joblib
import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'incremental_outputs'
summary = json.loads((OUTPUT / 'run_summary.json').read_text(encoding='utf-8'))
nb = nbformat.read(ROOT / 'dni_incremental_learning_framework.ipynb', as_version=4)
errors = [out for cell in nb.cells if cell.cell_type == 'code'
          for out in cell.get('outputs', []) if out.output_type == 'error']
assert not errors, errors
assert all(cell.execution_count is not None for cell in nb.cells if cell.cell_type == 'code')
pred = pd.read_csv(OUTPUT / 'final_test_predictions.csv', parse_dates=['timestamp'])
results = pd.read_csv(OUTPUT / 'final_test_comparison.csv')
monthly = pd.read_csv(OUTPUT / 'monthly_incremental_report.csv')
monthly_pred = pd.read_csv(OUTPUT / 'monthly_preupdate_predictions.csv', parse_dates=['timestamp'])
assert pred['timestamp'].dt.year.eq(2023).all()
assert monthly_pred['timestamp'].dt.year.eq(2022).all()
assert pred['timestamp'].is_unique and monthly_pred['timestamp'].is_unique
assert set(monthly.month) == {f'2022-{i:02d}' for i in range(1, 13)}
assert monthly.evaluation_phase.eq('before_update').all()


def metrics(y, p):
    mse = mean_squared_error(y, p)
    return {'R2': r2_score(y, p), 'MAE': mean_absolute_error(y, p),
            'MSE': mse, 'RMSE': float(np.sqrt(mse))}


for _, row in results.iterrows():
    subset = pred if row.condition == 'overall' else pred[pred.sky_condition == row.condition]
    column = 'Base Predicted DNI' if row.model == 'base' else 'Predicted DNI'
    assert len(subset) == row.n_samples
    for key, value in metrics(subset.DNI, subset[column]).items():
        np.testing.assert_allclose(row[key], value, rtol=1e-10)

for _, row in monthly.iterrows():
    subset = monthly_pred[monthly_pred.timestamp.dt.strftime('%Y-%m') == row.month]
    if row.condition != 'overall':
        subset = subset[subset.sky_condition == row.condition]
    assert len(subset) == row.n_month_samples
    for column, prefix in [('Predicted DNI', ''), ('Base Predicted DNI', 'base_')]:
        for key, value in metrics(subset.DNI, subset[column]).items():
            np.testing.assert_allclose(row[prefix + key], value, rtol=1e-10)

for column in ['Predicted DNI', 'Base Predicted DNI']:
    assert np.isfinite(pred[column]).all()
    assert (pred[column] >= 0).all()
    assert (pred[column] <= pred.GHI / np.maximum(pred['Solar Cos Zenith Angle'], 1e-6) + 1e-5).all()

# Unpickle the notebook's __main__.ReplayBuffer using its saved class definition.
REPLAY_CAPACITY_PER_CONDITION = summary['replay_capacity']
REPLAY_SAMPLE_SIZE_PER_UPDATE = summary['replay_sample_size']
SEED = summary['seed']
exec(nb.cells[13].source, globals())
for condition in ['overcast', 'partly_cloudy', 'clear_sky']:
    base_scaler = joblib.load(OUTPUT / f'base_model_{condition}.scaler.joblib')
    for scaler_path in OUTPUT.glob(f'incremental_{condition}_*.scaler.joblib'):
        scaler = joblib.load(scaler_path)
        np.testing.assert_array_equal(scaler.mean_, base_scaler.mean_)
        np.testing.assert_array_equal(scaler.scale_, base_scaler.scale_)
    for state_path in OUTPUT.glob(f'*_{condition}*.state.json'):
        state = json.loads(state_path.read_text(encoding='utf-8'))
        assert pd.Timestamp(state['trained_through']) < pd.Timestamp('2023-01-01')
    buffer = joblib.load(OUTPUT / f'incremental_{condition}_202212.buffer.joblib')
    assert buffer.data.timestamp.max() < pd.Timestamp('2023-01-01')
    assert buffer.data.timestamp.is_unique and len(buffer) <= REPLAY_CAPACITY_PER_CONDITION

print('PASS: all notebook code cells executed without errors.')
print('PASS: independent-test and monthly metrics match saved per-sample predictions.')
print('PASS: time boundaries, fixed scalers, replay state and physical prediction bounds.')
print(results.to_string(index=False))

overall = results[results.condition == 'overall'].set_index('model')
base, inc = overall.loc['base'], overall.loc['incremental']
mae_improvement = (base.MAE - inc.MAE) / base.MAE * 100
rmse_improvement = (base.RMSE - inc.RMSE) / base.RMSE * 100
lines = [
    '# DNI 增量学习实验结果', '',
    '已完成完整 Notebook 运行；以下为实际结果，未根据 2023 年测试结果重新调参。', '',
    '## 实验设置', '',
    '- 基础训练：2011—2021 年，96,337 条白天记录。',
    '- 增量更新：2022 年，8,759 条白天记录；逐月先评估再更新，共 12 个月。',
    '- 独立测试：2023 年，8,759 条白天记录，从未参与训练、标准化拟合或回放。',
    '- 每类天气一个网络，固定基础 scaler；基础学习率 0.001，微调学习率 0.0001。',
    '- 基础训练最多 100 轮；每月更新最多 15 轮；使用验证集早停及最佳 checkpoint。',
    '- 每类回放容量 12,000 条，每次最多抽取 3,000 条；随机种子 42。',
    f'- 训练与评估耗时：{summary["elapsed_seconds"] / 60:.2f} 分钟（不含数据下载）。', '',
    '## 2023 年独立测试', '',
    '| 模型 | 天气类别 | 样本数 | R² | MAE (W/m²) | RMSE (W/m²) | MSE |',
    '|---|---|---:|---:|---:|---:|---:|',
]
labels = {'base': '基础模型', 'incremental': '增量模型', 'overcast': '阴天',
          'partly_cloudy': '局部多云', 'clear_sky': '晴天', 'overall': '总体'}
for _, row in results.iterrows():
    lines.append(f'| {labels[row.model]} | {labels[row.condition]} | {row.n_samples} | '
                 f'{row.R2:.4f} | {row.MAE:.2f} | {row.RMSE:.2f} | {row.MSE:.2f} |')
lines += ['', f'总体 MAE 相对改善 {mae_improvement:.2f}%，RMSE 相对改善 {rmse_improvement:.2f}%（负数表示变差）。',
          f'总体 R² 从 {base.R2:.4f} 变为 {inc.R2:.4f}。', '',
          '## 验证与解释限制', '',
          '- 已从逐样本 CSV 独立重算所有最终和月度指标，结果一致。',
          '- 已核验 2023 年不在月度训练、最终回放或训练状态中；所有更新 scaler 与基础 scaler 一致。',
          '- 预测结果满足非负及原有物理上限约束。',
          '- 结果只针对当前数据的白天样本、单次随机种子实验；不是多站点或多次重复实验结论。',
          '- 太阳天顶角保留原始简化计算，未纳入经度、时区和均时差修正。',
          '- 每月模型输入是当月各时刻的气象特征；这不是提前预测未来天气。',
          '- 对比反映新月份数据、回放和微调参数的整体效果，不能单独归因于某一组件。', '',
          '## 文件', '',
          '- `final_test_comparison.csv`：完整测试指标。',
          '- `final_test_predictions.csv`：2023 年真实值、两组预测和特征。',
          '- `monthly_incremental_report.csv`：2022 年更新前的月度指标及固定基础模型对照。',
          '- `monthly_preupdate_predictions.csv`：2022 年更新前逐样本预测。',
          '- `data_split_report.csv`：时间划分及样本数量。',
          '- `run_summary.json`：运行参数与指标。',
          '- `.keras`、`.scaler.joblib`、`.buffer.joblib`、`.state.json`：匹配的模型和状态。', '']
(OUTPUT / '实验结果说明.md').write_text('\n'.join(lines), encoding='utf-8')
print(f'MAE improvement: {mae_improvement:.2f}%; RMSE improvement: {rmse_improvement:.2f}%')
