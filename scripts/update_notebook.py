"""Apply the chronological, leakage-free experiment to the original notebook."""
import ast
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'dni_incremental_learning_framework.ipynb'
nb = json.loads((ROOT / 'dni_incremental_learning_framework.original.ipynb').read_text(encoding='utf-8'))


def replace(index, source):
    nb['cells'][index]['source'] = textwrap.dedent(source).strip('\n').splitlines(keepends=True)


replace(0, '''
# DNI 增量学习：按时间划分、先评估再更新

- **2011—2021 年**：训练三个天空状况对应的基础模型。
- **2022 年**：每个月先用上月底模型预测整月并记录指标，然后用该月标签和历史回放数据更新。
- **2023 年**：完全独立的测试集；只在全部 2022 年更新结束后评估。
- 基础模型与更新后模型在同一个 2023 年测试集上比较，不使用测试指标选择参数。
- 保留原有 7 个特征、网络结构、Kt 分类阈值和预测后物理裁剪。
- 固定基础训练 scaler；保存模型、scaler、buffer 和运行报告。

这里估计的是同一时刻的 DNI，并非提前预测未来天气。月度预测在得到当月标签之前完成，
当月标签随后才参与训练。太阳天顶角仍沿用原 Notebook 的简化公式，结果有此近似限制。
''')
source = ''.join(nb['cells'][1]['source'])
source = source.replace('import os\n', 'import os\nos.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")\n')
source = source.replace('from dataclasses import dataclass', 'import joblib\nimport time\nfrom IPython.display import display')
source = source.replace('from keras import layers', 'from tensorflow.keras import layers')
source += '\n# Small dense models run efficiently with a bounded CPU thread pool.\ntf.config.threading.set_intra_op_parallelism_threads(4)\ntf.config.threading.set_inter_op_parallelism_threads(2)\n'
replace(1, source)

tree = ast.parse(''.join(nb['cells'][3]['source']))
urls = {int(node.targets[0].id[4:]): node.value.value for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith('path')}
replace(2, '## 1. 数据与实验配置\n\n首次运行下载原始数据；后续优先使用本地 `data_cache`。')
replace(3, 'DATA_URLS = ' + repr(dict(sorted(urls.items()))) + '''

DATA_CACHE = Path("data_cache")
DATA_CACHE.mkdir(exist_ok=True)
files = []
for year, url in DATA_URLS.items():
    cache_path = DATA_CACHE / f"nsrdb_{year}.csv"
    if cache_path.exists():
        yearly_df = pd.read_csv(cache_path)
    else:
        yearly_df = pd.read_csv(url, skiprows=2)
        yearly_df.to_csv(cache_path, index=False)
    # Published sheets contain empty trailing columns; these are not features.
    yearly_df = yearly_df.loc[:, ~yearly_df.columns.str.startswith("Unnamed:")]
    if not (pd.to_numeric(yearly_df["Year"]) == year).all():
        raise ValueError(f"Unexpected year in {cache_path}")
    files.append(yearly_df)
    print(f"Loaded {year}: {len(yearly_df):,} rows")
''')
replace(4, '''
all_files = pd.concat(files, ignore_index=True)
print(f"Combined shape: {all_files.shape}")
display(all_files.head())
''')
source = ''.join(nb['cells'][5]['source'])
source = source.replace('# DATA_PATH = "YOUR_DATA.csv"              # 改成你的总数据文件\n', '')
source = source.replace('OUTPUT_DIR = Path("incremental_outputs")', 'OUTPUT_DIR = Path("incremental_outputs")')
source = source.replace('TEST_START = "2023-01-01 00:00:00"', 'TEST_START = "2023-01-01 00:00:00"\nTEST_END_EXCLUSIVE = "2024-01-01 00:00:00"\nassert pd.Timestamp(BASE_TRAIN_END) < pd.Timestamp(INCREMENTAL_START) < pd.Timestamp(TEST_START) < pd.Timestamp(TEST_END_EXCLUSIVE)')
source += '\n# Fine-tuning uses a smaller learning rate and the fixed base scaler.\nINCREMENTAL_LEARNING_RATE = 0.0001\n'
replace(5, source)

source = ''.join(nb['cells'][9]['source'])
source = source.replace('df["timestamp"] = pd.to_datetime(df[TIME_COLS])', '''df["timestamp"] = pd.to_datetime(df[TIME_COLS])
    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps in source data")''')
source = source.replace('    # 基础时间特征', '''    required = RAW_FEATURE_COLS + [TARGET_COL]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df[required] = df[required].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(df[required].to_numpy()).all():
        raise ValueError("Missing or non-finite model input/target values")
    if (df[["GHI", "DNI"]] < 0).any().any():
        raise ValueError("Negative irradiance values require source-data review")

    # 基础时间特征''')
replace(9, source)

source = ''.join(nb['cells'][11]['source'])
source = source.replace('def fit_model(', '''class EpochProgress(keras.callbacks.Callback):
    def __init__(self, model_path):
        super().__init__()
        self.checkpoint = Path(model_path).name

    def on_epoch_end(self, epoch, logs=None):
        record = {"checkpoint": self.checkpoint, "epoch": epoch + 1,
                  "time": pd.Timestamp.now().isoformat(),
                  **{k: float(v) for k, v in (logs or {}).items()}}
        with (OUTPUT_DIR / "training_progress.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\\n")


def fit_model(''')
source = source.replace('    callbacks = [', '    callbacks = [\n        EpochProgress(model_path),')
source = source.replace('verbose=1', 'verbose=0')
source = source.replace('    return history', '    pd.DataFrame(history.history).to_csv(Path(model_path).with_suffix(".history.csv"), index=False)\n    return history')
replace(11, source)
replace(12, '''
## 5. 历史样本回放

每个天空状况最多保留 12,000 条历史数据，更新时最多抽取 3,000 条。
初始化按 DNI 分位数分桶抽样；后续超容量时一半留给最近数据，另一半从更早数据分桶抽样。
抽样不放回，使用原行索引排除已选样本，防止补样重复。任何未来数据都不能进入 buffer。
''')
replace(13, '''
class ReplayBuffer:
    def __init__(self, capacity=REPLAY_CAPACITY_PER_CONDITION, random_state=SEED):
        self.capacity = capacity
        self.rng = np.random.RandomState(random_state)
        self.data = pd.DataFrame()

    def __len__(self):
        return len(self.data)

    def update(self, new_df, bucket_col="DNI", n_buckets=5):
        if new_df.empty:
            return
        new_df = new_df.drop(columns=["year_month"], errors="ignore")
        combined = pd.concat([self.data, new_df], ignore_index=True)
        combined = combined.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
        if len(combined) > self.capacity:
            if self.data.empty:
                combined = self._bucket_sample(combined, bucket_col, self.capacity, n_buckets)
            else:
                recent_n = self.capacity // 2
                recent = combined.tail(recent_n) if recent_n else combined.iloc[:0]
                older = combined.drop(index=recent.index)
                representative = self._bucket_sample(older, bucket_col, self.capacity - recent_n, n_buckets)
                combined = pd.concat([recent, representative], ignore_index=True)
        self.data = combined.sort_values("timestamp").reset_index(drop=True)
        assert len(self.data) <= self.capacity
        assert self.data["timestamp"].is_unique

    def sample(self, n=REPLAY_SAMPLE_SIZE_PER_UPDATE, bucket_col="DNI", n_buckets=5):
        if self.data.empty:
            return self.data.copy()
        return self._bucket_sample(self.data, bucket_col, min(n, len(self.data)), n_buckets).reset_index(drop=True)

    def _bucket_sample(self, df, bucket_col="DNI", total_n=1000, n_buckets=5):
        work = df.reset_index(drop=True).copy()
        total_n = min(total_n, len(work))
        if len(work) <= total_n or total_n == 0:
            return work.iloc[:total_n].copy()
        buckets = pd.qcut(work[bucket_col], q=n_buckets, duplicates="drop")
        groups = list(work.groupby(buckets, observed=True))
        if not groups or total_n < len(groups):
            return work.sample(total_n, random_state=self.rng)
        per_bucket = total_n // len(groups)
        selected = pd.concat([g.sample(min(len(g), per_bucket), random_state=self.rng)
                              for _, g in groups])  # Keep original indices until補樣 completes.
        remaining = work.drop(index=selected.index)
        if len(selected) < total_n:
            selected = pd.concat([selected, remaining.sample(total_n - len(selected), random_state=self.rng)])
        assert selected.index.is_unique
        return selected.reset_index(drop=True)
''')
source = ''.join(nb['cells'][15]['source'])
source = source.replace('"R2": r2_score(y_true, y_pred)', '"R2": r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan')
source = source.replace('"MSE": mean_squared_error(y_true, y_pred),', '"MSE": mean_squared_error(y_true, y_pred),\n        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),')
source += '''

def save_artifact_state(cond, artifact, buffer):
    # Model/scaler/buffer carry matching checkpoint names for restartability.
    stem = Path(artifact["last_checkpoint"]).with_suffix("")
    joblib.dump(artifact["scaler"], str(stem) + ".scaler.joblib")
    joblib.dump(buffer, str(stem) + ".buffer.joblib")
    metadata = {"condition": cond, "trained_through": artifact["trained_through"],
                "checkpoint": str(artifact["last_checkpoint"]),
                "features": MODEL_FEATURE_COLS, "scaler_policy": "fixed_base_scaler"}
    Path(str(stem) + ".state.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
'''
replace(15, source)
source = ''.join(nb['cells'][17]['source'])
source = source.replace('"last_checkpoint": model_path', '"last_checkpoint": model_path,\n            "trained_through": str(cond_df["timestamp"].max())')
source = source.replace('        replay_buffers[cond] = buffer', '        replay_buffers[cond] = buffer\n        save_artifact_state(cond, artifacts[cond], buffer)')
source = source.replace('    return artifacts, replay_buffers', '    if set(artifacts) != set(CONDITIONS):\n        raise ValueError("Base data must contain all three sky conditions")\n    return artifacts, replay_buffers')
replace(17, source)
replace(18, '## 8. 月度切块\n\n仅生成 `[2022-01-01, 2023-01-01)` 的数据块，测试集不进入更新循环。')
source = ''.join(nb['cells'][19]['source'])
source = source.replace('df_daytime[df_daytime["timestamp"] >= pd.Timestamp(INCREMENTAL_START)]', 'df_daytime[(df_daytime["timestamp"] >= pd.Timestamp(INCREMENTAL_START)) &\n                         (df_daytime["timestamp"] < pd.Timestamp(TEST_START))]')
replace(19, source)
replace(20, '''
## 9. 月度循环：先评估，后更新

每月先使用上月底模型预测全部当月样本并记录指标，同时记录固定基础模型的对照指标。
完成预测后才读取当月 DNI 标签用于训练。更新时混合该月数据与历史回放数据，随机留出 20% 做验证，
固定基础 scaler，学习率降为 0.0001，最多微调 15 轮。
月度报告只包含更新前指标；不会把当月训练后的拟合分数当作泛化成绩。
''')
replace(21, '''
def incremental_update_for_condition(cond, monthly_df, artifacts, replay_buffers,
                                     epochs=INCREMENTAL_EPOCHS):
    cond_month = monthly_df[monthly_df["sky_condition"] == cond].copy().reset_index(drop=True)
    if cond_month.empty:
        return None
    if cond not in artifacts:
        raise ValueError(f"Missing base model: {cond}")
    assert cond_month["timestamp"].min() >= pd.Timestamp(INCREMENTAL_START)
    assert cond_month["timestamp"].max() < pd.Timestamp(TEST_START)
    artifact = artifacts[cond]
    assert pd.Timestamp(artifact["trained_through"]) < monthly_df["timestamp"].min()
    replay_df = replay_buffers[cond].sample(n=REPLAY_SAMPLE_SIZE_PER_UPDATE)
    if not replay_df.empty:
        assert replay_df["timestamp"].max() < monthly_df["timestamp"].min()
    train_df = pd.concat([cond_month, replay_df], ignore_index=True).drop_duplicates("timestamp")
    X, y = make_xy(train_df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # Keep the coordinate system expected by the previously learned weights.
    scaler = artifact["scaler"]
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    model = keras.models.load_model(artifact["last_checkpoint"])
    model.optimizer.learning_rate.assign(INCREMENTAL_LEARNING_RATE)
    month = monthly_df["timestamp"].dt.strftime("%Y%m").iloc[0]
    checkpoint = OUTPUT_DIR / f"incremental_{cond}_{month}.keras"
    history = fit_model(model, X_train_scaled, y_train, X_val_scaled, y_val,
                        model_path=checkpoint, epochs=epochs)
    artifact.update(model=keras.models.load_model(checkpoint), history=history,
                    last_checkpoint=checkpoint, trained_through=str(cond_month["timestamp"].max()))
    replay_buffers[cond].update(cond_month)
    save_artifact_state(cond, artifact, replay_buffers[cond])
    return {"condition": cond, "n_train": len(X_train), "n_validation": len(X_val),
            "epochs": len(history.history["loss"])}


def run_incremental_learning(df_daytime, artifacts, replay_buffers, base_artifacts):
    monthly_reports, monthly_predictions, training_reports = [], [], []
    for month, block in monthly_blocks(df_daytime):
        print(f"Processing {month}: predict first, then update", flush=True)
        for cond in CONDITIONS:
            assert pd.Timestamp(artifacts[cond]["trained_through"]) < block["timestamp"].min()
        # Predictions are computed BEFORE using any labels from this month for updating.
        pred = predict_all_conditions(block, artifacts)
        base_pred = predict_all_conditions(block, base_artifacts)
        for cond in CONDITIONS + ["overall"]:
            mask = np.ones(len(block), dtype=bool) if cond == "overall" else (block["sky_condition"] == cond).to_numpy()
            if not mask.any():
                continue
            y = block.loc[mask, TARGET_COL].to_numpy()
            monthly_reports.append({"month": month, "condition": cond,
                "evaluation_phase": "before_update", "n_month_samples": int(mask.sum()),
                **evaluate_regression(y, pred[mask]),
                **{f"base_{k}": v for k, v in evaluate_regression(y, base_pred[mask]).items()}})
        month_pred = block[["timestamp", "sky_condition", TARGET_COL]].copy()
        month_pred["Predicted DNI"] = pred
        month_pred["Base Predicted DNI"] = base_pred
        monthly_predictions.append(month_pred)
        # Save honest scores before training begins, so progress is inspectable.
        pd.DataFrame(monthly_reports).to_csv(OUTPUT_DIR / "monthly_incremental_report.csv", index=False)
        pd.concat(monthly_predictions, ignore_index=True).to_csv(OUTPUT_DIR / "monthly_preupdate_predictions.csv", index=False)
        for cond in CONDITIONS:
            result = incremental_update_for_condition(cond, block, artifacts, replay_buffers)
            if result is not None:
                training_reports.append({"month": month, **result})
        pd.DataFrame(training_reports).to_csv(OUTPUT_DIR / "monthly_training_report.csv", index=False)
    return pd.DataFrame(monthly_reports), artifacts, replay_buffers
''')
source = ''.join(nb['cells'][23]['source'])
source = source.replace('        if mask.sum() == 0 or cond not in artifacts:\n            continue', '        if mask.sum() == 0:\n            continue\n        if cond not in artifacts:\n            raise ValueError(f"Missing model for {cond}")')
source = source.replace('    preds = np.zeros(len(df))', '    if not df["sky_condition"].isin(CONDITIONS).all():\n        raise ValueError("Unknown sky condition")\n    preds = np.zeros(len(df))')
replace(23, source)
replace(24, '## 11. 完整实验\n\n从头运行时重训基础模型；保留固定基础模型作为 2023 年独立测试的对照。')
replace(25, '''
started = time.perf_counter()
(OUTPUT_DIR / "training_progress.jsonl").write_text("", encoding="utf-8")
df_raw = all_files.copy()
df_all = prepare_features(df_raw)
df_day = daytime_only(df_all)
base_df = df_day[df_day["timestamp"] <= pd.Timestamp(BASE_TRAIN_END)].copy()
incremental_df = df_day[(df_day["timestamp"] >= pd.Timestamp(INCREMENTAL_START)) &
                        (df_day["timestamp"] < pd.Timestamp(TEST_START))].copy()
test_df = df_day[(df_day["timestamp"] >= pd.Timestamp(TEST_START)) &
                 (df_day["timestamp"] < pd.Timestamp(TEST_END_EXCLUSIVE))].copy()
if any(part.empty for part in [base_df, incremental_df, test_df]):
    raise ValueError("Base, incremental and test periods must all contain data")
assert base_df["timestamp"].max() < incremental_df["timestamp"].min()
assert incremental_df["timestamp"].max() < test_df["timestamp"].min()

split_rows = []
for name, part in [("base_2011_2021", base_df), ("incremental_2022", incremental_df), ("test_2023", test_df)]:
    for cond in CONDITIONS + ["overall"]:
        subset = part if cond == "overall" else part[part["sky_condition"] == cond]
        split_rows.append({"split": name, "condition": cond, "n_samples": len(subset),
                           "start": str(subset["timestamp"].min()), "end": str(subset["timestamp"].max())})
split_report_df = pd.DataFrame(split_rows)
split_report_df.to_csv(OUTPUT_DIR / "data_split_report.csv", index=False)
print(f"Raw rows: {len(df_raw):,}; daytime rows: {len(df_day):,}")
display(split_report_df)

artifacts, replay_buffers = train_base_models(base_df)
# Separate model instances: loading the next checkpoint never changes these baselines.
base_artifacts = {cond: {"scaler": item["scaler"],
                         "model": keras.models.load_model(item["last_checkpoint"]),
                         "last_checkpoint": item["last_checkpoint"],
                         "trained_through": item["trained_through"]}
                  for cond, item in artifacts.items()}
monthly_report_df, artifacts, replay_buffers = run_incremental_learning(
    incremental_df, artifacts, replay_buffers, base_artifacts)
for cond in CONDITIONS:
    assert pd.Timestamp(artifacts[cond]["trained_through"]) < pd.Timestamp(TEST_START)
    assert replay_buffers[cond].data["timestamp"].max() < pd.Timestamp(TEST_START)
    np.testing.assert_array_equal(artifacts[cond]["scaler"].mean_, base_artifacts[cond]["scaler"].mean_)

# Only now evaluate 2023. These results do not control training or parameter selection.
comparison_rows = []
for label, models, prediction_col in [("base", base_artifacts, "Base Predicted DNI"),
                                      ("incremental", artifacts, "Predicted DNI")]:
    pred = predict_all_conditions(test_df, models)
    test_df[prediction_col] = pred
    for cond in CONDITIONS + ["overall"]:
        mask = np.ones(len(test_df), dtype=bool) if cond == "overall" else (test_df["sky_condition"] == cond).to_numpy()
        if mask.any():
            comparison_rows.append({"model": label, "condition": cond, "n_samples": int(mask.sum()),
                **evaluate_regression(test_df.loc[mask, TARGET_COL].to_numpy(), pred[mask])})
comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(OUTPUT_DIR / "final_test_comparison.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "final_test_predictions.csv", index=False)
test_metrics = comparison_df[(comparison_df["model"] == "incremental") & (comparison_df["condition"] == "overall")].iloc[0].to_dict()
summary = {"seed": SEED, "base_train_end": BASE_TRAIN_END,
           "incremental_start": INCREMENTAL_START, "incremental_end_exclusive": TEST_START,
           "test_start": TEST_START, "test_end_exclusive": TEST_END_EXCLUSIVE,
           "monthly_evaluation": "before_update", "fixed_base_scaler": True,
           "solar_zenith": "original_approximation", "daytime_only": True,
           "base_learning_rate": 0.001, "incremental_learning_rate": INCREMENTAL_LEARNING_RATE,
           "base_max_epochs": BASE_EPOCHS, "incremental_max_epochs": INCREMENTAL_EPOCHS,
           "replay_capacity": REPLAY_CAPACITY_PER_CONDITION,
           "replay_sample_size": REPLAY_SAMPLE_SIZE_PER_UPDATE,
           "raw_rows": len(df_raw), "daytime_rows": len(df_day),
           "tensorflow_version": tf.__version__, "elapsed_seconds": time.perf_counter() - started,
           "test_results": comparison_df.to_dict(orient="records")}
(OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("2023 independent test: fixed base vs. end-of-2022 incremental models")
display(comparison_df)
print(f"Elapsed: {summary['elapsed_seconds'] / 60:.1f} minutes")
''')
replace(26, '## 12. 结果图\n\n月度 MAE 使用更新前预测；2023 年预测曲线来自完全独立的测试集。')
replace(27, '''
if not monthly_report_df.empty:
    fig, ax = plt.subplots(figsize=(12, 5))
    for cond in CONDITIONS:
        temp = monthly_report_df[monthly_report_df["condition"] == cond]
        ax.plot(temp["month"], temp["MAE"], marker="o", label=cond)
    ax.set(title="2022 monthly MAE BEFORE update", xlabel="Month", ylabel="MAE (W/m²)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "monthly_preupdate_mae.png", dpi=150)
    plt.show()

fig, ax = plt.subplots(figsize=(12, 5))
plot_df = test_df.sort_values("timestamp").head(1000)
ax.plot(plot_df["timestamp"], plot_df["DNI"], label="Actual", alpha=0.8)
ax.plot(plot_df["timestamp"], plot_df["Base Predicted DNI"], label="Base", alpha=0.6)
ax.plot(plot_df["timestamp"], plot_df["Predicted DNI"], label="Incremental", alpha=0.7, linestyle="--")
ax.set(title="2023 independent test (first 1000 daytime samples)", xlabel="Timestamp", ylabel="DNI (W/m²)")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "independent_test_predictions.png", dpi=150)
plt.show()

overall = comparison_df[comparison_df["condition"] == "overall"].set_index("model")
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, metric in zip(axes, ["MAE", "RMSE"]):
    overall[metric].plot.bar(ax=ax, rot=0, color=["#6b7280", "#0d9488"])
    ax.set(title=f"2023 test {metric}", xlabel="Model", ylabel="W/m²")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "test_model_comparison.png", dpi=150)
plt.show()
''')
replace(28, '''
## 13. 如何解读输出

- `data_split_report.csv`：时间区间和各天气类别的样本数。
- `monthly_incremental_report.csv`：2022 年每月**更新前**的分类及总体指标，`base_*` 是固定基础模型对照。
- `monthly_preupdate_predictions.csv`：每月更新前的逐样本预测，可重新计算指标。
- `final_test_comparison.csv`：2023 年独立测试，包含基础/增量模型与分类/总体指标。
- `final_test_predictions.csv`：2023 年真实 DNI、两组预测及输入特征。
- `run_summary.json`：运行参数、耗时和最终指标。
- `*.keras`、`*.scaler.joblib`、`*.buffer.joblib`、`*.state.json`：同名模型及对应预处理/回放状态。
- `*.history.csv`、`training_progress.jsonl`：训练历史与实时进度。

**指标解释**：R² 越大越好，MAE/RMSE 越小越好；增量模型不一定优于基础模型，应以独立测试为准。
总体指标直接在所有逐样本预测上计算，不对三种天气的 R² 做简单平均。
本实验只评估白天，并保留原太阳角度近似。更新同时包含新月份、回放及较低学习率，
因此单次结果比较的是这套更新方案整体，不能单独归因于某一组件；若调整参数，应使用 2022 年验证，
不要反复根据 2023 年测试结果调参。joblib buffer 的恢复需先运行 ReplayBuffer 类定义。
''')
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []
        ast.parse(''.join(cell['source']))
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(f'Updated {path.name}: {len(nb["cells"])} cells; Python syntax checked.')
