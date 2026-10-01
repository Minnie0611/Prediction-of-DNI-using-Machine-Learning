"""Check time boundaries, replay uniqueness and predict-before-update ordering."""
import os
import tempfile
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
nb = nbformat.read(ROOT / 'dni_incremental_learning_framework.ipynb', as_version=4)
scope = {'__name__': 'experiment_checks'}
for index in [1, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]:
    exec(compile(nb.cells[index].source, f'cell_{index}', 'exec'), scope)
pd, np = scope['pd'], scope['np']
df = scope['daytime_only'](scope['prepare_features'](scope['all_files']))
blocks = scope['monthly_blocks'](df)
assert [month for month, _ in blocks] == [f'2022-{month:02d}' for month in range(1, 13)]
assert all(block['timestamp'].max() < pd.Timestamp('2023-01-01') for _, block in blocks)
assert np.isfinite(df[scope['MODEL_FEATURE_COLS'] + ['DNI']].to_numpy()).all()
print('PASS: real-data features are finite; monthly blocks contain only 2022.')

Buffer = scope['ReplayBuffer']
for dni in [np.zeros(101), np.arange(101), np.repeat([0, 5, 100], [60, 30, 11])]:
    source = pd.DataFrame({'timestamp': pd.date_range('2020-01-01', periods=101, freq='h'), 'DNI': dni})
    buf = Buffer(capacity=37)
    buf.update(source)
    assert len(buf) == 37 and buf.data['timestamp'].is_unique
    for size in [1, 3, 17, 37, 100]:
        sample = buf.sample(size)
        assert len(sample) == min(size, 37) and sample['timestamp'].is_unique
    new = pd.DataFrame({'timestamp': pd.date_range('2021-01-01', periods=23, freq='h'), 'DNI': np.arange(23)})
    buf.update(new)
    assert len(buf) == 37 and buf.data['timestamp'].is_unique
    assert set(new.tail(18)['timestamp']).issubset(set(buf.data['timestamp']))
    buf.update(new)
    assert len(buf) == 37 and buf.data['timestamp'].is_unique
print('PASS: replay capacity, diversity edge cases, uniqueness and newest-row retention.')

small = pd.DataFrame({'timestamp': pd.to_datetime(['2022-01-03', '2022-01-04', '2022-02-03', '2022-02-04', '2023-01-03']),
                      'sky_condition': ['overcast'] * 5, 'DNI': [100.] * 5})
artifacts = {cond: {'trained_through': '2021-12-31', 'value': 10.} for cond in scope['CONDITIONS']}
baseline = {cond: {'trained_through': '2021-12-31', 'value': 10.} for cond in scope['CONDITIONS']}
events = []


def predict(block, models):
    events.append(('predict', block['timestamp'].min()))
    return np.full(len(block), models['overcast']['value'])


def update(cond, block, models, buffers):
    events.append(('update', block['timestamp'].min()))
    models[cond]['value'] += 10
    models[cond]['trained_through'] = str(block['timestamp'].max())
    return {'condition': cond, 'epochs': 1}


scope['predict_all_conditions'] = predict
scope['incremental_update_for_condition'] = update
with tempfile.TemporaryDirectory(dir=ROOT / 'incremental_outputs') as temp:
    scope['OUTPUT_DIR'] = Path(temp)
    report, _, _ = scope['run_incremental_learning'](small, artifacts, {}, baseline)
assert report[report.condition == 'overall']['MAE'].tolist() == [90., 80.]
assert report[report.condition == 'overall']['base_MAE'].tolist() == [90., 90.]
assert [event[0] for event in events] == ['predict', 'predict', 'update', 'update', 'update'] * 2
assert not (report['month'] >= '2023-01').any()
print('PASS: predictions precede updates; baseline stays fixed; test data never enters the loop.')
