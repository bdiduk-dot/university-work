import contextlib
import csv
import io
import json
import random
import runpy
from pathlib import Path

root = Path(__file__).resolve().parent
prototype = root.parent / 'prototype' / 'practice_6.py'
with contextlib.redirect_stdout(io.StringIO()):
    previous = runpy.run_path(str(prototype))

rng = random.Random(71)
records = []
for index in range(60):
    transport = rng.choice(['велосипед', 'авто', 'пішки'])
    weather = rng.choice(['ясно', 'дощ'])
    distance = round(rng.uniform(0.5, 7.0), 1)
    speed = round(rng.uniform(*{'велосипед': (10, 18), 'авто': (15, 30), 'пішки': (4, 6)}[transport]), 1)
    ready = rng.randint(0, 12)
    items = rng.randint(1, 6)
    traffic = rng.randint(0, 9)
    factor = 1 + traffic * (0.045 if transport == 'авто' else 0.025)
    actual = round(max(2.0, ready + distance / speed * 60 * factor
                       + (2.5 if weather == 'дощ' else 0) + items * 0.35
                       + rng.gauss(0, 2.0)), 1)
    record = {'id': index + 301, 'distance_km': distance, 'speed_kmh': speed,
              'transport': transport, 'ready_min': ready, 'items': items,
              'traffic': traffic, 'weather': weather, 'actual_min': actual}
    prediction = previous['predict_delivery'](record)
    record.update(predicted_min=prediction['eta_min'], lower_min=prediction['interval_min'][0],
                  upper_min=prediction['interval_min'][1])
    record['error_min'] = round(record['predicted_min'] - actual, 1)
    record['correct'] = abs(record['error_min']) <= 5.0
    record['covered'] = record['lower_min'] <= actual <= record['upper_min']
    records.append(record)

count = len(records)
mae = sum(abs(row['error_min']) for row in records) / count
within_5 = sum(row['correct'] for row in records)
coverage = sum(row['covered'] for row in records)
summary = {'count': count, 'mae': round(mae, 4), 'within_5': within_5,
           'coverage': coverage, 'radius': previous['radius'], 'records': records}
with (root / 'test_results.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
(root / 'test_results.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'New test examples: {count}; MAE: {mae:.2f}; |error| <= 5: {within_5}/{count}; interval coverage: {coverage}/{count}')
print('First 12:')
for r in records[:12]:
    print(r)
print('Largest errors:')
for r in sorted(records, key=lambda x: abs(x['error_min']), reverse=True)[:6]:
    print(r)
