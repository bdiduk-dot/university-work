import json, math, random
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

random.seed(23)


transport_codes = {'велосипед': 0, 'авто': 1, 'пішки': 2}
weather_codes = {'ясно': 0, 'дощ': 1}
records = []
for index in range(300):
    transport = random.choice(list(transport_codes))
    weather = random.choice(list(weather_codes))
    distance = round(random.uniform(0.5, 7.0), 1)
    speed = round(random.uniform(*{'велосипед': (10, 18), 'авто': (15, 30), 'пішки': (4, 6)}[transport]), 1)
    ready = random.randint(0, 12)
    items = random.randint(1, 6)
    traffic = random.randint(0, 9)
                                                                       
    traffic_factor = 1 + traffic * (0.045 if transport == 'авто' else 0.025)
    target = max(2.0, ready + distance / speed * 60 * traffic_factor
                 + (2.5 if weather == 'дощ' else 0) + items * 0.35
                 + random.gauss(0, 2.0))
    records.append({'id': index + 1, 'distance_km': distance, 'speed_kmh': speed,
                    'transport': transport, 'ready_min': ready, 'items': items,
                    'traffic': traffic, 'weather': weather, 'actual_min': round(target, 1)})

def features(r):
    return [r['distance_km'], r['speed_kmh'], r['ready_min'], r['items'], r['traffic'],
            int(r['transport'] == 'велосипед'), int(r['transport'] == 'авто'),
            int(r['transport'] == 'пішки'), int(r['weather'] == 'дощ')]

train, calibration, test = records[:240], records[240:280], records[280:]
print(f'Усього: {len(records)}; навчання: {len(train)}; калібрування: {len(calibration)}; тест: {len(test)}')
print('Приклад входу:', {k: v for k, v in test[0].items() if k not in ('id', 'actual_min')})
print('Очікуваний час у тестовому записі:', test[0]['actual_min'], 'хв')

model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=23)
model.fit([features(r) for r in train], [r['actual_min'] for r in train])
cal_predictions = model.predict([features(r) for r in calibration])
errors = sorted(abs(r['actual_min'] - float(pred)) for r, pred in zip(calibration, cal_predictions))
rank = min(len(errors), math.ceil((len(errors) + 1) * 0.90))
radius = errors[rank - 1]

def predict_delivery(r):
    prediction = max(0.0, float(model.predict([features(r)])[0]))
    return {'eta_min': round(prediction, 1),
            'interval_min': [round(max(0.0, prediction - radius), 1), round(prediction + radius, 1)],
            'source': 'prototype_ml'}

print(f'Радіус інтервалу за калібрувальною вибіркою: {radius:.1f} хв')

results = []
for r in test:
    prediction = predict_delivery(r)
    error = abs(r['actual_min'] - prediction['eta_min'])
    results.append({'id': r['id'], 'distance_km': r['distance_km'], 'transport': r['transport'],
                    'traffic': r['traffic'], 'weather': r['weather'], 'speed_kmh': r['speed_kmh'], 'ready_min': r['ready_min'], 'items': r['items'], 'expected': r['actual_min'],
                    'actual': prediction['eta_min'], 'interval': prediction['interval_min'],
                    'correct': error <= 5.0})
mae = mean_absolute_error([r['actual_min'] for r in test],
                          [max(0.0, float(x)) for x in model.predict([features(r) for r in test])])
coverage = sum(x['interval'][0] <= x['expected'] <= x['interval'][1] for x in results) / len(results)
print('ID | км | км/год | транспорт | готовність хв | позицій | затори | погода | очікувано | прогноз | похибка ≤ 5 хв')
for x in results:
    print(f"{x['id']} | {x['distance_km']} | {x['speed_kmh']} | {x['transport']} | {x['ready_min']} | {x['items']} | {x['traffic']} | {x['weather']} | {x['expected']:.1f} | {x['actual']:.1f} | {'так' if x['correct'] else 'ні'}")
print(f'MAE на 20 записах: {mae:.2f} хв')
print(f'Похибка не більше 5 хв: {sum(x["correct"] for x in results)}/{len(results)}')
print(f'Покриття інтервалу: {coverage:.0%}')
summary = {'results': results, 'mae': round(float(mae), 2), 'coverage': coverage,
           'radius': round(radius, 2), 'example_input': {k:v for k,v in test[0].items() if k not in ('id','actual_min')},
           'example_output': predict_delivery(test[0])}
print('Приклад відповіді AI Core:', json.dumps(summary['example_output'], ensure_ascii=False))
