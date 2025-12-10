import pickle
import os

pkl_path = '/home/yy/pollution_mul/code/data_deal/data/sh/samples_48h.pkl'

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

results = []

for item in data:
    if 'images' in item and isinstance(item['images'], list):
        for img_path in item['images']:
            basename = os.path.basename(img_path)
            if basename.startswith('20141208') and basename.endswith('.jpg'):
                target = item.get('target', None)
                results.append({
                    'image': img_path,
                    'target': target
                })

# 打印结果
for r in results:
    print(f"Image: {r['image']}, Target: {r['target']}")