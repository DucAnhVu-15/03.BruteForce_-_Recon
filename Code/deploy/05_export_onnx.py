"""05 — Xuất model hgb_pkt sang ONNX + config triển khai, rồi kiểm tra ONNX khớp sklearn.

Chạy:  /home/ubuntu/thangdh/python_env/bin/python Code/deploy/05_export_onnx.py [--run 20261102_1532]

Ghi vào Model/hgb_pkt/<run>/:
- model.onnx          input `features` float32 [N, 97], output `label` int64 [N] + `probabilities` float32 [N, 3]
- deploy_config.json  thứ tự đặc trưng, lớp, ngưỡng, tham số cửa sổ và gộp đợt — router/tool đọc file này
- onnx_check.csv      so ONNX với sklearn trên val/test: lệch xác suất, số quyết định báo động khác nhau
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

ROOT = Path('/home/ubuntu/sepcung/03.Brute_Force_vs_Recon')
MODEL_ROOT = ROOT / 'Model' / 'hgb_pkt'
WIN_DIR = ROOT / 'Data_after_processing' / 'windows_pkt'
EPISODE_GAP_S = 60  # khớp 04b
LAN_PREFIX = '192.168.1.'  # khớp 03b

ap = argparse.ArgumentParser()
ap.add_argument('--run', default=None, help='run_id trong Model/hgb_pkt (mặc định: dòng cuối runs.csv)')
args = ap.parse_args()
run = args.run or str(pd.read_csv(MODEL_ROOT / 'runs.csv')['run_id'].iloc[-1])
run_dir = MODEL_ROOT / run
bundle = joblib.load(run_dir / 'model.joblib')
model, features, classes, thr = bundle['model'], bundle['features'], bundle['classes'], bundle['threshold']
print(f'run {run}: {type(model).__name__}, {len(features)} đặc trưng, lớp {classes}, ngưỡng {thr:.4f}')

# --- xuất ONNX ---
onx = to_onnx(model, initial_types=[('features', FloatTensorType([None, len(features)]))],
              options={type(model): {'zipmap': False}})
onx.doc_string = f'hgb_pkt {run}; score = 1 - probabilities[:, 0]; báo động khi score >= {thr}'
onnx_path = run_dir / 'model.onnx'
onnx_path.write_bytes(onx.SerializeToString())
print(f'-> {onnx_path} ({onnx_path.stat().st_size / 1024:.0f} KB)')

cfg = {
    'run_id': run, 'onnx': 'model.onnx', 'input': 'features', 'output_proba': 'probabilities',
    'features': features, 'classes': classes, 'threshold': thr,
    'score': '1 - p[benign]; lớp tấn công = argmax(p[recon], p[bruteforce])',
    'window': bundle['window'], 'lan_prefix': LAN_PREFIX, 'episode_gap_s': EPISODE_GAP_S,
}
(run_dir / 'deploy_config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
print(f"-> {run_dir / 'deploy_config.json'}")

# --- kiểm tra ONNX khớp sklearn trên val/test ---
sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
rows = []
for split in ['val', 'test']:
    X = pd.read_parquet(WIN_DIR / f'{split}.parquet', columns=features).to_numpy(np.float32)
    p_sk = model.predict_proba(X)
    p_ox = sess.run(['probabilities'], {'features': X})[0]
    a_sk, a_ox = (1 - p_sk[:, 0]) >= thr, (1 - p_ox[:, 0]) >= thr
    c_sk, c_ox = p_sk[:, 1:].argmax(1), p_ox[:, 1:].argmax(1)
    rows.append({'split': split, 'windows': len(X), 'max_abs_diff_proba': float(np.abs(p_sk - p_ox).max()),
                 'alarm_mismatch': int((a_sk != a_ox).sum()),
                 'attack_class_mismatch': int(((c_sk != c_ox) & a_sk & a_ox).sum())})
check = pd.DataFrame(rows)
check.to_csv(run_dir / 'onnx_check.csv', index=False)
print(check.to_string(index=False))
ok = (check['alarm_mismatch'] == 0).all() and (check['attack_class_mismatch'] == 0).all()
print('ONNX khớp sklearn ở mọi quyết định báo động.' if ok else
      'CẢNH BÁO: ONNX khác sklearn ở một số quyết định — xem onnx_check.csv trước khi triển khai.')
