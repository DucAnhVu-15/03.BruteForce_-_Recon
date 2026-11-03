"""Chạy model ONNX hgb_pkt trên pcap và báo brute-force / recon theo từng host — mô phỏng inference trên router.

Pipeline (giống router): tshark đọc gói -> cửa sổ lai 50 gói / 10 s theo host nguồn + ngữ cảnh 60/300 s (pkt_features,
chép từ 03b) -> ONNX -> score = 1 - p[benign] >= ngưỡng -> gộp đợt báo động theo host (lặng 60 s thì đóng đợt).

Ví dụ:
  PY=/home/ubuntu/thangdh/python_env/bin/python
  # một pcap
  $PY Code/deploy/test_bruteforce.py Data/real_data/brute_force/ssh/pcaps/task_0005.pcap --lab-clean
  # cả thư mục, chỉ các task thuộc split test (chưa dùng để train/chọn ngưỡng), kỳ vọng brute-force
  $PY Code/deploy/test_bruteforce.py Data/real_data/brute_force/ssh/pcaps --split test --expect bruteforce --lab-clean
  # pcap tự bắt trên router, chỉ xét host nghi vấn, lưu chi tiết từng cửa sổ
  $PY Code/deploy/test_bruteforce.py capture.pcap --host 192.168.1.50 --out-dir /tmp/kq
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pkt_features as pf  # noqa: E402

ROOT = Path('/home/ubuntu/sepcung/03.Brute_Force_vs_Recon')
MODEL_ROOT = ROOT / 'Model' / 'hgb_pkt'
WIN_DIR = ROOT / 'Data_after_processing' / 'windows_pkt'
# luồng SSH router <-> laptop tấn công do plink stream pcap về máy — artifact của cách bắt trong lab (xem notebook 01)
LAB_CLEAN = 'not (ip.addr == 192.168.1.1 and ip.addr == 192.168.1.236 and tcp.port == 22)'


def load_model(run):
    run = run or str(pd.read_csv(MODEL_ROOT / 'runs.csv')['run_id'].iloc[-1])
    cfg_path = MODEL_ROOT / run / 'deploy_config.json'
    if not cfg_path.exists():
        sys.exit(f'Chưa có {cfg_path} — chạy Code/deploy/05_export_onnx.py --run {run} trước.')
    cfg = json.loads(cfg_path.read_text())
    assert cfg['features'] == pf.FEATURES, 'thứ tự đặc trưng của model khác pkt_features.FEATURES'
    w = cfg['window']
    assert (w['n_pkt'], w['t_max'], tuple(w['ctx']), w['new_gap']) == (pf.N_PKT, pf.T_MAX, pf.CTX, pf.NEW_GAP), \
        'tham số cửa sổ của model khác pkt_features'
    sess = ort.InferenceSession(str(MODEL_ROOT / run / cfg['onnx']), providers=['CPUExecutionProvider'])
    return run, cfg, sess


def task_split():
    # sample_id -> split của các task lab đã trộn vào train/val/test (để biết pcap nào chưa từng dùng khi train)
    if not all((WIN_DIR / f'{s}.parquet').exists() for s in ['train', 'val', 'test']):
        return {}
    ids = pd.concat([pd.read_parquet(WIN_DIR / f'{s}.parquet', columns=['source', 'sample_id', 'split'])
                     for s in ['train', 'val', 'test']])
    ids = ids[ids['source'] == 'mix'].drop_duplicates('sample_id')
    return dict(zip(ids['sample_id'], ids['split']))


def sample_id(pcap):
    # .../<scenario>/pcaps/task_0005.pcap -> <scenario>_0005 (cách đặt sample_id của 03b)
    m = re.match(r'task_(\d+)', pcap.stem)
    return f'{pcap.parent.parent.name}_{int(m.group(1)):04d}' if m and pcap.parent.name == 'pcaps' else None


def episodes(win, gap):
    a = win[win['alarm']].sort_values(['host', 't_end'])
    if a.empty:
        return pd.DataFrame(columns=['host', 'start', 'end', 'class', 'alarm_windows', 'max_score'])
    a = a.assign(ep=(a.groupby('host')['t_end'].diff().fillna(np.inf) > gap).cumsum())
    return a.groupby('ep').agg(host=('host', 'first'), start=('t_end', 'min'), end=('t_end', 'max'),
                               **{'class': ('cls', lambda c: c.mode().iat[0])},
                               alarm_windows=('score', 'size'), max_score=('score', 'max')).reset_index(drop=True)


def run_pcap(pcap, cfg, sess, args):
    pk = pf.read_pcap(pcap, LAB_CLEAN if args.lab_clean else args.filter)
    if pk.empty:
        return None, None, 'không có gói nào (sau lọc)'
    pk['label'] = ''  # pkt_features cần cột này (chỉ dùng cho nhãn lúc train)
    try:
        win = pf.window_features(pk, keep_hosts=args.host or None, lan_only=not args.host)
    except AssertionError:
        return None, None, 'không có gói gửi từ host cần đánh giá'
    X = win[cfg['features']].to_numpy(np.float32)
    p = sess.run([cfg['output_proba']], {cfg['input']: X})[0]
    win = win[['host', 't_start', 't_end']].copy()
    win['score'] = 1 - p[:, 0]
    win['cls'] = np.array(cfg['classes'][1:])[p[:, 1:].argmax(axis=1)]
    win['alarm'] = win['score'] >= cfg['threshold']
    ep = episodes(win, cfg['episode_gap_s'])
    t0 = pk['ts'].min()
    ep[['start', 'end']] = ep[['start', 'end']] - t0  # giây kể từ gói đầu của pcap
    win.attrs['hours'] = (pk['ts'].max() - t0) / 3600
    return win, ep, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pcaps', nargs='+', help='file pcap/pcapng hoặc thư mục chứa chúng')
    ap.add_argument('--run', help='run_id trong Model/hgb_pkt (mặc định: dòng cuối runs.csv)')
    ap.add_argument('--host', action='append', help='chỉ đánh giá host này (lặp lại được); mặc định mọi host LAN')
    ap.add_argument('--lan-prefix', default=pf.LAN_PREFIX,
                    help=f'tiền tố IP của mạng LAN cần đánh giá (mặc định {pf.LAN_PREFIX}, vd 192.168.137. cho CIC IoT 2022)')
    ap.add_argument('--expect', choices=['bruteforce', 'recon', 'benign'],
                    help='kết quả mong đợi -> chấm ĐẠT/TRƯỢT cho từng pcap')
    ap.add_argument('--split', choices=['train', 'val', 'test', 'unused'],
                    help='chỉ chạy task lab thuộc split này (unused = task không được trộn vào dữ liệu)')
    ap.add_argument('--lab-clean', action='store_true', help='bỏ luồng SSH router<->laptop (artifact capture của lab)')
    ap.add_argument('--filter', help='display filter tshark (-Y) tự đặt; bị --lab-clean ghi đè')
    ap.add_argument('--limit', type=int, help='chỉ chạy N pcap đầu')
    ap.add_argument('--out-dir', type=Path, help='lưu windows.csv, episodes.csv, summary.csv')
    args = ap.parse_args()
    pf.LAN_PREFIX = args.lan_prefix  # window_features(lan_only=True) đọc biến này

    run, cfg, sess = load_model(args.run)
    files = []
    for p in map(Path, args.pcaps):
        files += sorted(p.glob('*.pcap*')) if p.is_dir() else [p]
    split_of = task_split() if args.split or any(sample_id(f) for f in files) else {}
    if args.split:
        files = [f for f in files if split_of.get(sample_id(f), 'unused') == args.split]
    files = files[:args.limit] if args.limit else files
    if not files:
        sys.exit('Không có pcap nào để chạy.')
    print(f'model hgb_pkt {run} | ngưỡng {cfg["threshold"]:.4f} | {len(files)} pcap\n')

    summary, all_win, all_ep = [], [], []
    for f in files:
        t = time.time()
        win, ep, err = run_pcap(f, cfg, sess, args)
        sid = sample_id(f)
        row = {'pcap': str(f), 'split': split_of.get(sid, 'unused') if sid else '-'}
        if err:
            summary.append({**row, 'note': err})
            print(f'{f.name}: {err}')
            continue
        bf, rc = ep[ep['class'] == 'bruteforce'], ep[ep['class'] == 'recon']
        row.update({'hours': round(win.attrs['hours'], 3), 'hosts': win['host'].nunique(), 'windows': len(win),
                    'alarm_windows': int(win['alarm'].sum()),
                    'episodes': len(ep), 'bruteforce_episodes': len(bf), 'recon_episodes': len(rc),
                    'max_score': round(float(win['score'].max()), 4),
                    'first_alarm_s': round(float(ep['start'].min()), 2) if len(ep) else np.nan,
                    'seconds': round(time.time() - t, 1)})
        if args.expect:
            row['pass'] = {'bruteforce': len(bf) > 0, 'recon': len(rc) > 0, 'benign': len(ep) == 0}[args.expect]
        summary.append(row)
        verdict = '' if not args.expect else ('  ĐẠT' if row['pass'] else '  TRƯỢT')
        print(f"{f.name} [{row['split']}]: {row['windows']} cửa sổ, {row['alarm_windows']} báo động, "
              f"{len(bf)} đợt bruteforce, {len(rc)} đợt recon, score max {row['max_score']}{verdict}")
        for _, e in ep.iterrows():
            print(f"    {e['host']:<15} {e['class']:<10} {e['start']:7.1f}s → {e['end']:7.1f}s  "
                  f"{e['alarm_windows']} cửa sổ, score max {e['max_score']:.3f}")
        all_win.append(win.assign(pcap=str(f)))
        all_ep.append(ep.assign(pcap=str(f)))

    s = pd.DataFrame(summary)
    print(f'\n== Tổng kết: {len(s)} pcap')
    if 'windows' in s:
        ok = s.dropna(subset=['windows'])
        print(f"   có báo động bruteforce: {(ok['bruteforce_episodes'] > 0).sum()}/{len(ok)}   "
              f"có báo động recon: {(ok['recon_episodes'] > 0).sum()}/{len(ok)}   "
              f"không báo động: {(ok['episodes'] == 0).sum()}/{len(ok)}")
        print(f"   báo động đầu tiên sau (trung vị): {ok['first_alarm_s'].median():.1f}s kể từ gói đầu pcap")
        h = ok['hours'].sum()
        print(f"   {h:.1f} giờ traffic: {ok['alarm_windows'].sum() / h:.2f} cửa sổ báo động/giờ, "
              f"{ok['episodes'].sum() / h:.2f} đợt/giờ (nếu pcap toàn benign thì đây là báo động giả)")
    if args.expect and 'pass' in s:
        print(f"   ĐẠT ({args.expect}): {int(s['pass'].fillna(False).sum())}/{len(s)}")
        fails = s.loc[s['pass'].fillna(False) == False, 'pcap'].tolist()  # noqa: E712
        if fails:
            print('   TRƯỢT:', ', '.join(Path(x).name for x in fails[:20]) + (' …' if len(fails) > 20 else ''))
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        s.to_csv(args.out_dir / 'summary.csv', index=False)
        if all_win:
            pd.concat(all_win).to_csv(args.out_dir / 'windows.csv', index=False)
            pd.concat(all_ep).to_csv(args.out_dir / 'episodes.csv', index=False)
        print(f'   -> {args.out_dir}')


if __name__ == '__main__':
    main()
