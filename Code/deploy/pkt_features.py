"""Đọc pcap và tính 97 đặc trưng cửa sổ lai (N gói hoặc T giây) cho model hgb_pkt.

- `read_pcap`: chép từ Code/predata/01_attack_packet_labeling.ipynb (tshark, cùng trường/cột).
- `window_features` và các hàm phụ: chép NGUYÊN VĂN từ Code/predata/03b_mix_and_packet_windows.ipynb (cell 5).
  Sửa 03b thì phải chép lại vào đây, nếu không đặc trưng lúc inference sẽ lệch với lúc train.
"""
import subprocess

import numpy as np
import pandas as pd

LAN_PREFIX = '192.168.1.'
N_PKT = 50                # chốt cửa sổ khi host gửi đủ số gói này
T_MAX = 10                # ... hoặc khi đã T_MAX giây kể từ gói đầu cửa sổ
CTX = (60, 300)           # giây — ngữ cảnh dài, tính tại thời điểm chốt cửa sổ
NEW_GAP = 300             # giây — đích/cặp/port là "mới" nếu host chưa gửi tới trong NEW_GAP giây trước
HOP = 10                  # chỉ dùng để gộp báo động theo khối 10 s khi báo cáo

FIELDS = ['frame.time_epoch', 'frame.len', 'eth.src', 'eth.dst', 'ip.src', 'ip.dst', 'ip.proto', 'ip.ttl',
          'tcp.srcport', 'tcp.dstport', 'tcp.flags', 'tcp.len', 'udp.srcport', 'udp.dstport', 'icmp.type',
          'arp.opcode', 'arp.src.hw_mac', 'arp.src.proto_ipv4', 'arp.dst.proto_ipv4']
COLS = ['ts', 'len', 'eth_src', 'eth_dst', 'ip_src', 'ip_dst', 'ip_proto', 'ttl',
        'tcp_sport', 'tcp_dport', 'tcp_flags', 'tcp_len', 'udp_sport', 'udp_dport', 'icmp_type',
        'arp_op', 'arp_sha', 'arp_spa', 'arp_tpa']
INT_COLS = ['len', 'ip_proto', 'ttl', 'tcp_sport', 'tcp_dport', 'tcp_len', 'udp_sport', 'udp_dport', 'icmp_type', 'arp_op']


def read_pcap(pcap, display_filter=None):
    cmd = ['tshark', '-r', str(pcap), '-n', '-T', 'fields', '-E', 'separator=\t', '-E', 'occurrence=f']
    if display_filter:  # thêm so với 01: lọc gói bằng display filter của tshark (-Y)
        cmd += ['-Y', display_filter]
    for f in FIELDS:
        cmd += ['-e', f]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f'tshark lỗi với {pcap}: {res.stderr.strip()}')
    lines = [l.split('\t') for l in res.stdout.splitlines() if l]
    df = pd.DataFrame(lines, columns=COLS) if lines else pd.DataFrame(columns=COLS)
    df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
    for c in INT_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int32')
    df['tcp_flags'] = df['tcp_flags'].map(lambda x: int(x, 16) if x else pd.NA).astype('Int32')
    # ip.proto lấy lớp ngoài: cổng TCP/UDP bên trong ICMP error không được tính là gói TCP/UDP
    is_icmp = df['ip_proto'] == 1
    df.loc[is_icmp, ['tcp_sport', 'tcp_dport', 'tcp_flags', 'tcp_len', 'udp_sport', 'udp_dport']] = pd.NA
    for c in ['eth_src', 'eth_dst', 'arp_sha']:
        df[c] = df[c].str.lower()
    return df


FLAGS = ['tcp', 'syn', 'synack', 'rst', 'fin_only', 'null', 'xmas', 'ack_only', 'udp', 'icmp_echo', 'arp_req', 'arp_rep']
CNT_TX = ['pkts', 'bytes'] + FLAGS
CNT_RX = ['rx_pkts', 'rx_rst', 'rx_synack', 'rx_icmp_unreach']
DISTINCT = ['d_dst', 'd_pair', 'd_port', 'd_syn_pair', 'd_arp', 'd_icmp']
RATIOS = ['syn_ratio', 'rx_rst_ratio', 'rx_unreach_ratio', 'syn_per_pair', 'port_per_dst', 'bytes_per_pkt']
NEW = ['dst', 'pair', 'port']
W_FEATS = (['w_pkts', 'w_by_count', 'w_dur', 'w_span', 'w_pps', 'w_iat_mean', 'w_iat_std', 'w_iat_max',
            'w_len_mean', 'w_len_std', 'w_len_min', 'w_len_max']
           + [f'w_r_{c}' for c in FLAGS]
           + ['w_rx_ratio', 'w_rx_rst_ratio', 'w_rx_synack_ratio', 'w_rx_unreach_ratio']
           + [f'w_{d}' for d in DISTINCT] + [f'w_new_{x}_ratio' for x in NEW])
FEATURES = ['f_' + c for c in W_FEATS] + [f'f_{c}_{S}' for S in CTX for c in CNT_TX + CNT_RX + DISTINCT + RATIOS]
US = 1_000_000
SHIFT = np.int64(1) << 40


def _col(s, fill=-1):
    return pd.to_numeric(s, errors='coerce').astype('float64').fillna(fill).to_numpy().astype(np.int64)


def _str(s):
    return s.replace('', pd.NA)


def packet_table(pk):
    f = _col(pk['tcp_flags'])
    proto = _col(pk['ip_proto'])
    icmp_t = _col(pk['icmp_type'])
    arp_op = _col(pk['arp_op'], 0)
    tcp_len = _col(pk['tcp_len'], 0)
    dport = np.where(proto == 6, _col(pk['tcp_dport']), np.where(proto == 17, _col(pk['udp_dport']), -1))
    tcp, udp, icmp = proto == 6, proto == 17, proto == 1
    fv = np.where(tcp, f, -1)
    return pd.DataFrame({
        # notebook 01 lưu ô trống là '' -> coi như NA để gói ARP lấy IP từ arp_spa/arp_tpa
        'src': _str(pk['ip_src']).fillna(_str(pk['arp_spa'])).to_numpy(object),
        'dst': _str(pk['ip_dst']).fillna(_str(pk['arp_tpa'])).to_numpy(object),
        'ts': pk['ts'].to_numpy(float),
        'len': _col(pk['len'], 0), 'dport': dport,
        'tcp': tcp, 'syn': tcp & ((fv & 0x12) == 0x02), 'synack': tcp & ((fv & 0x12) == 0x12),
        'rst': tcp & ((fv & 0x04) > 0), 'fin_only': tcp & ((fv & 0x17) == 0x01), 'null': tcp & (fv == 0),
        'xmas': tcp & ((fv & 0x29) == 0x29), 'ack_only': tcp & (fv == 0x10) & (tcp_len == 0),
        'udp': udp, 'icmp_echo': icmp & (icmp_t == 8), 'icmp_unreach': icmp & (icmp_t == 3),
        'arp_req': arp_op == 1, 'arp_rep': arp_op == 2,
        'attack': (pk['label'] == 'attack').to_numpy(),
    })


def _assign_windows(h, tt):
    # h, tt đã sắp theo (host, thời gian); trả về mã cửa sổ tăng dần cho từng gói
    wid = np.empty(len(h), np.int64)
    bounds = np.flatnonzero(np.r_[True, h[1:] != h[:-1], True])
    k = 0
    for a, e in zip(bounds[:-1], bounds[1:]):
        seg = tt[a:e]
        i, n = 0, e - a
        while i < n:
            j = min(i + N_PKT, int(np.searchsorted(seg, seg[i] + T_MAX * US, side='left')))
            wid[a + i:a + j] = k
            k += 1
            i = j
    return wid


def _ctx_distinct(h, item, tt, span, qkeys):
    # số item khác nhau host đã gửi tới trong (q - span, q]
    if len(h) == 0:
        return np.zeros(len(qkeys))
    o = np.lexsort((tt, item, h))
    h, item, tt = h[o], item[o], tt[o]
    new = np.r_[True, (h[1:] != h[:-1]) | (item[1:] != item[:-1]) | (tt[1:] > tt[:-1] + span)]
    st = np.flatnonzero(new)
    last = np.maximum.reduceat(tt, st)
    hh = h[st]
    ev_k = np.r_[hh * SHIFT + tt[st], hh * SHIFT + last + span]
    ev_d = np.r_[np.ones(len(st)), -np.ones(len(st))]
    o = np.argsort(ev_k, kind='stable')
    ev_k, cs = ev_k[o], np.cumsum(ev_d[o])
    idx = np.searchsorted(ev_k, qkeys, side='right') - 1
    return np.where(idx >= 0, cs[np.maximum(idx, 0)], 0)


def window_features(pk, keep_hosts=None, lan_only=False):
    t = packet_table(pk)
    t_base = np.floor(t['ts'].min())
    tt_all = np.round((t['ts'].to_numpy() - t_base) * US).astype(np.int64)
    hosts, inv = np.unique(np.r_[t['src'].fillna('').to_numpy(str), t['dst'].fillna('').to_numpy(str)], return_inverse=True)
    n = len(t)
    hs_all, hd_all = inv[:n].astype(np.int64), inv[n:].astype(np.int64)
    ok = hosts != ''
    if keep_hosts is not None:
        ok &= np.isin(hosts, list(keep_hosts))
    if lan_only:
        ok &= np.char.startswith(hosts, LAN_PREFIX)

    # gói gửi (tx) / nhận (rx) của các host cần đánh giá, sắp theo (host, thời gian)
    m = ok[hs_all]
    tx_idx = np.flatnonzero(m)[np.lexsort((tt_all[m], hs_all[m]))]
    m = ok[hd_all]
    rx_idx = np.flatnonzero(m)[np.lexsort((tt_all[m], hd_all[m]))]
    tx, rx = t.iloc[tx_idx].reset_index(drop=True), t.iloc[rx_idx].reset_index(drop=True)
    h, tt = hs_all[tx_idx], tt_all[tx_idx]
    tx_keys, rx_keys = h * SHIFT + tt, hd_all[rx_idx] * SHIFT + tt_all[rx_idx]
    assert len(tx), 'không có gói gửi của host cần đánh giá'

    wid = _assign_windows(h, tt)
    nW = int(wid[-1]) + 1
    st = np.flatnonzero(np.r_[True, wid[1:] != wid[:-1]])
    en = np.r_[st[1:], len(wid)] - 1
    S = lambda v: np.bincount(wid, weights=np.asarray(v, float), minlength=nW)
    cnt = S(np.ones(len(wid)))
    t_first, t_last = tt[st], tt[en]
    by_count = cnt >= N_PKT
    t_end = np.where(by_count, t_last, t_first + T_MAX * US)
    hw = h[st]
    f = {}

    # --- trong cửa sổ ---
    dur = (t_end - t_first) / US
    f['w_pkts'], f['w_by_count'], f['w_dur'] = cnt, by_count.astype(float), dur
    f['w_span'], f['w_pps'] = (t_last - t_first) / US, cnt / np.maximum(dur, 1e-3)
    same = wid[1:] == wid[:-1]
    iat, iw = np.diff(tt)[same] / US, wid[1:][same]
    n_iat = np.maximum(np.bincount(iw, minlength=nW), 1)
    mean = np.bincount(iw, iat, nW) / n_iat
    f['w_iat_mean'] = mean
    f['w_iat_std'] = np.sqrt(np.maximum(np.bincount(iw, iat ** 2, nW) / n_iat - mean ** 2, 0))
    mx = np.zeros(nW)
    np.maximum.at(mx, iw, iat)
    f['w_iat_max'] = mx
    ln = tx['len'].to_numpy(float)
    lm = S(ln) / cnt
    f['w_len_mean'], f['w_len_std'] = lm, np.sqrt(np.maximum(S(ln ** 2) / cnt - lm ** 2, 0))
    f['w_len_min'], f['w_len_max'] = np.minimum.reduceat(ln, st), np.maximum.reduceat(ln, st)
    for c in FLAGS:
        f[f'w_r_{c}'] = S(tx[c]) / cnt
    rvals = np.c_[np.ones(len(rx)), rx[['rst', 'synack', 'icmp_unreach', 'attack']].to_numpy(float)]
    rcs = np.vstack([np.zeros((1, 5)), np.cumsum(rvals, axis=0)])
    rw = (rcs[np.searchsorted(rx_keys, hw * SHIFT + t_end, side='right')]
          - rcs[np.searchsorted(rx_keys, hw * SHIFT + t_first, side='left')])
    for i, c in enumerate(['w_rx_ratio', 'w_rx_rst_ratio', 'w_rx_synack_ratio', 'w_rx_unreach_ratio']):
        f[c] = rw[:, i] / cnt

    dst = tx['dst'].fillna('').to_numpy(str)
    dstc = pd.factorize(dst)[0].astype(np.int64)
    dport = tx['dport'].to_numpy(np.int64)
    pair = dstc * 65536 + dport
    items = {
        'd_dst': (dst != '', dstc), 'd_pair': (dport >= 0, pair), 'd_port': (dport >= 0, dport),
        'd_syn_pair': (tx['syn'].to_numpy(bool), pair),
        'd_arp': (tx['arp_req'].to_numpy(bool) & (dst != ''), dstc),
        'd_icmp': (tx['icmp_echo'].to_numpy(bool), dstc),
    }
    for name, (mk, it) in items.items():
        f[f'w_{name}'] = np.bincount(np.unique(wid[mk] * (np.int64(1) << 32) + it[mk]) >> 32, minlength=nW).astype(float)
    for x, name in zip(NEW, ['d_dst', 'd_pair', 'd_port']):
        mk, it = items[name]
        idx = np.flatnonzero(mk)
        o = idx[np.lexsort((tt[idx], it[idx], h[idx]))]
        first = np.r_[True, (h[o][1:] != h[o][:-1]) | (it[o][1:] != it[o][:-1]) | (np.diff(tt[o]) > NEW_GAP * US)]
        f[f'w_new_{x}_ratio'] = np.bincount(wid[o], weights=first.astype(float), minlength=nW) / cnt

    # --- ngữ cảnh dài tại lúc chốt ---
    tvals = np.c_[np.ones(len(tx)), ln, tx[FLAGS].to_numpy(float), tx['attack'].to_numpy(float)]
    tcs = np.vstack([np.zeros((1, tvals.shape[1])), np.cumsum(tvals, axis=0)])
    qk = hw * SHIFT + t_end
    hi_t, hi_r = np.searchsorted(tx_keys, qk, side='right'), np.searchsorted(rx_keys, qk, side='right')
    for Sx in CTX:
        lo = hw * SHIFT + np.maximum(t_end - Sx * US, -1)  # không tràn sang khoá của host trước
        ct = tcs[hi_t] - tcs[np.searchsorted(tx_keys, lo, side='right')]
        cr = rcs[hi_r] - rcs[np.searchsorted(rx_keys, lo, side='right')]
        for i, c in enumerate(CNT_TX):
            f[f'{c}_{Sx}'] = ct[:, i]
        for i, c in enumerate(CNT_RX):
            f[f'{c}_{Sx}'] = cr[:, i]
        for name, (mk, it) in items.items():
            f[f'{name}_{Sx}'] = _ctx_distinct(h[mk], it[mk], tt[mk], Sx * US, qk)
        g = lambda c: np.maximum(f[f'{c}_{Sx}'], 1)
        f[f'syn_ratio_{Sx}'] = f[f'syn_{Sx}'] / g('tcp')
        f[f'rx_rst_ratio_{Sx}'] = f[f'rx_rst_{Sx}'] / g('pkts')
        f[f'rx_unreach_ratio_{Sx}'] = f[f'rx_icmp_unreach_{Sx}'] / g('pkts')
        f[f'syn_per_pair_{Sx}'] = f[f'syn_{Sx}'] / g('d_syn_pair')
        f[f'port_per_dst_{Sx}'] = f[f'd_port_{Sx}'] / g('d_dst')
        f[f'bytes_per_pkt_{Sx}'] = f[f'bytes_{Sx}'] / g('pkts')
    attack_trail = ct[:, -1] + cr[:, 4]  # ct/cr của CTX[-1] = 300 s

    out = pd.DataFrame({'host': hosts[hw], 't_start': t_base + t_first / US, 't_end': t_base + t_end / US})
    out['b10'] = np.floor(out['t_end'] / HOP).astype(np.int64)
    out['attack_tx_cur'], out['attack_trail'] = S(tx['attack']), attack_trail
    feats = pd.DataFrame({'f_' + k: v for k, v in f.items()})[FEATURES]
    return pd.concat([out, feats], axis=1)
