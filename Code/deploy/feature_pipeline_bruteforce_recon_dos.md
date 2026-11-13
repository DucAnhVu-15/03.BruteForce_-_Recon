# Luồng xử lý và cách lấy đặc trưng cho model `hgb_pkt`

Tài liệu mô tả đường đi của dữ liệu từ gói tin đến cảnh báo, và cách tính từng đặc trưng trong 97 đặc trưng
mà model nhận vào. Nguồn chuẩn là code:

- [`pkt_features.py`](pkt_features.py): đọc pcap và tính đặc trưng (dùng khi inference).
- Cell 5 của [`03b_mix_and_packet_windows.ipynb`](../predata/03b_mix_and_packet_windows.ipynb): bản giống hệt, dùng khi tạo dữ liệu train.

Hai bản này **phải giống nhau từng dòng**. Sửa một bên thì phải chép sang bên kia, nếu không đặc trưng lúc chạy
sẽ lệch với lúc train.

Các con số cụ thể (ngưỡng, số vòng) lấy theo run `20261111_0925`.

---

## 1. Tổng quan

```
              ┌──────────── train (một lần, offline) ─────────────┐
 pcap lab ──► 01 gán nhãn gói ─┐
 pcap TII ──► 01c gán nhãn DoS ─┼─► 03b trộn với nền UNSW ─► cửa sổ + 97 đặc trưng ─► 04b train HGB ─► 05 xuất ONNX
 pcap UNSW ─► 02 nền benign ────┤        (+ tăng cường DoS)                            chọn ngưỡng     + deploy_config.json
 pcap QoS/ZT ─► 01d nền khó ────┘
              └────────────────────────────────────────────────────┘

              ┌──────────── inference (router / test_bruteforce.py) ──────────────────────────────┐
 gói tin ──► đọc 19 trường ──► bảng gói ──► cửa sổ theo host nguồn ──► 97 đặc trưng ──► ONNX ──► score ──► đợt báo động
              (tshark)        (cờ, port)   (50 gói hoặc 10 s)          (37 + 2×30)       4 lớp    ≥ ngưỡng   (gộp 60 s)
              └────────────────────────────────────────────────────────────────────────────────────┘
```

Ý tưởng chính: **mỗi host LAN được đánh giá dựa trên các gói nó gửi đi.** Gói của host được gom thành từng cửa sổ
ngắn. Mỗi cửa sổ có đặc trưng của chính nó, cộng với đặc trưng ngữ cảnh 60 s và 300 s của cùng host tính tại lúc
chốt cửa sổ. Model chấm từng cửa sổ một cách độc lập.

Đặc trưng **không** dùng IP, MAC, TTL hay số port cụ thể. Model chỉ thấy hành vi: đếm, tỉ lệ, số phần tử khác nhau
và thời gian.

---

## 2. Bước 1 — Đọc gói tin

`read_pcap()` gọi `tshark -T fields`, lấy **lần xuất hiện đầu tiên** (`occurrence=f`) của 19 trường:

| Trường tshark | Cột | Ghi chú |
|---|---|---|
| `frame.time_epoch` | `ts` | giây, số thực |
| `frame.len` | `len` | byte cả khung (gồm header Ethernet) |
| `eth.src`, `eth.dst` | `eth_src`, `eth_dst` | không dùng làm đặc trưng |
| `ip.src`, `ip.dst` | `ip_src`, `ip_dst` | dùng để xác định host gửi/nhận |
| `ip.proto` | `ip_proto` | 6 = TCP, 17 = UDP, 1 = ICMP |
| `ip.ttl` | `ttl` | không dùng làm đặc trưng |
| `tcp.srcport`, `tcp.dstport` | `tcp_sport`, `tcp_dport` | |
| `tcp.flags` | `tcp_flags` | hex → số nguyên |
| `tcp.len` | `tcp_len` | độ dài payload TCP |
| `udp.srcport`, `udp.dstport` | `udp_sport`, `udp_dport` | |
| `icmp.type` | `icmp_type` | |
| `arp.opcode`, `arp.src.hw_mac`, `arp.src.proto_ipv4`, `arp.dst.proto_ipv4` | `arp_op`, `arp_sha`, `arp_spa`, `arp_tpa` | |

Hai quy tắc làm sạch:

- **Gói ICMP**: xoá mọi cột TCP/UDP. Gói ICMP lỗi (vd port unreachable) mang header TCP/UDP của gói gốc bên trong,
  và các cổng đó không được tính là gói TCP/UDP.
- **ARP**: gói ARP không có `ip.src`/`ip.dst`, nên lấy IP từ `arp_spa`/`arp_tpa` (ở bước 2).

Có thể lọc gói trước khi xử lý bằng display filter (`-Y`), ví dụ `--lab-clean` trong `test_bruteforce.py`.

---

## 3. Bước 2 — Bảng gói: phân loại từng gói

`packet_table()` biến mỗi gói thành một dòng gồm: `src`, `dst`, `ts`, `len`, `dport` và các cờ đúng/sai.

- `src` = `ip_src`, nếu trống thì lấy `arp_spa`. `dst` = `ip_dst`, nếu trống thì lấy `arp_tpa`.
- `dport` = `tcp_dport` nếu là TCP, `udp_dport` nếu là UDP, còn lại là `-1` (không có port).

Bảng phân loại. `f` là giá trị cờ TCP; `hdr` nghĩa là gói TCP **có** header TCP:

| Tên | Điều kiện | Ý nghĩa |
|---|---|---|
| `tcp` | `ip_proto == 6` | mọi gói TCP, kể cả mảnh IP |
| `syn` | `hdr` và `f & 0x12 == 0x02` | SYN không kèm ACK (mở kết nối, quét SYN, SYN flood) |
| `synack` | `hdr` và `f & 0x12 == 0x12` | SYN-ACK (trả lời port mở) |
| `rst` | `hdr` và `f & 0x04` | có cờ RST |
| `fin_only` | `hdr` và `f & 0x17 == 0x01` | FIN, không SYN/RST/ACK (quét FIN) |
| `null` | `hdr` và `f == 0` | không cờ nào (quét NULL) |
| `xmas` | `hdr` và `f & 0x29 == 0x29` | FIN + PSH + URG (quét XMAS) |
| `ack_only` | `hdr` và `f == 0x10` và `tcp_len == 0` | ACK thuần không dữ liệu |
| `udp` | `ip_proto == 17` | |
| `icmp_echo` | ICMP và `icmp_type == 8` | ping (echo request) |
| `icmp_unreach` | ICMP và `icmp_type == 3` | destination unreachable (chỉ dùng ở chiều nhận) |
| `arp_req` / `arp_rep` | `arp_op == 1` / `2` | |

> **Mảnh IP.** Gói TCP lớn bị phân mảnh: các mảnh sau không có header TCP nên `tcp.flags` trống. Những gói này
> vẫn tính là `tcp`, nhưng **không bật cờ nào** (`hdr = tcp & (flags có giá trị)`). Trước bản sửa v4, cờ trống bị
> điền thành `-1` (bật mọi bit) nên mỗi mảnh bị đếm cùng lúc là SYN-ACK, RST và XMAS. Model train trước bản sửa
> (≤ `20261105_1007`) phải dùng `pkt_features.py` cũ mới khớp.

---

## 4. Bước 3 — Chọn host và cắt cửa sổ

### Host được đánh giá

- Mặc định (`lan_only=True`): mọi IP bắt đầu bằng `LAN_PREFIX` (`192.168.1.`), cả khi gửi lẫn khi nhận.
- Hoặc chỉ các host liệt kê trong `keep_hosts` (`--host` của `test_bruteforce.py`).

Với mỗi host có hai dòng gói, đều sắp theo thời gian:
- **tx**: các gói host **gửi đi** (`src == host`). Cửa sổ được cắt trên dòng này.
- **rx**: các gói host **nhận về** (`dst == host`). Chỉ dùng để tính tỉ lệ phản hồi.

### Cửa sổ lai: 50 gói hoặc 10 giây

Cắt tham lam trên dòng tx của từng host. Cửa sổ mở ở gói đầu tiên, rồi gom các gói kế tiếp cho tới khi:

- **đủ `N_PKT = 50` gói** (chốt theo số gói), hoặc
- gói kế tiếp đến **muộn hơn `T_MAX = 10` s** tính từ gói đầu cửa sổ (chốt theo thời gian),

tuỳ điều kiện nào đến trước. Gói kế tiếp sẽ mở cửa sổ mới.

**Thời điểm chốt `t_end`:**
- chốt theo số gói → thời điểm của gói thứ 50 (`t_last`);
- chốt theo thời gian → `t_first + 10 s`.

Nhờ cách cắt này, flood hoặc quét nhanh được chốt chỉ sau vài mili-giây, còn tấn công chậm vẫn được chốt sau tối đa 10 s.

Ví dụ (đúng với kiểm thử trong 03b cell 6): host quét SYN 120 port, mỗi 0,01 s một gói → 3 cửa sổ 50 / 50 / 20 gói.
Hai cửa sổ đầu chốt theo số gói, cửa sổ cuối chốt theo thời gian.

---

## 5. Bước 4 — 37 đặc trưng trong cửa sổ (`f_w_*`)

Ký hiệu: `cnt` = số gói tx trong cửa sổ; `dur` = `t_end − t_first`. Mọi "tỉ lệ" đều chia cho `cnt`.

### Hình dạng và thời gian (12)

| Đặc trưng | Cách tính |
|---|---|
| `f_w_pkts` | `cnt` |
| `f_w_by_count` | 1 nếu chốt theo số gói (đủ 50), 0 nếu chốt theo thời gian |
| `f_w_dur` | `t_end − t_first` (giây); cửa sổ chốt theo thời gian luôn là 10 |
| `f_w_span` | `t_last − t_first`: khoảng thời gian thực sự có gói |
| `f_w_pps` | `cnt / max(dur, 0,001)`: gói/giây |
| `f_w_iat_mean`, `f_w_iat_std`, `f_w_iat_max` | trung bình / độ lệch chuẩn / lớn nhất của khoảng cách giữa hai gói tx liên tiếp trong cửa sổ (0 nếu chỉ có 1 gói) |
| `f_w_len_mean`, `f_w_len_std`, `f_w_len_min`, `f_w_len_max` | thống kê `frame.len` của các gói tx |

### Tỉ lệ loại gói gửi (12)

`f_w_r_<loại>` = (số gói tx thuộc loại đó) / `cnt`, với loại ∈
`tcp, syn, synack, rst, fin_only, null, xmas, ack_only, udp, icmp_echo, arp_req, arp_rep`
(định nghĩa ở bảng mục 3).

### Phản hồi nhận về (4)

Đếm các gói **rx** của host có `ts` trong `[t_first, t_end]`, rồi chia cho `cnt`:

| Đặc trưng | Tử số |
|---|---|
| `f_w_rx_ratio` | mọi gói nhận |
| `f_w_rx_rst_ratio` | gói nhận có RST (quét gặp port đóng) |
| `f_w_rx_synack_ratio` | gói nhận là SYN-ACK (quét gặp port mở) |
| `f_w_rx_unreach_ratio` | ICMP unreachable nhận về (quét UDP gặp port đóng) |

### Số phần tử khác nhau (6)

| Đặc trưng | Đếm số giá trị khác nhau của | Chỉ xét gói |
|---|---|---|
| `f_w_d_dst` | IP đích | có đích |
| `f_w_d_pair` | cặp (IP đích, port đích) | có port (TCP/UDP) |
| `f_w_d_port` | port đích | có port |
| `f_w_d_syn_pair` | cặp (IP đích, port đích) | gói `syn` |
| `f_w_d_arp` | IP được hỏi trong ARP | `arp_req` |
| `f_w_d_icmp` | IP đích | `icmp_echo` |

### Tỉ lệ "mới" (3)

Một gói là **mới** với host nếu host chưa gửi gì tới cùng phần tử đó trong `NEW_GAP = 300` s trước (hoặc chưa từng
gửi). Phần tử ở đây là đích / cặp / port, như cột "đếm" ở bảng trên.

| Đặc trưng | Cách tính |
|---|---|
| `f_w_new_dst_ratio` | số gói tới **đích mới** / `cnt` |
| `f_w_new_pair_ratio` | số gói tới **cặp (đích, port) mới** / `cnt` |
| `f_w_new_port_ratio` | số gói tới **port mới** / `cnt` |

Quét port gần như gói nào cũng là cặp mới (tỉ lệ ≈ 1). Brute-force và flood lặp lại cùng một cặp (tỉ lệ ≈ 0).
Trên router chỉ cần một bảng "lần gửi gần nhất" theo (host, phần tử).

---

## 6. Bước 5 — 60 đặc trưng ngữ cảnh (`f_<tên>_60`, `f_<tên>_300`)

Tính tại thời điểm chốt `t_end`, trên khoảng **(t_end − S, t_end]** với S = 60 s và S = 300 s. Mỗi S có 30 đặc trưng,
tổng 60. Ngữ cảnh giúp bắt các tấn công rất thưa (vd host discovery chỉ 6–14 gói) mà một cửa sổ đơn lẻ không thấy.

### Đếm gói gửi (14 mỗi S)

`f_pkts_S` (số gói tx), `f_bytes_S` (tổng `frame.len`), và số gói tx từng loại:
`f_tcp_S, f_syn_S, f_synack_S, f_rst_S, f_fin_only_S, f_null_S, f_xmas_S, f_ack_only_S, f_udp_S, f_icmp_echo_S, f_arp_req_S, f_arp_rep_S`.

### Đếm gói nhận (4 mỗi S)

`f_rx_pkts_S`, `f_rx_rst_S`, `f_rx_synack_S`, `f_rx_icmp_unreach_S`.

### Số phần tử khác nhau (6 mỗi S)

`f_d_dst_S, f_d_pair_S, f_d_port_S, f_d_syn_pair_S, f_d_arp_S, f_d_icmp_S`: cùng định nghĩa như mục 5, nhưng là số
phần tử khác nhau host đã gửi tới trong khoảng ngữ cảnh.

### Tỉ lệ (6 mỗi S)

`max(x, 1)` để tránh chia cho 0:

| Đặc trưng | Công thức | Gợi ý ý nghĩa |
|---|---|---|
| `f_syn_ratio_S` | `syn / max(tcp, 1)` | phần TCP là SYN |
| `f_rx_rst_ratio_S` | `rx_rst / max(pkts, 1)` | bị RST lại bao nhiêu |
| `f_rx_unreach_ratio_S` | `rx_icmp_unreach / max(pkts, 1)` | |
| `f_syn_per_pair_S` | `syn / max(d_syn_pair, 1)` | ≈ 1: quét; rất lớn: flood hoặc brute-force cùng cặp |
| `f_port_per_dst_S` | `d_port / max(d_dst, 1)` | nhiều port trên một đích: quét port |
| `f_bytes_per_pkt_S` | `bytes / max(pkts, 1)` | kích thước gói trung bình |

### Thứ tự đầu vào

Model nhận một vector `float32` có 97 phần tử, **đúng thứ tự** `pkt_features.FEATURES`:

1. 37 đặc trưng `f_w_*`, theo thứ tự ở mục 5;
2. 30 đặc trưng `_60`, theo thứ tự: đếm gửi → đếm nhận → phần tử khác nhau → tỉ lệ;
3. 30 đặc trưng `_300`, cùng thứ tự.

Danh sách và thứ tự cũng được lưu trong `deploy_config.json` (khoá `features`). `test_bruteforce.py` kiểm tra hai
danh sách này khớp nhau trước khi chạy.

---

## 7. Bước 6 — Từ đặc trưng tới cảnh báo

1. **ONNX** (`model.onnx`, input `features`) trả về `probabilities`: 4 cột theo thứ tự
   `['benign', 'recon', 'bruteforce', 'dos']`.
2. **Điểm báo động**: `score = 1 − p[benign]`.
3. **Báo động** khi `score ≥ threshold`. Ngưỡng lấy từ `deploy_config.json`; với run `20261111_0925` là 0,9314 (run `20261105_1550`: 0,8821).
4. **Lớp tấn công** = lớp có xác suất lớn nhất trong 3 lớp tấn công (bỏ benign). Vì vậy một cửa sổ có thể bị báo động
   ngay cả khi không lớp tấn công nào vượt 0,5.
5. **Gộp đợt (episode)**: các cửa sổ báo động của **cùng host** cách nhau ≤ `episode_gap_s = 60` s gộp thành một đợt.
   Lớp của đợt là lớp xuất hiện nhiều nhất trong các cửa sổ của đợt đó.

---

## 8. Phía train: dữ liệu và nhãn

### Nguồn dữ liệu

| Notebook | Vai trò |
|---|---|
| `01` | Pcap lab (attacker `192.168.1.236` → mục tiêu `192.168.1.182`): gán nhãn **từng gói** `attack` / `aux` theo luật của từng kỹ thuật recon/SSH |
| `02` | Nền benign UNSW IoT 2016 (20 ngày): đổi IP/MAC cho khớp mạng lab, chia ngày train/val/test 12/4/4 |
| `01c` | DoS TII-SSRC-23 (hping3, một máy IP thật): ICMP/SYN/UDP/ACK/FIN/RST flood. Đợt ≥ 100 gói/s, gom thành phiên (= task), chia theo thời gian. Kèm benign TII (video/audio/text) làm nền khó |
| `01d` | Nền khó thêm (từ v5): pcap app tiêu dùng QoS (tách gói gộp > MTU, gỡ NAT chiều về, đổi dải LAN, chia theo lần bắt) và pcap ZeroTier (split theo thư mục, train có bản tăng cường) |
| `03b` | Trộn mỗi task tấn công vào một đoạn nền cùng split, cắt cửa sổ, tính đặc trưng, gán nhãn, trọng số |
| `04b` | Train `HistGradientBoostingClassifier`, chọn số vòng theo val loss, chọn ngưỡng, đánh giá test |
| `05` | Xuất ONNX và `deploy_config.json`; kiểm tra ONNX khớp sklearn |

### Trộn (03b)

Mỗi task được đặt vào một đoạn nền UNSW, đệm `PAD = 300` s trước và sau tấn công. Với xác suất 80%, một client UNSW
được đổi thành IP attacker (persona: máy tấn công cũng có traffic thường), và một camera được đổi thành IP mục tiêu.

### Nhãn cửa sổ

| Nhãn | Điều kiện | Dùng train? |
|---|---|---|
| `recon` / `bruteforce` / `dos` | cửa sổ của attacker chứa ≥ 1 gói `attack` | có |
| `post_attack` | cửa sổ attacker không còn gói attack, nhưng có gói attack trong 300 s trước | không |
| `victim` | cửa sổ của mục tiêu, có gói attack trong 300 s trước | không |
| `benign` | còn lại (cả cửa sổ nền UNSW/TII) | có |

### Tăng cường DoS (03b, `DOS_AUG = 3`)

Mỗi task DoS có thêm 3 bản biến đổi, cùng split với task gốc, trộn vào ngày nền khác:

- **Phản hồi nạn nhân**: giữ 0% (xác suất 50%), 0–50% (25%) hoặc 100% (25%) số gói trả lời.
- **Tốc độ** (40%): lấy mẫu con xuống 200–3000 gói/s, nếu task nhanh hơn mức đó.
- **Port đích** (50%): rải ra 2–10 port cố định. Riêng UDP có thể mỗi gói một port ngẫu nhiên.
- **Kích thước gói** (50%): đặt thành một hằng số từ 54 B (TCP) hoặc 42 B (ICMP/UDP) đến 1500 B.

Lý do: dữ liệu DoS gốc chỉ có hping3 (1 port, kích thước cố định, nạn nhân luôn trả lời). Không có bước tăng cường
thì model học các lối tắt đó và trượt flood thật.

### Trọng số và ngưỡng

- Mỗi task có tổng trọng số 1. Mỗi kịch bản có tổng tỉ lệ với √(số cửa sổ). Mỗi lớp có tổng trọng số bằng nhau.
  Benign TII chiếm 10% trọng số lớp benign. Không cửa sổ nào vượt 0,1% tổng trọng số.
- **Ngưỡng** chọn trên các cửa sổ nền UNSW của val sao cho báo động giả ≤ 0,2 cửa sổ/giờ.

---

## 9. Triển khai trên router: trạng thái cần giữ

Mọi đặc trưng tính được **trực tuyến**, không cần lưu lại pcap:

| Đặc trưng | Trạng thái cần giữ cho mỗi host LAN |
|---|---|
| Trong cửa sổ | bộ đệm các gói tx của cửa sổ đang mở (≤ 50 gói: thời gian, kích thước, cờ, đích, port) + bộ đếm rx từ `t_first` |
| Tỉ lệ "mới" | bảng lần gửi gần nhất theo (host, đích), (host, đích, port), (host, port); xoá mục cũ hơn 300 s |
| Ngữ cảnh 60/300 s | bộ đếm tx/rx theo từng loại, chia khối thời gian (vd khối 1 s) trong 300 s gần nhất |
| Số phần tử khác nhau 60/300 s | tập phần tử kèm thời điểm gặp gần nhất, hết hạn sau 60/300 s |
| Gộp đợt | thời điểm báo động gần nhất và lớp của đợt đang mở |

Mỗi khi một cửa sổ chốt: tính 97 số → chạy ONNX một lần → so ngưỡng → cập nhật đợt.

---

## 10. Phạm vi và giới hạn

- **Đánh giá theo host gửi.** Host được chấm dựa trên gói nó gửi đi. DDoS hoặc flood giả IP nguồn (mỗi nguồn chỉ vài
  gói) nằm ngoài phạm vi.
- **HTTP flood** ngoài phạm vi, thường bị báo là `bruteforce`.
- **Quét tốc độ rất cao** (~15.000 gói/s) có thể bị gắn lớp `dos` thay vì `recon`; vẫn có báo động.
- **Máy nạn nhân** của một trận quét hoặc flood đôi khi bị báo động, vì nó gửi lại hàng loạt RST/SYN-ACK.
- **Đồng bộ code**: sửa đặc trưng thì phải sửa cả `pkt_features.py` lẫn 03b cell 5, chạy lại `03b → 04b → 05`,
  và dùng đúng `pkt_features.py` của run đó khi inference.
