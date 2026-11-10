# Brute-force vs Recon — IDS theo cửa sổ gói tin trên router

Phát hiện **trinh sát mạng (recon: quét port, host discovery)**, **SSH brute-force** và **DoS flood từ một máy (ICMP/SYN/UDP/ACK/FIN/RST…)** từ traffic
mà router OpenWrt nhìn thấy trên `br-lan`, phân biệt với traffic bình thường (benign).
Model nhẹ (HistGradientBoosting → ONNX, ~500 KB) để chạy trên router.

> Repo chỉ chứa code. Dữ liệu pcap, dữ liệu đã xử lý và model **không** được đưa lên
> (xem `.gitignore`); chạy lại pipeline để tạo chúng.

## Ý tưởng

- **Dữ liệu tấn công:** pcap bắt bằng tcpdump trên router lab. Attacker là laptop `192.168.1.236`,
  mục tiêu là Raspberry Pi `192.168.1.182`. Mỗi gói được gán nhãn theo luật IP/MAC/port/cờ của
  từng kỹ thuật.
- **Dữ liệu DoS:** TII-SSRC-23 (hping3, một máy tấn công IP thật): ICMP/SYN/UDP/ACK/FIN/RST flood.
  Mỗi phiên flood là một task, chỉ giữ đợt ≥ 100 gói/s, đổi địa chỉ sang attacker/target của lab rồi trộn
  vào nền như task lab. Benign của TII (video/audio/text, cùng cặp máy) làm nền "khó" chống báo nhầm.
  CWR/ECN/PSH/URG flood không dùng khi train (để test loại flood chưa thấy); HTTP flood ngoài phạm vi.
- **Tăng cường DoS (từ v4, `DOS_AUG` trong 03b):** mỗi task DoS thêm 3 bản biến đổi — bỏ phản hồi của nạn nhân,
  hạ tốc độ (200–3000 gói/s), rải port đích (TCP 2–10 port, UDP có thể mỗi gói một port), đổi kích thước gói
  (54/42–1500 B). Không có bước này model học lối tắt của hping3/TII ("nạn nhân luôn trả lời", "1 port",
  "gói 104 B") và trượt flood thật (ICMP/SYN flood tự bắt, UDP flood Mirai).
- **Nền benign:** UNSW IoT traces 2016 (20 ngày traffic LAN). Đổi IP/MAC cho khớp mạng lab rồi
  trộn với gói tấn công ở mức bảng gói, theo timestamp.
- **Chia tập trước khi trộn:** tấn công chia theo task (70/15/15), nền chia theo ngày (12/4/4).
  Nhờ vậy val/test không bị rò dữ liệu.
- **Cửa sổ lai theo host nguồn:** một cửa sổ đóng khi đủ **50 gói** hoặc hết **10 s**.
  Mỗi cửa sổ có thêm đặc trưng ngữ cảnh **60 s / 300 s** của cùng host, tổng cộng **97 đặc trưng**.
  Đặc trưng không dùng IP, MAC hay TTL tuyệt đối.
- **4 lớp, một model:** `benign` / `recon` / `bruteforce` / `dos` (từ v3; run `20261102_1532` là bản 3 lớp chưa có `dos`). Điểm báo động `score = 1 − p(benign)`.
  Ngưỡng chọn trên val sao cho báo động giả ≤ mục tiêu mỗi giờ.
  Các cửa sổ báo động của cùng host, cách nhau ≤ 60 s, được gộp thành một đợt (episode).

## Cấu trúc

```
Code/
├── predata/
│   ├── 01_attack_packet_labeling.ipynb    # kiểm kê task, tách gói bằng tshark, gán nhãn từng gói
│   ├── 02_benign_unsw_packets.ipynb       # tách gói UNSW, đổi IP/MAC, chia ngày train/val/test
│   ├── 01c_dos_tii_packet_labeling.ipynb  # DoS TII-SSRC-23: đợt -> phiên (task), gán nhãn, chia theo thời gian; benign TII
│   ├── 03_mix_and_window_features.ipynb   # (v1) trộn + cửa sổ thời gian 10/60/300 s
│   └── 03b_mix_and_packet_windows.ipynb   # (v2) trộn + cửa sổ lai 50 gói / 10 s + ngữ cảnh 60/300 s
├── train/
│   ├── 04_train_eval.ipynb                # train/đánh giá trên cửa sổ thời gian (v1)
│   └── 04b_train_eval_packet_windows.ipynb# train HGB trên cửa sổ gói, chọn ngưỡng theo FP/giờ, đánh giá test
└── deploy/
    ├── feature_pipeline_bruteforce_recon_dos.md  # mô tả chi tiết luồng xử lý và cách tính 97 đặc trưng
    ├── pkt_features.py                    # đọc pcap (tshark) + trích đặc trưng cửa sổ, giống hệt 03b
    ├── 05_export_onnx.ipynb               # xuất model.onnx + deploy_config.json, kiểm tra ONNX khớp sklearn
    └── test_bruteforce.py                 # chạy model ONNX trên pcap bất kỳ, in đợt báo động theo host
```

Thư mục được tạo khi chạy, không có trong repo:

| Thư mục | Nội dung |
|---|---|
| `Data/real_data/` | pcap tấn công lab (`brute_force/ssh`, `recon/*`) |
| `Data/public/unsw_iot2016/pcap/` | nền benign UNSW IoT 2016 |
| TII-SSRC-23 `pcap/{malicious/dos,benign}` | DoS và benign TII (đường dẫn `TII` trong 01c) |
| `Data_after_processing/` | bảng gói `.parquet` (`attack_packets/`, `benign_packets/`, `benign_tii/`), cửa sổ `windows_pkt/{train,val,test}`, báo cáo `reports/` |
| `Model/hgb_pkt/<YYYYMMDD_HHMM>/` | `model.joblib`, `model.onnx`, `deploy_config.json`, chỉ số và biểu đồ; `runs.csv` ghi lịch sử các lần train |

## Cài đặt

- Python 3.10, cài thư viện bằng `pip install -r requirements.txt`.
- Cần `tshark` (Wireshark ≥ 3.6) trong `PATH`.
- Đường dẫn gốc `ROOT` đang được ghi cứng trong notebook và script
  (`/home/ubuntu/sepcung/03.Brute_Force_vs_Recon`). Sửa lại nếu clone repo ra chỗ khác.

## Chạy

1. Chạy lần lượt các notebook `01` → `02` → `01c` → `03b` → `04b`.
   Notebook `04b` ghi kết quả vào `Model/hgb_pkt/<run>/` và thêm một dòng vào `runs.csv`.
2. Xuất ONNX:
   chạy notebook `Code/deploy/05_export_onnx.ipynb` (biến `RUN = None` lấy run mới nhất trong `runs.csv`,
   hoặc đặt `RUN = '20261102_1532'`).
3. Test trên pcap:
   ```bash
   # các task SSH lab thuộc split test, kỳ vọng phát hiện brute-force
   python Code/deploy/test_bruteforce.py Data/real_data/brute_force/ssh/pcaps --split test --expect bruteforce --lab-clean

   # flood DoS (model 4 lớp)
   python Code/deploy/test_bruteforce.py syn_flood.pcap --expect dos

   # pcap tự bắt hoặc pcap public: chỉ xét một số host, đổi tiền tố LAN, lưu chi tiết
   python Code/deploy/test_bruteforce.py capture.pcap --host 192.168.1.50 --out-dir out/
   python Code/deploy/test_bruteforce.py cic_iot_2022/ --lan-prefix 192.168.137. --expect benign
   ```
   Kết quả trong `--out-dir` gồm `summary.csv` (mỗi pcap một dòng),
   `windows.csv` (score từng cửa sổ) và `episodes.csv` (các đợt báo động).

## Kết quả (run `20261102_1532`)

Ngưỡng 0,747, chọn sao cho FP ≤ 0,2/giờ trên val. Kết quả trên tập test:

| Chỉ số | Giá trị |
|---|---|
| Macro F1 (3 lớp) | 0,994 |
| Tỷ lệ task tấn công được phát hiện | 98,8 % |
| Báo động giả | 1,77 cửa sổ/giờ · 0,42 đợt/giờ |
| ONNX so với sklearn | lệch xác suất < 1e-6, 0 quyết định khác nhau |

Thử trên dữ liệu public không dùng khi train:

| Bộ dữ liệu | Phát hiện |
|---|---|
| TII-SSRC-23 | SSH / FTP / Telnet brute-force: 100 % · HTTP brute-force: 94 % |
| Edge-IIoTset | HTTP brute-force: 100 % · Port scanning: 100 % · OS fingerprinting: ~51 % |
| Gotham 2025 | recon: 100 % |
| TII-SSRC-23 | recon: 79 % cửa sổ của attacker (quét XMAS chỉ 52 %) · DNS brute-force: bỏ sót |

## Hạn chế đã biết

- **Phạm vi DoS: một máy tấn công dùng IP thật.** DDoS / flood giả IP nguồn (`hping3 --rand-source`, botnet)
  không thuộc phạm vi — mỗi nguồn chỉ có vài gói nên cửa sổ theo host gửi không thấy. HTTP flood, Slowloris
  và gói dị dạng (ping of death, teardrop) cũng ngoài phạm vi; HTTP flood thường bị báo là `bruteforce`.
- **Quét tốc độ rất cao** (~15.000 gói/s, vd quét FIN/RST/fragment của TII) bị báo là `dos` thay vì `recon`
  — vẫn có báo động, chỉ sai lớp; ở tốc độ đó quét và flood gần như không phân biệt được.
- **Flood lẫn traffic dày khác của cùng máy** (vd ICMP flood ~4.000 gói/s kèm ~500 gói/s TCP) có thể bị báo
  `recon`/`bruteforce` thay vì `dos`: persona benign của máy tấn công khi train chỉ có traffic thưa.
- Đã sửa (v4): gói TCP không có cờ (mảnh IP, vd hping3 payload lớn) trước đây bị đếm thành SYN-ACK + RST + XMAS
  trong `packet_table`; model train trước bản sửa phải dùng `pkt_features.py` cũ thì mới khớp.

- **Báo nhầm máy nạn nhân.** Máy bị quét hoặc bị brute-force gửi lại hàng loạt RST/SYN-ACK,
  trông giống một máy đang quét, nên đôi khi bị báo là `recon`. Nguyên nhân là cửa sổ `victim`
  chưa được dùng khi train. Hướng sửa: train với cửa sổ `victim` gán nhãn benign, thêm đặc trưng
  "ai khởi tạo kết nối", hoặc lọc ở bước gộp cảnh báo.
- Thiết bị client có traffic dồn dập hoặc nhiều gói ACK thuần (ví dụ tải HTTPS lớn) có thể
  gây báo động giả `bruteforce`.
- Một số loại chưa có hoặc có ít dữ liệu train, nên phát hiện còn yếu: quét XMAS,
  OS fingerprinting, brute-force qua giao thức không phải TCP (DNS).
