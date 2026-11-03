# Brute-force vs Recon — IDS theo cửa sổ gói tin trên router

Phát hiện **trinh sát mạng (recon: quét port, host discovery)** và **SSH brute-force** từ traffic
mà router OpenWrt nhìn thấy trên `br-lan`, phân biệt với traffic bình thường (benign).
Model nhẹ (HistGradientBoosting → ONNX, ~500 KB) để chạy trên router.

> Repo chỉ chứa code. Dữ liệu pcap, dữ liệu đã xử lý và model **không** được đưa lên
> (xem `.gitignore`); chạy lại pipeline để tạo chúng.

## Ý tưởng

- **Dữ liệu tấn công:** pcap bắt bằng tcpdump trên router lab. Attacker là laptop `192.168.1.236`,
  mục tiêu là Raspberry Pi `192.168.1.182`. Mỗi gói được gán nhãn theo luật IP/MAC/port/cờ của
  từng kỹ thuật.
- **Nền benign:** UNSW IoT traces 2016 (20 ngày traffic LAN). Đổi IP/MAC cho khớp mạng lab rồi
  trộn với gói tấn công ở mức bảng gói, theo timestamp.
- **Chia tập trước khi trộn:** tấn công chia theo task (70/15/15), nền chia theo ngày (12/4/4).
  Nhờ vậy val/test không bị rò dữ liệu.
- **Cửa sổ lai theo host nguồn:** một cửa sổ đóng khi đủ **50 gói** hoặc hết **10 s**.
  Mỗi cửa sổ có thêm đặc trưng ngữ cảnh **60 s / 300 s** của cùng host, tổng cộng **97 đặc trưng**.
  Đặc trưng không dùng IP, MAC hay TTL tuyệt đối.
- **3 lớp:** `benign` / `recon` / `bruteforce`. Điểm báo động `score = 1 − p(benign)`.
  Ngưỡng chọn trên val sao cho báo động giả ≤ mục tiêu mỗi giờ.
  Các cửa sổ báo động của cùng host, cách nhau ≤ 60 s, được gộp thành một đợt (episode).

## Cấu trúc

```
Code/
├── predata/
│   ├── 01_attack_packet_labeling.ipynb    # kiểm kê task, tách gói bằng tshark, gán nhãn từng gói
│   ├── 02_benign_unsw_packets.ipynb       # tách gói UNSW, đổi IP/MAC, chia ngày train/val/test
│   ├── 03_mix_and_window_features.ipynb   # (v1) trộn + cửa sổ thời gian 10/60/300 s
│   └── 03b_mix_and_packet_windows.ipynb   # (v2) trộn + cửa sổ lai 50 gói / 10 s + ngữ cảnh 60/300 s
├── train/
│   ├── 04_train_eval.ipynb                # train/đánh giá trên cửa sổ thời gian (v1)
│   └── 04b_train_eval_packet_windows.ipynb# train HGB trên cửa sổ gói, chọn ngưỡng theo FP/giờ, đánh giá test
└── deploy/
    ├── pkt_features.py                    # đọc pcap (tshark) + trích đặc trưng cửa sổ, giống hệt 03b
    ├── 05_export_onnx.py                  # xuất model.onnx + deploy_config.json, kiểm tra ONNX khớp sklearn
    └── test_bruteforce.py                 # chạy model ONNX trên pcap bất kỳ, in đợt báo động theo host
```

Thư mục được tạo khi chạy, không có trong repo:

| Thư mục | Nội dung |
|---|---|
| `Data/real_data/` | pcap tấn công lab (`brute_force/ssh`, `recon/*`) |
| `Data/public/unsw_iot2016/pcap/` | nền benign UNSW IoT 2016 |
| `Data_after_processing/` | bảng gói `.parquet`, cửa sổ `windows_pkt/{train,val,test}`, báo cáo `reports/` |
| `Model/hgb_pkt/<YYYYMMDD_HHMM>/` | `model.joblib`, `model.onnx`, `deploy_config.json`, chỉ số và biểu đồ; `runs.csv` ghi lịch sử các lần train |

## Cài đặt

- Python 3.10, cài thư viện bằng `pip install -r requirements.txt`.
- Cần `tshark` (Wireshark ≥ 3.6) trong `PATH`.
- Đường dẫn gốc `ROOT` đang được ghi cứng trong notebook và script
  (`/home/ubuntu/sepcung/03.Brute_Force_vs_Recon`). Sửa lại nếu clone repo ra chỗ khác.

## Chạy

1. Chạy lần lượt các notebook `01` → `02` → `03b` → `04b`.
   Notebook `04b` ghi kết quả vào `Model/hgb_pkt/<run>/` và thêm một dòng vào `runs.csv`.
2. Xuất ONNX:
   ```bash
   python Code/deploy/05_export_onnx.py                 # mặc định lấy run mới nhất trong runs.csv
   python Code/deploy/05_export_onnx.py --run 20261102_1532
   ```
3. Test trên pcap:
   ```bash
   # các task SSH lab thuộc split test, kỳ vọng phát hiện brute-force
   python Code/deploy/test_bruteforce.py Data/real_data/brute_force/ssh/pcaps --split test --expect bruteforce --lab-clean

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

- **Báo nhầm máy nạn nhân.** Máy bị quét hoặc bị brute-force gửi lại hàng loạt RST/SYN-ACK,
  trông giống một máy đang quét, nên đôi khi bị báo là `recon`. Nguyên nhân là cửa sổ `victim`
  chưa được dùng khi train. Hướng sửa: train với cửa sổ `victim` gán nhãn benign, thêm đặc trưng
  "ai khởi tạo kết nối", hoặc lọc ở bước gộp cảnh báo.
- Thiết bị client có traffic dồn dập hoặc nhiều gói ACK thuần (ví dụ tải HTTPS lớn) có thể
  gây báo động giả `bruteforce`.
- Một số loại chưa có hoặc có ít dữ liệu train, nên phát hiện còn yếu: quét XMAS,
  OS fingerprinting, brute-force qua giao thức không phải TCP (DNS).
