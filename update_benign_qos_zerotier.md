# Update: thêm nền benign QoS và ZeroTier

Mục tiêu: giảm báo nhầm trên traffic thật mà nền UNSW/TII chưa có.

Trạng thái (11/11/2026):
- **v5, run `20261111_0925`:** đã train. Phần QoS xong (kết quả ở cuối file); ZeroTier vẫn bị báo RECON vì chưa có dữ liệu train.
- **v5b, chưa train:** đã chạy lại `01d` và `03b`, còn phải chạy `04b` → `05`. Có ba thay đổi:
  1. thêm ZeroTier mô phỏng cho train/val (mục 3b của `01d`, `ZT_SIM`); `pc71.pcap` vẫn ở test;
  2. tăng cường DoS thêm flood ICMP/UDP payload lớn bị phân mảnh IP (`augment_dos` trong `03b`);
  3. ONNX: ngưỡng báo động cách điểm cửa sổ val một khoảng ≤ 1e-4 (`pick_threshold` trong `04b`), và ngưỡng tách của cây
     được làm tròn xuống float32 khi xuất (`05`), vì skl2onnx không tự chỉnh ngưỡng cho HGB.

## Vấn đề

Hai loại traffic thật bị model `20261105_1550` báo nhầm:

- **ZeroTier bị báo `recon`.** Khi tìm đường tới peer (hole punching), ZeroTier gửi dồn dập gói UDP ~70 B
  tới hàng chục IP, mỗi IP 1–2 port, hầu như không có phản hồi. Mẫu này trông giống `udp_port_scan`.
  Bỏ các gói ZeroTier khỏi pcap thì hết báo động.
- **Gọi video, stream, tải file bị báo `dos`:** Teams ~16 %, Meet ~9 %, Google Drive ~9 % số cửa sổ.

## Dữ liệu

- `Data/pcap_data_QoS`: traffic app trên PC/iOS/Android (gọi thoại/video, video, game, tải file, chạy nền),
  mỗi app bắt 5 lần. Sau khi bỏ 63 file hỏng hoàn toàn còn 367 file, khoảng 14,6 giờ. 25 file bị cụt vẫn giữ
  phần đọc được. Có hai kiểu lỗi do cách bắt:
  - dải `192.168.11.` / `192.168.5.` bắt sau 2 lớp NAT, nên gói chiều về có đích là IP WAN của NAT thay vì IP máy.
    Phải **gỡ NAT** (ghép IP và port phía xa với máy đã gửi tới đó), nếu không model học rằng
    "flood không có phản hồi là benign";
  - dải `192.168.239.` bắt ngay trên máy, nên có gói TCP gộp dài hơn MTU (GRO/TSO, tới 7,8 % số gói ở Android).
    Phải **tách lại** thành gói ≤ 1514 B như router thấy.
- ZeroTier: để pcap trong `Data/pcap_zerotier/{train,val,test}/`; thư mục quyết định split. Hiện chỉ có
  `test/pc71.pcap` (17 s, 13 cửa sổ). Cần bắt thêm vài giờ trên 2–3 máy, gồm các trường hợp: ZeroTier chạy không,
  khởi động lại dịch vụ, đổi mạng, SSH/scp qua ZeroTier. Nên chia train/val theo máy hoặc theo trường hợp.

## Các bước

1. **Xong — notebook `Code/predata/01d_benign_extra.ipynb`:** đọc pcap QoS và ZeroTier.
   - QoS: tách gói gộp, gỡ NAT (gán lại được 98,3 % gói chiều về), đổi dải LAN về `192.168.1.x`.
     Chia theo lần bắt: lần 1–3 vào train, lần 4 vào val, lần 5 vào test.
   - ZeroTier: mỗi file train thêm 5 bản tăng cường. Mỗi bản đổi số peer, số port cục bộ, tỉ lệ phản hồi và
     nhịp đợt (đặc trưng không dùng IP/port cụ thể, nên chỉ đổi địa chỉ thì không tạo ra mẫu mới).
   - Ghi parquet vào `Data_after_processing/benign_{qos,zt}/` và `reports/01d_extra_parts.csv`.
2. **Xong — sửa `03b`:** thêm nguồn nền `benign_qos` và `benign_zt` giống `benign_tii`. Mỗi nguồn chiếm một phần
   cố định trong trọng số lớp benign (`HARD_BENIGN_SHARE`: TII 10 %, QoS 10 %, ZeroTier 5 %).
3. **Xong — train lại** `03b` → `04b` → `05_export_onnx`, ra run `20261111_0925`.
   `04b` có thêm bảng báo nhầm theo nguồn, loại traffic và từng file.
4. **Đang làm — ZeroTier mô phỏng (v5b):** 360 mẫu (300 train, 60 val; khoảng 77 nghìn cửa sổ train) mô phỏng
   keepalive tới root, dò đường tới nhiều địa chỉ của peer từ 1–3 port với gói dò 70 B, có hoặc không có phiên dữ liệu.
   Model v5 báo nhầm khoảng 15% cửa sổ dò đường mô phỏng, tức mô phỏng có chạm đúng kiểu gây lỗi.
   **Lưu ý:** mô phỏng được đối chiếu với thống kê của `pc71.pcap` khi viết, nên `pc71` không còn là bài test độc lập
   hoàn toàn; kiểm tra thật là chạy trên router. Đạt khi `pc71.pcap` có ≤ 1 cửa sổ báo nhầm, còn recall recon/DoS
   (nhất là `udp_port_scan`, `dos_udp`) giảm không quá 1 điểm %.
5. **Tốt nhất vẫn là pcap ZeroTier thật:** bắt pcap, để vào `Data/pcap_zerotier/train/` (và `val/`), giảm hoặc tắt
   `ZT_SIM`, rồi chạy lại `01d` → `03b` → `04b` → `05`.
6. **Nếu bước 4–5 chưa đạt:** thêm đặc trưng "tỉ lệ gói tới đích trong LAN" và "số port nguồn khác nhau".
   Phải sửa cả `pkt_features.py`, `03b` và code trích đặc trưng trên router.

## Kết quả run `20261111_1058` (v5b)

Chọn ngưỡng theo cửa sổ (0,2 cửa sổ/giờ val) ra 0,993, vì một máy UNSW upload dài (`.237`, val) có điểm 0,998–0,999.
Ngưỡng đó làm quét thật trên router (điểm 0,75–0,97) không bị báo. Đã đổi sang chọn theo **đợt** báo nhầm
(0,15 đợt/giờ val) thì ngưỡng là **0,690**. Kết quả test ở ngưỡng 0,690:

- báo nhầm nền UNSW 1,36 cửa sổ/giờ, 0,33 đợt/giờ;
- phát hiện task 98,6 %, recall cửa sổ recon 99,9 %;
- QoS 0,02 %; `pc71.pcap` 0/13 cửa sổ báo nhầm;
- ONNX khớp sklearn ở mọi quyết định (lệch xác suất ≤ 6e-7).

`runs.csv` vẫn ghi chỉ số ở ngưỡng cũ 0,993. Ngưỡng cũ được giữ trong `model.joblib` / `run_config.json` ở khoá `threshold_window_rule`.

## Kết quả run `20261111_0925` (v5)

Hai model được chấm trên cùng tập test mới (có thêm QoS test và `pc71.pcap`):

| Chỉ số (test) | `20261105_1550` (cũ) | `20261111_0925` (mới) |
|---|---|---|
| Ngưỡng (FP ≤ 0,2/giờ trên val) | 0,882 | 0,931 |
| Báo nhầm trên nền UNSW | 0,41 cửa sổ/giờ · 0,15 đợt/giờ | 0,16 cửa sổ/giờ · 0,09 đợt/giờ |
| Báo nhầm trên QoS | 2,8 % số cửa sổ (videocall 8,7 %), gần như toàn `dos` | 0 % |
| Báo nhầm trên TII | 0 % | 0 % |
| `pc71.pcap` (ZeroTier) | 9/13 cửa sổ báo `recon` | 7/13 cửa sổ báo `recon` (chưa có dữ liệu train) |
| Phát hiện task tấn công | 98,6 % | 98,6 % |
| Recall cửa sổ DoS | 92,7 % | 92,5 % |
| Macro F1 (cửa sổ) | 0,983 | 0,984 |

- `dos_icmp` giảm nhiều nhất. Cả 4 mẫu test vẫn được phát hiện, nhưng recall cửa sổ giảm từ 0,69 xuống 0,55 và
  độ trễ trung vị tăng từ 4 s lên 21 s. Trên val, `dos_icmp` vẫn đạt 1,000. Task ICMP ở test dùng payload rất lớn, nên
  gần như toàn mảnh IP: chỉ khoảng 2% gói có header ICMP echo. Train lúc đó chỉ có 1 task dạng này, và model lại vừa học
  các đợt upload gói lớn không có phản hồi của QoS là benign. v5b thêm tăng cường mảnh IP để sửa chỗ này.
- Ngưỡng tăng từ 0,882 lên 0,931 vì vài máy UNSW (`.237`, `.106`, ngày 16-10-07 thuộc val) gửi dồn dập tới 1 đích mà
  không có phản hồi, và model mới chấm các cửa sổ đó giống DoS hơn.
- ONNX so với sklearn: chỉ 2 cửa sổ mỗi tập lệch xác suất quá 1e-4 (tối đa 0,008), do ngưỡng cây rất lớn (byte trong
  300 s) bị làm tròn sang float32. Có 1 cửa sổ val khác quyết định báo động, nhưng đó là cửa sổ dùng để chọn ngưỡng
  (điểm bằng đúng ngưỡng). Trên test không có quyết định nào khác.

## Rủi ro

Sau khi học ZeroTier, model có thể bỏ sót quét ngang UDP thật (một port trên nhiều máy).
Đặc trưng ở bước 5 giúp tách hai trường hợp này, vì ZeroTier chủ yếu gửi ra ngoài LAN.
