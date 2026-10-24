# QUICKSTART — วิธีรันโปรเจกต์

คู่มือนี้สมมติว่าคุณอยู่ที่โฟลเดอร์ `football` แล้ว

---

## 0. Python ที่เครื่องนี้

`python` เฉย ๆ จะชน Windows Store alias ให้ใช้ `run.ps1` ที่เตรียมไว้ (มันหา Python ให้เอง)
หรือเรียกตรง ๆ ด้วย path นี้:

```
C:\Users\sitti\AppData\Local\Programs\Python\Python313\python.exe
```

---

## 1. รัน demo ทันที (ไม่ต้องมี API key)

ทดสอบว่า pipeline ครบวงจรทำงานได้:

```powershell
.\run.ps1 demo
```

หรือถ้าไม่อยากใช้สคริปต์:

```powershell
C:\Users\sitti\AppData\Local\Programs\Python\Python313\python.exe -m fbpredict demo
```

จะได้:

- `data/synthetic_matches.csv` — ข้อมูล**สังเคราะห์** 220 นัด
- `data/demo.db` — SQLite **แยกไฟล์ต่างหาก** (ไม่ปนกับ `data/football.db` ที่เก็บข้อมูลจริง)
- `reports/demo_backtest.md` — รายงานผล

> ⚠️ ข้อมูลชุดนี้ **ไม่ใช่หลักฐาน** ทุกแถวติด `is_synthetic = 1` และ provider เป็น `synthetic`
> ห้ามส่งเป็นผลการดึงข้อมูลจริงเด็ดขาด — ใช้เพื่อทดสอบโค้ดเท่านั้น
> และ `backtest` จะไม่ดึงข้อมูลชุดนี้มารวม เว้นแต่สั่ง `--include-synthetic`

---

## 2. รันเทสต์

```powershell
.\run.ps1 test
```

หรือ

```powershell
C:\Users\sitti\AppData\Local\Programs\Python\Python313\python.exe -m pytest -q
```

ตอนนี้มี **211 เทสต์** ครอบคลุม: สูตรคิดผลตาม README §4, การกัน data leakage ของ rolling,
look-ahead guard ของ DB, กฎเลือกตลาด/ข้ามนัด, walk-forward, การ hash-lock โมเดล,
กฎราคาพูลชุดที่สอง (threshold / no-vig / audit), feature set v1-v2-v2_novig,
และ guard ที่กันการใช้ราคาปิดผิดเวลา

---

## 3. ใส่ API key จริง

```powershell
Copy-Item .env.example .env
notepad .env
```

ใส่คีย์ที่คุณมี (มีอันไหนใส่ อันนั้น):

| ตัวแปร | ใช้ทำอะไร | สมัคร |
|---|---|---|
| `FIVE_DOLLAR_FOOTBALL_API_KEY` | **ราคาเปิด + ผล + สกอร์ HT (แนะนำ)** | https://5dollarfootballapi.com/signup |
| `FOOTBALL_DATA_ORG_KEY` | ผลแข่ง (label) | https://www.football-data.org/client/register |
| `API_FOOTBALL_KEY` | ผลแข่ง (label) | https://www.api-football.com/ |
| `ODDSPAPI_KEY` | ราคาเปิด AH / 1X2 / O-U | https://oddspapi.io/ |

เช็คว่าอ่านค่าได้:

```powershell
.\run.ps1 show-config
```

---

## 4. ลำดับงานตาม README §9

### ทางลัด (แนะนำ) — 5DollarFootballAPI ทำทุกอย่างจบในตัว

API นี้ให้ **ราคาเปิด + ผล + สกอร์ HT** ในคำขอเดียว → **ไม่ต้อง join ราคากับผลเอง**

```powershell
# 1) ดูว่าแผนอะไร โควตาเหลือเท่าไร และ league id ที่ใช้ได้
.\run.ps1 probe-5dollar

# 2) ดู odds block จริงว่า map เข้าคอลัมน์เราถูกไหม
.\run.ps1 probe-5dollar --fixture-id 123456

# 3) ดึงทั้งฤดูกาล (มีราคา ถ้าอยู่ Pro ขึ้นไป)
.\run.ps1 fetch-5dollar --league 3120672213 --season 26/27

# 4) backtest ได้เลย
.\run.ps1 backtest --threshold 0.03
```

ถ้าอยู่แผนฟรี (`include=odds` บน list เป็นฟีเจอร์เสียเงิน) ให้ดึงราคาทีละนัด:

```powershell
.\run.ps1 fetch-5dollar --league 3120672213 --season 26/27 --per-fixture-odds
```

> ⚠️ แผนฟรีต้องใส่ attribution "Football data by 5DollarFootballAPI" ในรายงาน
> ดูรายละเอียดใน [`DATA_SOURCES.md`](DATA_SOURCES.md) ข้อ 0

---

### ทางเดิม — แยกแหล่งราคากับผล

### ขั้น 1 — ดึงผลแข่ง (label) มาก่อน

```powershell
.\run.ps1 fetch-results --provider football-data-org --competition PL --season 2025
```

หรือ

```powershell
.\run.ps1 fetch-results --provider api-football --league EPL --season 2025
```

ทุก response จะถูกเก็บดิบไว้ที่ `data/raw/<provider>/<วันที่>/...json`
และในตาราง `raw_payload` พร้อม `fetched_at` → **นี่คือหลักฐานการดึงข้อมูลเอง**

### ขั้น 2 — ตรวจ OddsPapi ว่าคืนอะไรจริง (README §9.1)

```powershell
.\run.ps1 probe-odds --fixture-id 1234567
```

จะพิมพ์โครงสร้าง response จริงที่ได้ + รายชื่อ collection ที่iterated ได้
**ห้ามเชื่อบทความโปรโมต** ต้องดู response จริงเท่านั้น

### ขั้น 3 — จับคู่ราคากับผล

```powershell
.\run.ps1 link --tolerance 36 --dry-run
.\run.ps1 link --tolerance 36
```

จะรายงานว่าจับได้กี่คู่, กี่คู่จับไม่ได้, กี่คู่กำกวม (ambiguous) — **ไม่เดาเงียบ ๆ**

### ขั้น 4 — backtest

```powershell
.\run.ps1 backtest --threshold 0.03
```

ถ้าต้องการเก็บ prediction ไว้ไป settle ทีหลัง:

```powershell
.\run.ps1 backtest --save-predictions
```

`backtest` จะไม่ดึงข้อมูลสังเคราะห์มารวมโดยอัตโนมัติ ถ้าต้องการให้ใส่ `--include-synthetic`

### ขั้น 4.5 — settle (ปิดวง plan B)

หลังผลแข่งเข้ามาแล้ว เติม `outcome_unit` / `profit` ให้ run ที่บันทึกไว้:

```powershell
.\run.ps1 settle --run-id bt-xxxxxxxxxx
```

นัดที่ยังไม่มีผลจะถูกข้าม (ไม่เดา) และรายงานว่ายังเหลือกี่นัด

### ขั้น 5 (แผน B) — ล็อกโมเดลแล้วทำ forward test

```powershell
.\run.ps1 predict-forward --model models\model_v1.pkl
.\run.ps1 verify-model --model models\model_v1.pkl
```

---

## 4.5 กฎราคาพูลชุดที่สอง + สองโหมดเวลา (feature set v1 / v2 / v2_novig)

### วัดผลกฎก่อนเทรน

```powershell
.\run.ps1 audit-rules            # ตาราง hit rate ต่อกฎ R1–R6 แยกตามตลาด
.\run.ps1 audit-rules --novig    # เทียบในราคาไร้ vig (ทดสอบความทนต่อ margin)
.\run.ps1 audit-rules --water    # วัดกฎ R7 เดี่ยว ๆ (ราคาน้ำ 0.5/1 บนตลาดทีมต่อ)
```

> ⚠️ `audit-rules` เป็น **คำบรรยาย in-sample** — กฎและข้อมูลมาจากคนเดียวกัน ช่วงเดียวกัน
> ไม่ใช่หลักฐาน edge ให้อ่าน 95% CI เทียบ break-even ของกฎนั้น และ n < 30 = สรุปไม่ได้
> กฎ R4a (Ladbrokes Over HT) ถูกปล่อยเป็น NaN เพราะแผนเราไม่มีฟีดของ Ladbrokes
> รายงานจะเขียนว่า "ทดสอบไม่ได้" — **ห้ามไปแทนด้วยเจ้ามืออื่นเงียบ ๆ** (สเปก §6.2)

### เลือก feature set ตอน backtest

| `--feature-set` | ใช้เมื่อ |
|---|---|
| `v1` (ค่าเริ่มต้น) | ชุดเดิม 30 feature ตาม brief |
| `v2` | + กฎ R1–R6 จากราคาที่พิมพ์ในภาพ (ราคาดิบ) |
| `v2_novig` | + กฎ R1–R6 หลังแปลงเป็นราคาไร้ vig |
| `water` | ใช้เฉพาะกฎใหม่ R7 (ราคาน้ำ 0.5/1) — กฎอื่นพักไว้ทั้งหมด |

**ห้ามรัน v2 กับ v2_novig รวมในโมเดลเดียว** — สเปก §1 ให้เก็บสองชุดแยกกัน
(ตารางชุดที่สองไม่ตรงกับ `tip.jpg` เช่น AH −0.75: 1.60–1.64 vs 1.72–1.89)

```powershell
.\run.ps1 backtest --require-odds --feature-set v2 --stem backtest_epl_v2
.\run.ps1 backtest --require-odds --feature-set v2_novig --stem backtest_epl_v2_novig
```

### สองโหมดเวลา (สำคัญเรื่อง data leakage)

ราคาปิด (closing) เกิดก่อนเตะไม่นาน → feature ที่พึ่งราคาปิด
(`fav_no_drop` / `dog_no_drop`) ใช้ได้เฉพาะเมื่อเวลาทำนายอยู่ใกล้เตะ:

| โหมด | คำสั่ง | ราคาปิด |
|---|---|---|
| ปกติ (24 ชม.) | `backtest ...` | **ถูกตัดทิ้ง** (เป็น NaN ทั้งคอลัมน์ → โมเดลไม่เห็น) |
| late (ก่อนเตะ 5 นาที) | `backtest ... --predicted-before-minutes 5 --use-closing-features` | ถูกเติมค่าและเข้าโมเดล |

ถ้าขอ `--use-closing-features` ทั้งที่เวลาทำนายยัง 24 ชม. (หรือห่างเกิน
15 นาทีจากเตะ) ระบบ **refuse ทันที** ไม่ใช่แค่เตือน — รันไม่ผ่านโดยเจตนา
(guard เดียวกันถูกใช้ตอน `predict-forward` ด้วย)

---

## 5. คำสั่งทั้งหมด

| คำสั่ง | ทำอะไร |
|---|---|
| `init-db` | สร้างไฟล์ SQLite |
| `show-config` | พิมพ์ค่าตั้งต้น + คีย์ที่ยังขาด |
| `demo` | รัน pipeline บนข้อมูลสังเคราะห์ (ลง `data/demo.db`) |
| `backtest` | walk-forward backtest (จาก DB หรือ `--csv`, `--save-predictions`, `--feature-set`, `--predicted-before-minutes`, `--use-closing-features`) |
| `audit-rules` | ตาราง hit rate ต่อกฎ R1–R6 (in-sample; `--novig`, `--water` สำหรับกฎ R7) |
| `fetch-results` | ดึงผลแข่งจาก football-data.org / api-football (`--refresh` เพื่อบังคับยิงใหม่) |
| `probe-5dollar` | ตรวจแผน/โควตา/league id + odds block จริงของ 5DollarFootballAPI |
| `fetch-5dollar` | ดึงราคาเปิด + ผล + สกอร์ HT จาก 5DollarFootballAPI (`--per-fixture-odds`, `--cache-only`) |
| `probe-odds` | ตรวจ response จริงของ OddsPapi |
| `scrape-thscore` | ดึง Thscore (ต้องใส่ `--i-understand-robots`) |
| `link` | จับคู่ราคา ↔ ผล ด้วยชื่อทีม + เวลาเตะ |
| `settle` | เติมผล/กำไรให้ run ที่บันทึกไว้ |
| `predict-forward` | ทำนายนัดที่ยังไม่มีผล (แผน B) |
| `verify-model` | ตรวจ hash ของโมเดลที่ล็อกไว้ |

---

## 6. โครงไฟล์

```
football/
├── fbpredict/
│   ├── config.py         ค่าตั้งต้นทั้งหมด (อ่านจาก .env)
│   ├── db.py             SQLite + คลังหลักฐานดิบ + look-ahead guard
│   ├── labels.py         สูตรคิดผล AH / O-U + ตัด vig
│   ├── features.py       feature ตาม README §3 (rolling ใช้ .shift(1))
│   ├── model.py          multi-output + กฎเลือกตลาด/ข้ามนัด + ล็อกโมเดล
│   ├── price_pool_rules.py  กฎราคาพูลชุดที่สอง (R1–R6) + no-vig + audit
│   ├── backtest.py       walk-forward 80/20 + blind_view + closing guard
│   ├── report.py         log loss / Brier / กำไรหลังหักค่าน้ำ + ส่วนสเปก §7.4/§9
│   ├── linking.py        จับคู่ราคา ↔ ผล
│   ├── demo.py           ข้อมูลสังเคราะห์ (ติดป้ายชัด)
│   ├── cli.py            command line
│   └── providers/        ตัวดึงข้อมูลแต่ละเจ้า
├── tests/                211 เทสต์
├── data/                 DB + ข้อมูลดิบ (ไม่ commit)
├── models/               โมเดลที่ล็อกไว้
├── reports/              รายงานผล
├── .env.example
├── run.ps1
├── QUICKSTART.md
└── DATA_SOURCES.md       ← อ่านเรื่อง robots.txt / ToS ที่นี่
```

---

## 7. อ่านผลลัพธ์ยังไง (สำคัญ)

ในรายงาน `reports/*.md` ให้ดู 3 อย่างนี้:

1. **`model log loss` เทียบ `baseline log loss`**
   ถ้าโมเดลแพ้ตลาด → ยังไม่มี edge ต่อให้ ROI เป็นบวกก็ตาม
   (demo ตัวอย่างแสดงเคสนี้ชัด ๆ: ROI +3% แต่ log loss แพ้ตลาดทั้งสองหัว
   = กำไรมาจากความบังเอิญ ไม่ใช่ความสามารถ)

2. **จำนวนนัดใน test set**
   ถ้า < 100 รายงานจะเตือนเองว่าคลาดเคลื่อน ±15% — ยังสรุปอะไรไม่ได้
   README §8.5 เขียนถูกแล้ว: 20 นัดบอกอะไรไม่ได้ทางสถิติ

3. **`Stability across the test window`**
   ถ้ากำไรมาจากช่วงเดียว แล้วช่วงอื่นขาดทุน = ยังไม่นิ่ง

---

## 8. กับดักที่โค้ดกันไว้แล้ว (อย่าไปปิด)

| กับดัก | กันยังไง |
|---|---|
| Data leakage จาก rolling | ทุก rolling ใช้ `.shift(1)` + คำนวณแยกตามทีม |
| โมเดลเห็นผลก่อนเตะ | `predicted_at` ต้อง < `kickoff_at` ทั้งใน Python และใน `CHECK` ของ SQLite |
| สุ่มแบ่ง train/test | `time_split()` เรียงตามเวลาเท่านั้น ไม่มี `shuffle` |
| แอบแก้โมเดลระหว่าง forward test | `ModelBundle.save()` เก็บ SHA-256 + git commit, `verify-model` ตรวจ |
| โมเดลทายตลาดที่ตัวเองไม่รู้ | `select_market()` ข้ามเมื่อ edge < threshold |
| เอา push มาปนเป็น binary | `ah_binary` / `ou_binary` คืน `None` แล้วตัดออก รายงานบอกจำนวนที่ตัด |
| เผลอใช้ราคาปิด | รายงานเตือนทุกครั้งให้ยืนยันว่าเป็นราคาเปิด |
| โมเดลแอบเห็นราคาปิดตอนทาย 24 ชม. | โหมดปกติเติม `fav_no_drop` / `dog_no_drop` เป็น NaN และตัดออกจาก feature_columns; ขอ `--use-closing-features` นอกหน้าต่าง 15 นาที = **refuse** (backtest + predict-forward) |
| โมเดลเรียนรู้จากกฎที่ทดสอบไม่ได้ | R4a (ไม่มีฟีด Ladbrokes) เป็น NaN ทั้งคอลัมน์ → ตัดออกและรายงานในหัวข้อ "Rules that stay untestable" — audit ก็ไม่วัดกฎนี้ |
| ราคาหายแล้วกฎยิงมั่ว | ราคาที่ขาดบนด้านที่กฎตรวจ → `match=0 / near=0 / dist=NaN` (ไม่ตีความว่าเข้าเงื่อนไข) |
| ใช้ราคาเจ้าอื่นแทน Ladbrokes เงียบ ๆ | ไม่มีช่องทางให้ทำ — R4a เป็น NaN ตายตัว เว้นแต่เพิ่มฟีดจริง |
| เอา margin ของภาพไปใช้กับเจ้ามือที่ margin ต่าง | มีชุด `v2_novig` เทียบ + รายงานเทียบ overround ของภาพกับของจริง (สเปก §9.4) |
