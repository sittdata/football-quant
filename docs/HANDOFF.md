# Football Prediction Project — Handoff Document

> **สร้างเมื่อ:** 2026-10-22  
> **โปรเจกต์:** `c:\Users\sitti\OneDrive\Desktop\football`  
> **วัตถุประสงค์:** สรุปสิ่งที่ทำมาทั้งหมดใน session นี้ เพื่อส่งต่อให้ AI ตำแหน่งถัดไปดำเนินการต่อ

---

## 1. ภาพรวมโปรเจกต์

โปรเจกต์นี้คือระบบทำนายผลฟุตบอลสำหรับการเดิมพัน ใช้ **Machine Learning (Logistic Regression / XGBoost)** ร่วมกับ **กฎราคาพูล (Price Pool Rules)** จากภาพตาราง (`tip.jpg` และ spec `สเปกกฎราคาพูล.md`)

### สิ่งที่โมเดลทำนาย

ระบบทำนาย **2 ตลาด** พร้อมกัน (multi-output classifier):

| Head | ทำนายอะไร | ผลลัพธ์ |
|---|---|---|
| `p_fav_cover` | ความน่าจะเป็นที่ทีมต่อ (favourite) ชนะ handicap | เลือกแทง `AH_fav` หรือ `AH_dog` |
| `p_over` | ความน่าจะเป็นที่สกอร์รวมสูงกว่าเส้น | เลือกแทง `OU_over` หรือ `OU_under` |

ระบบเลือกตลาดที่มี **edge สูงสุด** เทียบกับ vig-free implied probability ของโต๊ะ

---

## 2. ผลการวิเคราะห์และข้อค้นพบ

### 2.1 `p_over` (OU_over) — ใช้ไม่ได้จริง

จากการทดสอบย้อนหลัง (backtest) ทุกรายการในโฟลเดอร์ `reports/`:

- โมเดลประเมินสกอร์สูงได้ **แย่กว่าโต๊ะ** ทุกครั้ง (log loss สูงกว่า baseline)
- `OU_over` ให้ผลขาดทุนในทุก backtest รายการ
- สาเหตุ: ตลาด O/U ของ Pinnacle มีประสิทธิภาพสูงมาก และ feature กฎราคาพูลถูกออกแบบมาเพื่อ AH ไม่ใช่ O/U

### 2.2 `AH_dog` และ `OU_under` — มี edge จริง

| ตลาด | Backtest ROI | หมายเหตุ |
|---|---|---|
| `AH_dog` | **+22.47%** | 44 bets, กำไร 9.887 units |
| `OU_under` | **+20.94%** | 48 bets, กำไร 10.050 units |
| รวม | **+21.67%** | 92 bets จาก 137 test matches |

> ⚠️ **คำเตือน:** Average edge 18.4% สูงผิดปกติมาก (real edge ปกติ 1-5%) บ่งบอกว่าโมเดล over-confident (miscalibrated) ยังต้องการ Forward Test ก่อนใช้จริง

### 2.3 กฎที่พิสูจน์แล้วว่าใช้ไม่ได้ (ตัดทิ้ง)

| กลุ่มกฎ | เหตุผลที่ตัด |
|---|---|
| **R3a, R3b** (OU_over FT) | Over bias — แย่กว่าโต๊ะ, CI ไม่ผ่าน break-even |
| **R4a** (HT Ladbrokes Over) | Untestable — ไม่มี Ladbrokes feed ใน plan นี้ |
| **R4b** (OU_over HT draw price) | Over bias เช่นกัน |
| **R5_1..R5_5** (single-point rows) | n < 30 ทุกกฎ — ไม่มีนัยสำคัญทางสถิติ |
| **R6_1** (single-point row) | n < 30 เช่นกัน |
| **R7** (water rule) | ยังไม่ผ่าน Forward Test — เก็บโค้ดไว้แต่ไม่ใช้ |

### 2.4 กฎที่เหลือและยังใช้ได้ (core rules)

| กฎ | AH Line | ทิศทาง | คำอธิบาย |
|---|---|---|---|
| **R1** | -0.25 | Favourite | Pool fav ต่ำ, draw สูง, dog สูง |
| **R1** | -0.50 | Favourite | เช่นเดียวกัน |
| **R2a** | -0.25 | **Underdog** | Pool สัญญาณ upset |
| **R2a** | -0.50 | **Underdog** | เช่นเดียวกัน |
| **R2b** | -0.50 | **Underdog** | variant ที่สอง |
| **R2c** | -0.50 | **Underdog** | variant ที่สาม |

---

## 3. การเปลี่ยนแปลงโค้ด

### 3.1 `fbpredict/config.py` — เปลี่ยนค่า default

```python
# เดิม:
edge_threshold: float = 0.03          # edge ขั้นต่ำ 3%
blacklist_markets: tuple = ()          # ไม่มี blacklist

# ใหม่:
edge_threshold: float = 0.06          # edge ขั้นต่ำ 6%
blacklist_markets: tuple = ("OU_over",)  # ห้ามแทง OU_over เด็ดขาด
```

อ่านจาก environment ได้:
```env
FB_EDGE_THRESHOLD=0.06
FB_BLACKLIST_MARKETS=OU_over
```

### 3.2 `fbpredict/price_pool_rules.py` — เพิ่ม `RULES_FILTERED`

เพิ่ม rule set ใหม่ที่ตัดกฎใช้ไม่ได้ออก เหลือแค่ R1 + R2:

```python
RULES_FILTERED: tuple[RuleRow, ...] = (
    RuleRow("R1",  (-0.25,), le(2.09), ge(3.25), ge(2.80)),
    RuleRow("R1",  (-0.5,),  le(1.85), ge(3.30), ge(4.00)),
    RuleRow("R2a", (-0.25,), ge(2.20), le(3.25), le(2.99)),
    RuleRow("R2a", (-0.5,),  ge(1.83), le(3.60), le(3.80)),
    RuleRow("R2b", (-0.5,),  ge(1.92), le(3.55), le(3.75)),
    RuleRow("R2c", (-0.5,),  ge(2.00), le(3.50), le(3.65)),
)
RULES_FILTERED_NOVIG: tuple[RuleRow, ...] = build_novig_table(RULES_FILTERED)
```

> `RULES_V2` (ชุดเต็ม) และ `RULES_WATER` (R7) **ยังคงอยู่ในโค้ด** เพื่อ backward compatibility — เพียงแต่ไม่ถูกใช้โดย default

### 3.3 `fbpredict/features.py` — เพิ่ม feature sets ใหม่

เพิ่ม 2 feature sets:
- `"filtered"` — v1 features + R1+R2 rules (raw prices)
- `"filtered_novig"` — v1 features + R1+R2 rules (vig-free probabilities)

```python
FEATURE_SETS = {
    "v1":             FEATURE_COLUMNS,
    "v2":             FEATURE_COLUMNS + RULE_SET_V2_FEATURES + CLOSING_FEATURES,
    "v2_novig":       FEATURE_COLUMNS + RULE_SET_V2_NOVIG_FEATURES + CLOSING_FEATURES_NOVIG,
    "water":          RULE_WATER_FEATURES,
    "filtered":       FEATURE_COLUMNS + RULE_SET_FILTERED_FEATURES + CLOSING_FEATURES,       # ใหม่
    "filtered_novig": FEATURE_COLUMNS + RULE_SET_FILTERED_NOVIG_FEATURES + CLOSING_FEATURES_NOVIG,  # ใหม่
}
```

### 3.4 `fbpredict/model.py` — ไม่เปลี่ยน

`select_market()` รองรับ `blacklist_markets` และ `allowed_markets` อยู่แล้ว

---

## 4. ผล Backtest ล่าสุด

**File:** `reports/backtest_filtered_novig.md`  
**Config:** `feature_set=filtered_novig`, `edge_threshold=0.06`, `blacklist=OU_over`

| metric | ค่า |
|---|---|
| Train | 548 matches |
| Test | 137 matches |
| Bets taken | **92** (skip 45) |
| Profit | **+19.938 units** |
| ROI | **+21.67%** |
| Hit rate | 60.87% |
| Max drawdown | 3.35 units |

**แยกตามตลาด:**

| ตลาด | Bets | Profit | ROI |
|---|---|---|---|
| AH_dog | 44 | +9.887 | +22.47% |
| OU_under | 48 | +10.050 | +20.94% |
| OU_over | 0 | — | blacklisted |

**Stability (4 periods — ไม่มี period ขาดทุน):**

| Period | Date Range | ROI |
|---|---|---|
| 1 | May 13 – May 23 | +27.61% |
| 2 | May 23 – Aug 29 | +11.79% |
| 3 | Aug 30 – Oct 10 | +18.86% |
| 4 | Oct 10 – Nov 02 | +28.42% |

---

## 5. โครงสร้างไฟล์สำคัญ

```
football/
├── fbpredict/
│   ├── config.py            ← [แก้] edge=0.06, blacklist=OU_over
│   ├── price_pool_rules.py  ← [แก้] เพิ่ม RULES_FILTERED + RULES_FILTERED_NOVIG
│   ├── features.py          ← [แก้] เพิ่ม "filtered" + "filtered_novig" feature sets
│   ├── model.py             ← ไม่เปลี่ยน
│   ├── backtest.py          ← ไม่เปลี่ยน
│   ├── labels.py            ← ไม่เปลี่ยน
│   └── cli.py               ← ไม่เปลี่ยน
├── reports/
│   ├── backtest_filtered_novig.md   ← ผล backtest ล่าสุด (config ใหม่)
│   ├── backtest_epl_v2.md           ← backtest v2 เก่า (อ้างอิง)
│   ├── rule_audit.md                ← audit กฎทั้งหมด
│   └── water_rule.md                ← audit R7
└── สเปกกฎราคาพูล (ส่งให้ AI เทรนโมเดล).md  ← spec ต้นฉบับ
```

---

## 6. สิ่งที่ยังต้องทำต่อ (Next Steps)

### Priority 1 — Forward Test (ก่อนใช้เงินจริง)

Backtest เป็น in-sample เท่านั้น ต้อง paper-trade คู่จริงก่อน:

1. รันระบบก่อนแข่ง 24 ชั่วโมง → บันทึกว่าระบบเลือกแทงอะไร
2. รอผลจริง → บันทึก W/L/P
3. สะสมอย่างน้อย **100-200 คู่** → ดูว่า edge ยังคงอยู่ไหม

### Priority 2 — Calibration Check

Average edge 18.4% สูงผิดปกติ (real edge ปกติ 1-5%):
- ทำ calibration plot: `predicted_prob` vs `actual_hit_rate`
- ถ้า over-confident → พิจารณา Platt scaling หรือเพิ่ม `edge_threshold` ต่อไป

### Priority 3 — เปรียบเทียบ `filtered` vs `filtered_novig`

ยังไม่ได้ทดสอบ `filtered` (raw prices) เทียบกับ `filtered_novig` (vig-free)

### Priority 4 — ทดสอบ Multi-league

ปัจจุบันรันแค่ EPL ควรทดสอบ La Liga, Bundesliga, Serie A

---

## 7. วิธีรัน

```bash
# Backtest ด้วย config ใหม่ (filtered_novig + edge 6% + blacklist OU_over)
python -m fbpredict backtest --feature-set filtered_novig --league EPL

# เปรียบเทียบกับ raw prices
python -m fbpredict backtest --feature-set filtered --league EPL

# ดู config ปัจจุบัน
python -m fbpredict config
```

**`.env` ที่ควรตั้ง:**
```env
FB_EDGE_THRESHOLD=0.06
FB_BLACKLIST_MARKETS=OU_over
FB_LEAGUE=EPL
FB_BOOKMAKER=Pinnacle
FB_BOOKMAKER_SLUG=bet365
FIVE_DOLLAR_FOOTBALL_API_KEY=<your_key>
```

---

## 8. ข้อจำกัดสำคัญ

> **Backtest ≠ Real Performance**  
> ROI +21.67% เป็น in-sample เท่านั้น ยังไม่มีหลักฐานว่า edge จะคงอยู่ในอนาคต

> **ห้ามใช้ OU_over** — blacklist แล้ว แต่ถ้าใครปิด blacklist แล้วรัน โมเดลจะเลือก OU_over ได้ ซึ่งข้อมูลพิสูจน์แล้วว่าขาดทุน

> **R4a ยัง untestable** — ต้องการ Ladbrokes Over HT price ซึ่งไม่มีใน plan ปัจจุบัน โค้ดจะ emit NaN อัตโนมัติ

> **draw_ht_odds ขาด 100%** — ฟีเจอร์นี้ไม่มีข้อมูลเลย (0/137) โมเดล drop อัตโนมัติก่อน fit
