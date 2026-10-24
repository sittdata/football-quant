# แหล่งข้อมูล — สถานะที่ตรวจมาแล้ว และข้อควรระวัง

เอกสารนี้บันทึกว่าอะไร **ตรวจแล้ว** กับอะไร **ยังไม่ยืนยัน** เพื่อไม่ให้ใครเผลออ้าง
ข้อมูลจากบทความโปรโมตเป็นความจริง (README §8.8)

---

## 0. 5DollarFootballAPI (แนะนำที่สุด — ราคา + ผล ใน API เดียว)

- Base URL: `https://api.5dollarfootballapi.com/v1`
- Auth: `Authorization: Bearer fb_live_...`
- เอกสาร: https://5dollarfootballapi.com/docs

### ทำไมตัวนี้เหมาะกับโปรเจกต์นี้ที่สุด

คำขอเดียวได้ครบทั้ง **ผล + สกอร์ HT + ราคาเปิด** → **ไม่ต้อง join ราคากับผลเอง**
(ตัดปัญหาข้อ 6 ด้านล่างทิ้งได้เลย)

ตลาดที่ได้จาก `/v1/fixtures/{id}/odds` มีครบทั้ง `opening`, `closing`, `inplay`:

| คอลัมน์ของเรา | มาจาก |
|---|---|
| `ah_open` | `asian_handicap.opening.line` (เก็บเป็นค่าลบเสมอ = แฮนดิแคปของทีมต่อ) |
| `ah_fav_price` / `ah_dog_price` | `asian_handicap.opening.home/away` (สลับฝั่งตาม `fav_side`) |
| `ah_close` | `asian_handicap.closing.line` |
| `pool_fav_open` / `pool_dog_open` | `1x2.opening.home/away` (เอาค่าน้อย/มาก) |
| `pool_fav_close` / `pool_dog_close` | `1x2.closing.home/away` (สลับฝั่ง fav/dog ตาม `fav_side` เหมือน opening) |
| `draw_ft_odds` / `draw_ft_close` | `1x2.opening.draw` / `1x2.closing.draw` |
| `ou_line_open` / `ou_line_close` | `goal_line.opening/closing.line` |
| `ou_over_price` / `ou_under_price` | `goal_line.opening.over/under` |
| `ah_half_open` / `ah_half_close` | `asian_handicap_half.opening/closing.line` (ค่าลบ = แฮนดิแคปทีมต่อ) |
| `goal_line_half_open` / `goal_line_half_close` | `goal_line_half.opening/closing.line` (เส้นสูงต่ำครึ่งแรก) |
| `btts_open` / `btts_close` | `btts.opening/closing.yes` (ราคา "ยิงทั้งคู่") |
| `odds_format_raw` / `odds_format_stored` | ตรึงเป็น `decimal` — API นี้เป็น decimal ตลอดเส้นทาง ไม่มีการแปลงหน่วย |
| `ht_home` / `ht_away` | `goals.half_home` / `goals.half_away` |
| `home_goals` / `away_goals` | `goals.home` / `goals.away` |

**การได้ราคา `closing` มาด้วยทำให้ทดสอบกฎแถวสีแดงใน `tip.jpg` ได้จริง**
("-0.5 ไม่ไหลไป -0.75" คือกฎเรื่องการเคลื่อนไหวของเส้น ไม่ใช่ราคา)
โค้ดเก็บเป็น feature `ah_drift` และ `ah_not_drifted_to_075`

### ⚠️ สอง shape ที่ต้องระวัง

| endpoint | `asian_handicap.opening` | มีราคาไหม |
|---|---|---|
| `/v1/leagues/{id}/fixtures?include=odds` | `-0.25` (แค่เส้น) | ❌ |
| `/v1/fixtures/{id}/odds` | `{"line": -0.25, "home": 1.95, "away": 1.85}` | ✅ |

`include=odds` บน list เป็นฟีเจอร์**เสียเงิน** → แผนฟรีต้องดึงราคาทีละนัด (`--per-fixture-odds`)

### ❌ สิ่งที่ตัวนี้ไม่มี: `draw_ht_odds` และเจ้ามือ Ladbrokes

`/fixtures/{id}/odds` ไม่มีตลาด 1X2 ครึ่งแรก → **ราคาเสมอ HT เป็น NULL**
ถ้าจะได้ต้องใช้ `/v1/fixtures/{id}/odds/history?market=1x2_half` ซึ่งเป็น **Ultra เท่านั้น**
→ กฎ HT สองข้อใน README (`draw_ht >= 2.35` / `2.45`) จะทดสอบไม่ได้บนแผนฟรี
(แต่ **สกอร์ HT** ได้ครบ ซึ่งเป็นส่วนที่หายากที่สุด)

เส้นครึ่งแรกที่พอมี (`asian_handicap_half`, `goal_line_half`) ถูกเก็บแล้ว
แต่ **ไม่ใช่ราคา Over HT ของ Ladbrokes** ที่กฎ R4a ระบุ → ห้ามใช้แทนกัน
กฎ R4a จึงถูกปล่อยเป็น **NaN** และรายงานว่า "ทดสอบไม่ได้" ทุกฉบับ
(`rules.untestable` ใน `price_pool_rules.py` — โมเดลจะไม่เห็นกฎนี้เลย
ระบบจะรายงานด้วยว่า feature นี้ถูกตัดเพราะไม่มีข้อมูล)

### แผนและโควตา

| แผน | ราคา | ลีก | เจ้ามือ | ย้อนหลัง | โควตา |
|---|---|---|---|---|---|
| Free | $0 | top-5 | bet365 | 3 เดือน | 60/ชม. (20/นาที) |
| Community (นักศึกษา) | $0 | top-5 + 1 ประเทศ | bet365 | 12 เดือน | 300/ชม. |
| **Education (มหาวิทยาลัย)** | **$0** | 130+ | 20 เจ้ามือ | 12 เดือน | 10/นาที |
| Pro | $5/เดือน | 137 | bet365 | 12 เดือน | 10/นาที |
| Ultra | $25/เดือน | 1600+ | 20 เจ้ามือ | ทั้งหมด + tick history | 40/นาที |

**Education Access ฟรีสำหรับมหาวิทยาลัย** — อาจารย์สมัครเป็นคอร์สแล้วแจกโค้ดให้นักศึกษา
→ https://5dollarfootballapi.com/docs/education
(ถ้าเป็นงานเรียน ลองขออันนี้ จะได้ 130+ ลีก และย้อนหลัง 12 เดือน)

ทุก response มี header `X-RateLimit-Limit/Remaining/Reset` และ 429 จะมี `Retry-After`
โค้ดอ่าน header เหล่านี้แล้วหน่วงเวลาตามที่เซิร์ฟเวอร์สั่ง

### คำสั่ง

```powershell
.\run.ps1 probe-5dollar                      # ดูแผน + โควตา + league id
.\run.ps1 probe-5dollar --fixture-id 123456  # ดู odds block จริงที่ map เข้าคอลัมน์เรา
.\run.ps1 fetch-5dollar --league 3120672213 --season 26/27
```

ถ้าอยู่แผนฟรี (list ไม่มีราคา) ให้ดึงราคาทีละนัด:

```powershell
.\run.ps1 fetch-5dollar --league 3120672213 --season 26/27 --per-fixture-odds
```

ทุก response ถูก cache ไว้ → ถ้าแก้ schema/mapping แล้วอยากเติมคอลัมน์ใหม่ใน DB
โดยไม่ยิง API เลย ให้รันคำสั่งเดิมพร้อม `--cache-only`:

```powershell
.\run.ps1 fetch-5dollar --league 4160026622 --per-fixture-odds --cache-only
```

### ⚠️ Attribution บังคับ (แผนฟรี)

ข้อกำหนดระบุว่า ถ้าแสดงข้อมูลต่อสาธารณะบนแผน $0 ต้องใส่
**"Football data by 5DollarFootballAPI"** พร้อมลิงก์ไปที่เว็บไซต์
(ในรายงาน งานนำเสนอ README หรือ footer ก็ได้)

```
Football data by [5DollarFootballAPI](https://5dollarfootballapi.com)
```

---

## 1. OddsPapi (ราคาเปิด)

- Base URL: `https://api.oddspapi.io/v4`
- Endpoint: `/odds`, `/historical-odds`, `/bookmakers`, `/markets`
- แผนฟรี: **250 requests/เดือน** (ตามบทความของเขาเอง — **ยังไม่ยืนยันด้วยการเรียกจริง**)
- เจ้ามืออ้างอิง: **Pinnacle**

### ยังไม่ยืนยัน ❓

- คืน **ราคาเปิดย้อนหลัง** จริงไหม และมี **เส้นสูงต่ำเปิด** + **ราคาเสมอ HT** ไหม
- 1 แมตช์กินกี่ request (สำคัญมาก — 250/เดือน ÷ กี่ request ต่อนัด = จำนวนนัดที่ backtest ได้)
- response มีสกอร์ปนมาไหม (เอกสารบอกไม่มี)

### วิธีตรวจ — ทำก่อนเขียนอะไรทั้งนั้น

```powershell
.\run.ps1 probe-odds --fixture-id <id>
```

คำสั่งนี้จะ:

1. ยิง `/bookmakers` และ `/markets` จริง แล้วพิมพ์โครงสร้างที่ได้
2. ยิง `/odds` และ `/historical-odds` ของ fixture นั้น
3. เก็บ payload ดิบลง `data/raw/oddspapi/...` + ตาราง `raw_payload`
4. รายงานว่าเจอฟิลด์ score/goal/result ปนมาหรือไม่

**ถ้าผลออกมาว่าไม่มีราคาเปิดย้อนหลัง → ทั้งโปรเจกต์หยุดตรงนี้** เพราะกฎในตาราง
(`tip.jpg`) ใช้ราคาเปิดล้วน ๆ

---

## 2. API-Football (ผลแข่ง / label)

- Base: `https://v3.football.api-sports.io`
- Endpoint ที่ใช้: `/fixtures?league=<id>&season=<year>`
- แผนฟรี: **100 requests/วัน** — 1 request ได้ทั้งฤดูกาล → ประหยัดมาก
- League id ที่ map ไว้ในโค้ด: EPL=39, LaLiga=140, Bundesliga=78, SerieA=135, Ligue1=61, UCL=2

```powershell
.\run.ps1 fetch-results --provider api-football --league EPL --season 2025
```

---

## 3. football-data.org (ผลแข่ง / label)

- Base: `https://api.football-data.org/v4`
- Endpoint: `/competitions/<code>/matches?season=<year>`
- แผนฟรี: ลีกหลัก
- competition code: PL, PD, BL1, SA, FL1, CL, WC, EC

```powershell
.\run.ps1 fetch-results --provider football-data-org --competition PL --season 2025
```

---

## 4. Thscore (thscore99.com) — ⚠️ อ่านก่อนใช้

### สิ่งที่ตรวจแล้ว (ยิงจริง 21 ต.ค. 2026)

| ประเด็น | ผล |
|---|---|
| HTML ดิบมีข้อมูลแมตช์ไหม | **ไม่มี** — `#gameList` ว่างเปล่า ข้อมูลยัดด้วย JS |
| ต้องล็อกอินไหม (หน้า schedule) | ไม่ต้อง (`_needToken = 0`) |
| ย้อนหลังได้กี่วัน | **±7 วัน** (ตัวเลือกมี `f7`=14 ต.ค. ถึง `n7`=28 ต.ค. เท่านั้น) |
| มีสกอร์ไหม | **มี** — ทั้ง FT และ HT (อยู่ในเทมเพลต `{$hscore}` / `{$hhscore}`) |
| ราคาเปิด | **ยังไม่ยืนยัน** ต้องเปิดหน้าดีเทล (`toOddsDetail`) ดูเอง |
| robots.txt | **ห้าม** `/ajax/`, `/odata/`, `/bin/`, `/gf/`, `/scripts/`, `*.txt`, `/football/teamplayertech/`, `/football/teamstatistics/` |
| ข้อจำกัดความรับผิด | ระบุ "ข้อมูลเพื่อการอ้างอิงเท่านั้น" |
| กรรมสิทธิ์ | ห้ามทำซ้ำ/เผยแพร่โดยมีจุดประสงค์แสวงหากำไร |
| เจ้าของเว็บ | VAST PROFIT TECHNOLOGY LIMITED |

**endpoint ที่ข้อมูลอยู่ (`/ajax/`, `/odata/`) ถูกห้ามใน robots.txt**
แปลว่า automated access เป็นพื้นที่สีเทา — โค้ดจึงบังคับให้ต้องยืนยันเองก่อนทุกครั้ง

### ข้อจำกัดที่ทำให้ใช้เป็นแหล่งหลักไม่ได้

- ±7 วัน → **backtest 100+ นัดไม่ได้** จากเว็บนี้
- ยังไม่รู้ว่าราคาที่โชว์เป็นราคาเปิดหรือราคาปัจจุบัน

### ใช้ยังไงให้ปลอดภัย

```powershell
# ติดตั้ง playwright ก่อน (ครั้งเดียว)
C:\Users\sitti\AppData\Local\Programs\Python\Python313\python.exe -m pip install playwright
C:\Users\sitti\AppData\Local\Programs\Python\Python313\python.exe -m playwright install chromium

# ดูก่อน ไม่บันทึก
.\run.ps1 scrape-thscore --day today --i-understand-robots

# บันทึกสกอร์ (ราคาไม่เดา — ปล่อย NULL ไว้)
.\run.ps1 scrape-thscore --day today --i-understand-robots --save
```

โค้ดบังคับ guardrail เหล่านี้ไว้แล้ว:

- ต้องใส่ `--i-understand-robots` ทุกครั้ง ไม่งั้นไม่รัน
- หน่วงเวลาระหว่างคำขอ (`FB_REQUEST_DELAY`, ค่าเริ่มต้น 2 วินาที)
- **ไม่เดาความหมายของ 6 ช่องราคา** — มันเปลี่ยนตามการตั้งค่าผู้ใช้
  โค้ดจะพิมพ์ค่าที่เห็นออกมาให้ตรวจด้วยตา (`odds_span_report`) แล้วปล่อยคอลัมน์ราคาเป็น NULL
- เก็บ HTML ดิบไว้เป็นหลักฐาน

### ถ้าอาจารย์ถาม

พูดตรง ๆ ได้ว่า: ใช้ Thscore เป็น **แหล่งเสริม** สำหรับสกอร์ HT/FT และการตรวจสอบด้วยมือ
ส่วนแหล่งหลักเป็น API ที่มีข้อตกลงการใช้งานชัดเจน — และอ้าง robots.txt/disclaimer
ที่ตรวจมาแล้วข้างบนเป็นเหตุผล

---

## 5. ทางเลือกสำรอง

| เจ้า | จุดเด่น | ข้อควรระวัง |
|---|---|---|
| TheStatsAPI | มี AH เปิด + ย้อนหลัง 10 ปี | ยังไม่ทราบแผนฟรี |
| The Odds API | เอกสารดี | ย้อนหลังกินเครดิต 10 เท่า |

---

## 5.5 กฎราคาพูลสองชุด: `tip.jpg` กับภาพ R1–R6 (ห้ามรวมกัน)

โปรเจกต์นี้มีตารางกฎราคาพูล **สองชุด** ที่เก็บแยกกันเด็ดขาดตามสเปก §1:

| ชุด | ที่มา | โค้ด | จำนวนแถว |
|---|---|---|---|
| ชุดเดิม | `tip.jpg` (brief) | `features.RULES` | 8 แถว (แฮนดิแคป −0.25 ถึง −2.0) |
| ชุดที่สอง | ภาพ 2 ใบในสเปก | `price_pool_rules.RULES_V2` | 16 แถว (R1–R6) + ชุด no-vig |

**⚠️ สองชุดไม่ตรงกัน** ตัวอย่างที่สเปกยกเอง: AH −0.75
ชุด `tip.jpg` ว่า **1.60–1.64** แต่ภาพชุดที่สองว่า **1.72–1.79 / 1.80–1.89** (สองแถวแยกกัน)
สเปกห้ามรวมตาราง ห้ามเลือกข้าง — โค้ดจึงมี feature set แยก:
`v1` (ชุดเดิม) / `v2` (+R1–R6 ราคาดิบ) / `v2_novig` (+ชุดแปลงเป็นราคาไร้ vig)
และ **ห้ามรัน v2 กับ v2_novig รวมในโมเดลเดียว** (แผนระบุไว้)

การวัดผลกฎชุดที่สองดูได้จาก:

```powershell
.\run.ps1 audit-rules            # ตาราง hit rate ต่อกฎ (in-sample เท่านั้น)
.\run.ps1 audit-rules --novig    # ใช้ราคาไร้ vig ทั้งสองฝั่ง
```

นี่คือ **คำบรรยาย in-sample ไม่ใช่หลักฐาน edge** — กฎกับข้อมูลมาจากคนเดียวกัน
ช่วงเดียวกัน ต้องอ่านช่วงความเชื่อมั่น (95% CI) เทียบจุดคุ้มทุนของกฎนั้น ๆ เสมอ
ถ้า n < 30 รายงานจะเขียนว่า "สรุปไม่ได้"

- กฎ **R4a** ต้องมีราคา Over HT ของ Ladbrokes → ถูกปล่อยเป็น NaN (ดูข้อ 0)
- กฎ R3a/R4b ใช้เงื่อนไขราคาเสมออย่างเดียว → ทดสอบได้ (มีหมายเหตุในตาราง)
- ภาพทั้งสองมี margin ราว 1.03–1.14 แต่ bet365 ของเราวัดได้ 1.05–1.09 (เฉลี่ย ~1.07)
  → จึงมีชุด `v2_novig` ไว้เทียบว่ากฎทนการเปลี่ยนเจ้ามือหรือไม่ (สเปก §9.4)

---

## 6. การจับคู่ราคา ↔ ผล (จุดที่พังง่ายที่สุด)

ราคาและผลมักมาจากคนละเจ้า → ต้อง join เองด้วย **ชื่อทีม + เวลาเตะ**

```powershell
.\run.ps1 link --tolerance 36 --dry-run
```

`linking.py` ทำอะไรให้:

- normalize ชื่อทีม (ตัวพิมพ์, ตัว accent, เครื่องหมาย, คำฟุ่มเฟือยอย่าง FC/CF/SC/AFC)
- **ไม่สลับเหย้า/เยือน** — ต้อง match ทิศทางเดียวกัน
- เวลาเตะต้องต่างกันไม่เกิน `--tolerance` ชั่วโมง
- รายงานคู่ที่จับไม่ได้ และคู่ที่กำกวม (เลือกอันที่เวลาใกล้สุด แล้ว**บอกว่ากำกวม**)

ถ้าจับคู่ได้น้อยผิดปกติ ให้สงสัย timezone หรือชื่อทีมก่อน อย่าเพิ่งไปแก้ threshold

---

## 7. Checklist ก่อนส่งงาน

- [ ] มี `data/raw/` ที่เก็บ payload ดิบ + วันที่ดึง (หลักฐาน)
- [ ] ตาราง `raw_payload` มี `fetched_at` ครบ
- [ ] ระบุชื่อ **เจ้ามือ** ที่ใช้ในรายงาน (README §8.6)
- [ ] ยืนยันว่าเป็น **ราคาเปิด** ไม่ใช่ราคาปิด (README §8.2)
- [ ] test set ≥ 100 นัด (ถ้าน้อยกว่านั้น ให้เขียนว่าเป็น proof of concept)
- [ ] รายงานเทียบ baseline จากราคาน้ำเสมอ
- [ ] ล็อกกฎ + โมเดลก่อนทดสอบ ไม่ปรับตามข้อมูล test (README §8.4)
- [ ] ระบุว่าข้อมูลชุดไหนจริง ชุดไหนสังเคราะห์
