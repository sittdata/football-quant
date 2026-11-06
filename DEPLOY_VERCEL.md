# วิธีเชื่อมต่อและ Deploy ขึ้น Vercel (ขั้นตอนง่ายๆ)

เราได้สร้างเว็บ Dashboard ทำนายผลบอลพร้อม API ให้เรียบร้อยแล้ว รองรับ Vercel 100%

---

### วิธีที่ 1: Deploy ผ่าน GitHub (แนะนำที่สุด - ง่ายสุด 3 คลิก)

เนื่องจากโปรเจกต์นี้ผูกกับ GitHub อยู่แล้ว (`sittdev/football-quant`):

1. **บันทึกและ Push โค้ดขึ้น GitHub:**
   ```bash
   git add .
   git commit -m "Add web prediction dashboard and Vercel deployment"
   git push origin main
   ```

2. **เปิดเว็บ Vercel:**
   - เข้าเว็บ [vercel.com](https://vercel.com) แล้วล็อกอินด้วย GitHub
   - กดปุ่ม **"Add New..."** มุมขวาบน -> เลือก **"Project"**
   - จะเห็นรายชื่อ Repository ของคุณ ให้กดปุ่ม **"Import"** ที่ชื่อ `football-quant`

3. **กดปุ่ม Deploy:**
   - ทุกอย่างเราตั้งค่า `vercel.json` ไว้พร้อมหมดแล้ว ไม่ต้องเปลี่ยนค่าอะไรเลย
   - กดปุ่ม **"Deploy"** สีฟ้า
   - รอ 30 วินาที จะได้ URL เว็บไซต์ทันที เช่น:
     `https://football-quant.vercel.app`

---

### วิธีที่ 2: Deploy ผ่าน Terminal ทันที (ไม่ต้องเปิดหน้าเว็บ)

ถ้ามีบัญชี Vercel อยู่แล้ว สามารถพิมพ์คำสั่งนี้ใน Terminal เพื่อ Deploy ขึ้นไปได้เลย:

```bash
npx vercel
```

- ระบบจะถามยืนยัน ให้กด `y` และ Enter ไปเรื่อยๆ
- Vercel จะอัปโหลดและส่งลิงก์หน้าเว็บกลับมาให้ใน Terminal ทันที

---

### สิ่งที่ระบบแสดงบนหน้าเว็บ Vercel:
1. **Interactive Match Setup:**
   - เลือกทีมเหย้า / ทีมเยือน จาก **133 สโมสรใน 5 ลีกใหญ่ยุโรป** (Premier League, La Liga, Serie A, Bundesliga, Ligue 1)
   - ดึงสถิติจริงล่าสุด **xG (เกมรุก) และ xGA (เกมรับ)** มาแสดงอัตโนมัติ
   - ปรับเลื่อนราคาต่อรอง (Handicap) และค่าน้ำได้อิสระ
2. **Live AI Probability Bar:**
   - แสดงหลอดความน่าจะเป็นแบบ Real-time:
     - 🔴 ทีมเยือนชนะ (%)
     - 🟡 เสมอ (%)
     - 🟢 เจ้าบ้านชนะ (%)
   - มีเหรียญ Badge ประกาศชื่อทีมที่โมเดลคาดว่าจะชนะ
3. **Serverless API สำหรับทดสอบ:**
   - สามารถส่ง HTTP POST ไปที่ `https://<your-vercel-domain>/api/predict` เพื่อรับค่า JSON ทำนายผลได้ทันที
