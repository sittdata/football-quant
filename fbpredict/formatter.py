"""Match prediction card formatter.

Transforms raw model picks into clean, human-friendly Match Cards
with Thai timezone (+7), intuitive betting badges, confidence ratings,
and multiple output styles (Markdown, Telegram HTML, Terminal Box).
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Mapping

THAI_TZ = timezone(timedelta(hours=7))

THAI_MONTHS = [
    "", "ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.",
    "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."
]

THAI_DAYS = [
    "จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"
]


def format_thai_kickoff(kickoff_raw: Any) -> str:
    """Convert raw UTC timestamp/string to Thai date/time string."""
    if not kickoff_raw:
        return "ไม่ระบุเวลา"
    
    dt: datetime | None = None
    if isinstance(kickoff_raw, datetime):
        dt = kickoff_raw
    elif isinstance(kickoff_raw, (int, float)):
        dt = datetime.fromtimestamp(kickoff_raw, tz=timezone.utc)
    else:
        try:
            # Parse ISO or standard formats
            dt = datetime.fromisoformat(str(kickoff_raw).replace("Z", "+00:00"))
        except Exception:
            try:
                dt = datetime.strptime(str(kickoff_raw)[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except Exception:
                return str(kickoff_raw)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
        
    local_dt = dt.astimezone(THAI_TZ)
    day_name = THAI_DAYS[local_dt.weekday()]
    month_name = THAI_MONTHS[local_dt.month]
    return f"{day_name} {local_dt.day} {month_name} • {local_dt.strftime('%H:%M')} น."


def parse_pick_details(row: Mapping[str, Any], pick: Any) -> dict[str, Any]:
    """Parse a row and pick into structured human-friendly components."""
    home = str(row.get("home_team") or "Home")
    away = str(row.get("away_team") or "Away")
    league = str(row.get("league") or "Football")
    kickoff_str = format_thai_kickoff(row.get("kickoff_at"))
    
    market = getattr(pick, "market", None) or row.get("market")
    side_odds = getattr(pick, "side_odds", None) or row.get("side_odds") or 0.0
    edge = getattr(pick, "edge", None) or row.get("edge") or 0.0
    p_model = getattr(pick, "p_model", None) or row.get("p_model")
    p_implied = getattr(pick, "p_implied", None) or row.get("p_implied")
    
    ah_open = row.get("ah_open")
    ou_line = row.get("ou_line_open")
    fav_side = row.get("fav_side", "home")
    
    # Try float conversion
    try:
        ah_val = abs(float(ah_open)) if ah_open is not None else 0.0
    except (ValueError, TypeError):
        ah_val = 0.0

    try:
        ou_val = float(ou_line) if ou_line is not None else 2.5
    except (ValueError, TypeError):
        ou_val = 2.5

    # Derive recommendation text & team focus
    if market == "AH_fav":
        fav_team = home if fav_side == "home" else away
        handicap_str = f"-{ah_val:.2f}"
        badge_text = f"ต่อ {fav_team}"
        pick_display = f"ต่อ {fav_team} (แต้มต่อ {handicap_str})"
        market_type = "แฮนดิแคป (Asian Handicap)"
        lines_info = f"ราคาเปิด AH: {row.get('ah_open', ah_val):+.2f} | เส้น OU: {ou_val:.2f}"
    elif market == "AH_dog":
        dog_team = away if fav_side == "home" else home
        handicap_str = f"+{ah_val:.2f}"
        badge_text = f"รอง {dog_team}"
        pick_display = f"รอง {dog_team} (ได้แต้มต่อ {handicap_str})"
        market_type = "แฮนดิแคป (Asian Handicap)"
        lines_info = f"ราคาเปิด AH: {row.get('ah_open', ah_val):+.2f} | เส้น OU: {ou_val:.2f}"
    elif market == "OU_under":
        badge_text = f"สกอร์ต่ำ Under {ou_val:.2f}"
        pick_display = f"สกอร์ต่ำ (Under {ou_val:.2f} ประตู)"
        market_type = "สกอร์สูง/ต่ำ (Over/Under)"
        lines_info = f"เส้นสูงต่ำ: {ou_val:.2f} | AH: {row.get('ah_open', 0.0):+.2f}"
    elif market == "OU_over":
        badge_text = f"สกอร์สูง Over {ou_val:.2f}"
        pick_display = f"สกอร์สูง (Over {ou_val:.2f} ประตู)"
        market_type = "สกอร์สูง/ต่ำ (Over/Under)"
        lines_info = f"เส้นสูงต่ำ: {ou_val:.2f} | AH: {row.get('ah_open', 0.0):+.2f}"
    else:
        badge_text = "ข้าม (ไม่เข้าเกณฑ์)"
        pick_display = "ไม่แนะนำให้เดิมพัน (Edge ต่ำกว่าเกณฑ์)"
        market_type = "-"
        lines_info = f"AH: {row.get('ah_open', 0.0):+.2f} | OU: {ou_val:.2f}"

    # Confidence rating
    edge_pct = edge * 100
    if edge_pct >= 15.0:
        tier = "ultra"
        tier_title = "🔥🔥🔥 ทีเด็ดระดับ 5 ดาว (Ultra Value)"
        stars = "⭐⭐⭐⭐⭐"
        grade = "A+"
    elif edge_pct >= 10.0:
        tier = "high"
        tier_title = "🔥 ทีเด็ดเน้นๆ 4 ดาว (High Value)"
        stars = "⭐⭐⭐⭐"
        grade = "A"
    elif edge_pct >= 6.0:
        tier = "medium"
        tier_title = "⭐️ คู่น่าลงทุน 3 ดาว (Value Pick)"
        stars = "⭐⭐⭐"
        grade = "B+"
    else:
        tier = "low"
        tier_title = "🟢 คู่น่าจับตา (Watchlist)"
        stars = "⭐⭐"
        grade = "B"

    prob_text = ""
    if p_model is not None and p_implied is not None:
        prob_text = f"โมเดลประเมิน {p_model*100:.1f}% vs ตลาด {p_implied*100:.1f}%"

    return {
        "home": home,
        "away": away,
        "match": f"{home} vs {away}",
        "league": league,
        "kickoff_thai": kickoff_str,
        "market": market,
        "badge_text": badge_text,
        "pick_display": pick_display,
        "market_type": market_type,
        "side_odds": float(side_odds) if side_odds else 0.0,
        "edge": edge,
        "edge_pct": edge_pct,
        "tier": tier,
        "tier_title": tier_title,
        "stars": stars,
        "grade": grade,
        "prob_text": prob_text,
        "lines_info": lines_info,
    }


def format_card_markdown(details: dict[str, Any]) -> str:
    """Render a single match card in Markdown."""
    lines = [
        f"### ⚽ {details['match']}",
        f"**🏆 ลีก:** `{details['league']}` | ⏰ **{details['kickoff_thai']}**",
        f"> 🎯 **ชี้เป้า:** **{details['pick_display']}**",
        f"> 💰 **ค่าน้ำ:** `{details['side_odds']:.2f}` | 📈 **ความคุ้มค่า (Edge):** `+{details['edge_pct']:.1f}%` {details['stars']}",
    ]
    if details['prob_text']:
        lines.append(f"> 📊 **การคำนวณ:** {details['prob_text']}")
    lines.append(f"*ข้อมูลตลาด: {details['lines_info']}*")
    return "\n".join(lines)


def format_card_telegram(details: dict[str, Any]) -> str:
    """Render a single match card in Telegram HTML format."""
    return (
        f"⚽ <b>{details['match']}</b>\n"
        f"🏆 <i>{details['league']}</i> | ⏰ <b>{details['kickoff_thai']}</b>\n"
        f"🎯 <b>ชี้เป้า:</b> <u>{details['pick_display']}</u>\n"
        f"💰 ค่าน้ำ: <code>{details['side_odds']:.2f}</code> | 📈 Edge: <b>+{details['edge_pct']:.1f}%</b> {details['stars']}\n"
        f"ℹ️ <i>{details['lines_info']}</i>"
    )


def format_card_terminal(details: dict[str, Any]) -> str:
    """Render a clean Unicode terminal box for console logs."""
    w = 70
    border = "─" * w
    lines = [
        f"┌{border}┐",
        f"│ ⚽ {details['match']:<{w - 5}} │",
        f"│ 🏆 {details['league']}  •  ⏰ {details['kickoff_thai']:<{w - 18}} │",
        f"├{border}┤",
        f"│ 🎯 ชี้เป้า : {details['pick_display']:<{w - 12}} │",
        f"│ 💰 ค่าน้ำ  : {details['side_odds']:<6.2f}  │  📈 Edge : +{details['edge_pct']:.1f}%  {details['stars']:<{w - 38}} │",
    ]
    if details["prob_text"]:
        lines.append(f"│ 📊 สถิติ   : {details['prob_text']:<{w - 12}} │")
    lines.append(f"│ ℹ️ ตลาด    : {details['lines_info']:<{w - 12}} │")
    lines.append(f"└{border}┘")
    return "\n".join(lines)


def format_matches_report(
    cards_data: list[dict[str, Any]],
    *,
    fmt: str = "markdown",
    group_by_tier: bool = True,
) -> str:
    """Format full collection of predictions grouped by confidence tier."""
    if not cards_data:
        return "ไม่มีคู่แข่งขันที่เข้าเกณฑ์การลงทุนในขณะนี้"

    # Sort by edge descending
    sorted_cards = sorted(cards_data, key=lambda c: c["edge"], reverse=True)
    
    if not group_by_tier:
        if fmt == "telegram":
            return "\n\n".join(format_card_telegram(c) for c in sorted_cards)
        elif fmt == "terminal":
            return "\n".join(format_card_terminal(c) for c in sorted_cards)
        else:
            return "\n\n---\n\n".join(format_card_markdown(c) for c in sorted_cards)

    # Group by tiers
    tiers = [
        ("ultra", "🔥🔥🔥 ทีเด็ดระดับ 5 ดาว (Edge ≥ 15%)"),
        ("high", "🔥 ทีเด็ดเน้นๆ 4 ดาว (Edge 10% - 14.9%)"),
        ("medium", "⭐️ คู่น่าลงทุน 3 ดาว (Edge 6% - 9.9%)"),
        ("low", "🟢 คู่น่าจับตา (Edge < 6%)"),
    ]

    output_sections: list[str] = []
    
    for tier_key, tier_heading in tiers:
        matching = [c for c in sorted_cards if c["tier"] == tier_key]
        if not matching:
            continue

        if fmt == "telegram":
            sec_lines = [f"<b>{tier_heading}</b>", ""]
            sec_lines.append("\n\n".join(format_card_telegram(c) for c in matching))
            output_sections.append("\n".join(sec_lines))
        elif fmt == "terminal":
            sec_lines = [f"\n=== {tier_heading} ==="]
            sec_lines.append("\n".join(format_card_terminal(c) for c in matching))
            output_sections.append("\n".join(sec_lines))
        else:
            sec_lines = [f"## {tier_heading}", ""]
            sec_lines.append("\n\n---\n\n".join(format_card_markdown(c) for c in matching))
            output_sections.append("\n".join(sec_lines))

    sep = "\n\n" + ("=" * 40) + "\n\n" if fmt != "terminal" else "\n"
    return sep.join(output_sections)
