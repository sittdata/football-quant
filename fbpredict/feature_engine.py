"""Quant Feature Engine for Football Analytics.
Author: cook45

Calculates strictly historical, leakage-free features from SQLite database:
1. Team rolling offensive & defensive form (Goals For / Goals Against).
2. Combined match tempo & Expected Total goals.
3. Fatigue & Rest Days differentials.
4. Data adequacy guardrails (rejects matches with insufficient historical samples).
"""

from __future__ import annotations

import math
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from fbpredict import labels

# Thai to Standard English Team Name Mapping for Top 5 European Leagues
NAME_MAPPINGS: dict[str, str] = {
    # EPL
    "แมนซิตี้": "Man City", "แมนเชสเตอร์ซิตี": "Man City", "แมนฯ ซิตี้": "Man City", "แมนเชสเตอร์ ซิตี้": "Man City",
    "ลิเวอร์พูล": "Liverpool", "หงส์แดง": "Liverpool",
    "อาร์เซนอล": "Arsenal", "อาร์เซน่อล": "Arsenal", "ปืนใหญ่": "Arsenal",
    "เชลซี": "Chelsea", "สิงห์บลู": "Chelsea", "สิงห์บลูส์": "Chelsea",
    "แมนยู": "Man Utd", "แมนเชสเตอร์ ยูไนเต็ด": "Man Utd", "แมนฯ ยูไนเต็ด": "Man Utd", "ปีศาจแดง": "Man Utd",
    "สเปอร์ส": "Tottenham", "ท็อตแน่ม ฮ็อทสเปอร์": "Tottenham", "ท็อตแนม ฮ็อตสเปอร์": "Tottenham",
    "แอสตัน วิลล่า": "Aston Villa", "แอสตันวิลล่า": "Aston Villa", "วิลล่า": "Aston Villa",
    "นิวคาสเซิล": "Newcastle", "นิวคาสเซิ่ล": "Newcastle", "สาลิกาดง": "Newcastle",
    "ไบรท์ตัน": "Brighton", "ไบรท์ตันฯ": "Brighton", "ไบรท์ตัน โฮฟ อัลเบี้ยน": "Brighton",
    "เบรนท์ฟอร์ด": "Brentford",
    "เวสต์แฮม": "West Ham", "เวสต์แฮม ยูไนเต็ด": "West Ham",
    "ฟูแล่ม": "Fulham", "ฟูแลม": "Fulham",
    "เอฟเวอร์ตัน": "Everton",
    "คริสตัล พาเลซ": "Crystal Palace", "พาเลซ": "Crystal Palace",
    "บอร์นมัธ": "Bournemouth",
    "วูล์ฟแฮมป์ตัน": "Wolves", "วูล์ฟส์": "Wolves",
    "น็อตติ้งแฮม ฟอเรสต์": "Nott'm Forest", "ฟอเรสต์": "Nott'm Forest",
    "เลสเตอร์ ซิตี้": "Leicester", "เลสเตอร์": "Leicester",
    "อิปสวิช": "Ipswich", "อิปสวิช ทาวน์": "Ipswich",
    "เซาแธมป์ตัน": "Southampton", "นักบุญ": "Southampton",
    "ลีดส์": "Leeds", "ซันเดอร์แลนด์": "Sunderland",

    # La Liga
    "เรอัล มาดริด": "Real Madrid", "มาดริด": "Real Madrid", "ราชันชุดขาว": "Real Madrid",
    "บาร์เซโลน่า": "Barcelona", "บาร์เซโลนา": "Barcelona", "บาร์ซ่า": "Barcelona",
    "แอตเลติโก มาดริด": "Atl Madrid", "แอตฯ มาดริด": "Atl Madrid", "ตราหมี": "Atl Madrid",
    "กิโรน่า": "Girona", "คิโรน่า": "Girona",
    "แอธเลติก บิลเบา": "Ath Bilbao", "บิลเบา": "Ath Bilbao", "แอธเลติก คลับ": "Athletic Club",
    "เรอัล โซเซียดาด": "Real Sociedad", "โซเซียดาด": "Real Sociedad",
    "เรอัล เบติส": "Real Betis", "เบติส": "Real Betis",
    "บียาร์เรอัล": "Villarreal", "เรือดำน้ำสีเหลือง": "Villarreal",
    "เซบีย่า": "Sevilla", "เซบียา": "Sevilla",
    "บาเลนเซีย": "Valencia", "ไอ้ค้างคาว": "Valencia",
    "มายอร์ก้า": "Mallorca", "เซลต้า บีโก้": "Celta Vigo",
    "โอซาซูน่า": "Osasuna", "เกตาเฟ่": "Getafe", "ราโย บาเยกาโน่": "Rayo Vallecano",
    "อลาเบส": "Alaves", "ซีดี อลาเบส": "CD Alaves", "เอสปันญ่อล": "Espanyol",
    "ลาส พัลมาส": "Las Palmas", "เรอัล บายาโดลิด": "Valladolid", "เลกาเนส": "Leganes",

    # Serie A
    "อินเตอร์ มิลาน": "Inter", "อินเตอร์": "Inter", "งูใหญ่": "Inter",
    "เอซี มิลาน": "AC Milan", "มิลาน": "AC Milan", "ปีศาจแดงดำ": "AC Milan",
    "ยูเวนตุส": "Juventus", "ยูเว่": "Juventus", "ม้าลาย": "Juventus",
    "อตาลันต้า": "Atalanta", "นาโปลี": "Napoli", "โรม่า": "Roma", "ลาซิโอ": "Lazio",
    "ฟิออเรนติน่า": "Fiorentina", "โบโลญญ่า": "Bologna", "โตริโน่": "Torino",
    "เจนัว": "Genoa", "มอนซ่า": "Monza", "เวโรน่า": "Verona", "เลชเช่": "Lecce",
    "กายารี่": "Cagliari", "อูดิเนเซ่": "Udinese", "เอ็มโปลี": "Empoli",
    "ปาร์ม่า": "Parma", "โคโม่": "Como", "เวเนเซีย": "Venezia",

    # Bundesliga
    "บาเยิร์น มิวนิค": "Bayern Munich", "บาเยิร์น": "Bayern Munich", "เสือใต้": "Bayern Munich",
    "ไบเออร์ เลเวอร์คูเซ่น": "Leverkusen", "เลเวอร์คูเซ่น": "Leverkusen", "ห้างขายยา": "Leverkusen",
    "ดอร์ทมุนด์": "Dortmund", "โบรุสเซีย ดอร์ทมุนด์": "Dortmund", "เสือเหลือง": "Dortmund",
    "แอร์เบ ไลป์ซิก": "RB Leipzig", "ไลป์ซิก": "RB Leipzig",
    "สตุ๊ตการ์ท": "Stuttgart", "ไอน์ทรัค แฟร้งค์เฟิร์ต": "Ein Frankfurt", "แฟร้งค์เฟิร์ต": "Ein Frankfurt",
    "ฮอฟเฟ่นไฮม์": "Hoffenheim", "โวล์ฟสบวร์ก": "Wolfsburg", "เอาก์สบวร์ก": "Augsburg",
    "ไฟร์บวร์ก": "Freiburg", "แวร์เดอร์ เบรเมน": "Werder Bremen", "เบรเมน": "Werder Bremen",
    "โบรุสเซีย มึนเช่นกลัดบัค": "M'gladbach", "กลัดบัค": "M'gladbach",
    "ยูเนี่ยน เบอร์ลิน": "Union Berlin", "โบคุม": "Bochum", "ไมนซ์ 05": "Mainz",

    # Ligue 1
    "ปารีส แซงต์-แชร์กแมง": "Paris SG", "ปารีส": "Paris SG", "เปแอสเช": "Paris SG",
    "โมนาโก": "Monaco", "แบรสต์": "Brest", "ลีลล์": "Lille",
    "นีซ": "Nice", "ลียง": "Lyon", "โอลิมปิก ลียง": "Lyon",
    "มาร์กเซย": "Marseille", "โอลิมปิก มาร์กเซย": "Marseille",
    "ล็องส์": "Lens", "แรนส์": "Rennes", "ตูลูส": "Toulouse",
    "สตราส์บูร์ก": "Strasbourg", "น็องต์": "Nantes", "แร็งส์": "Reims",

    # Germany 2. Bundesliga
    "ฮัมบูร์ก": "Hamburg", "ฮัมบูร์เกอร์": "Hamburg",
    "ชาลเก้": "Schalke 04", "ชาลเก้ 04": "Schalke 04",
    "แฮร์ธ่า เบอร์ลิน": "Hertha", "แฮร์ธ่า": "Hertha",
    "โคโลญจน์": "FC Koln", "โคโลญ": "FC Koln",
    "ดุสเซลดอร์ฟ": "Fortuna Dusseldorf", "ฟอร์ทูน่า ดุสเซลดอร์ฟ": "Fortuna Dusseldorf",
    "ฮันโนเวอร์": "Hannover", "ฮันโนเวอร์ 96": "Hannover",
    "เนิร์นแบร์ก": "Nurnberg", "ไกเซอร์สเลาเทิร์น": "Kaiserslautern",
    "พาเดอร์บอร์น": "Paderborn", "กรอยเธอร์ เฟือร์ธ": "Greuther Furth",
    "คาร์ลสรูห์": "Karlsruhe", "มักเดบวร์ก": "Magdeburg",

    # Spain Segunda Division
    "เรอัล ซาราโกซ่า": "Zaragoza", "ซาราโกซ่า": "Zaragoza",
    "สปอร์ติ้ง กิฆอน": "Sp Gijon", "กิฆอน": "Sp Gijon",
    "เลบานเต้": "Levante", "เอลเช่": "Elche", "เออิบาร์": "Eibar",
    "เรอัล โอเบียโด้": "Oviedo", "โอเบียโด้": "Oviedo",
    "ราซิ่ง ซานตานเดร์": "Santander", "เตเนริเฟ่": "Tenerife",
    "กาดิซ": "Cadiz", "กรานาด้า": "Granada", "อัลเมเรีย": "Almeria",
    "อัลบาเซเต้": "Albacete", "บูร์โกส": "Burgos", "การ์ตาเกน่า": "Cartagena",

    # Italy Serie B
    "ซามพ์โดเรีย": "Sampdoria", "ปาแลร์โม่": "Palermo",
    "สเปเซีย": "Spezia", "ซาสซูโอโล่": "Sassuolo",
    "โฟรซิโนเน่": "Frosinone", "ซาแลร์นิตาน่า": "Salernitana",
    "เครโมเนเซ่": "Cremonese", "ปิซ่า": "Pisa",
    "บารี่": "Bari", "เบรสชา": "Brescia", "เชเซน่า": "Cesena",
    "โมเดน่า": "Modena", "เรจจาน่า": "Reggiana", "ซุดติโรล": "Sudtirol",

    # Netherlands Eredivisie
    "อาแจ็กซ์": "Ajax", "อาแจ็กซ์ อัมสเตอร์ดัม": "Ajax",
    "พีเอสวี": "PSV Eindhoven", "พีเอสวี ไอนด์โฮเฟ่น": "PSV Eindhoven",
    "เฟเยนูร์ด": "Feyenoord", "เฟเยนูร์ด ร็อตเธอร์ดัม": "Feyenoord",
    "อาแซด": "AZ Alkmaar", "อาแซด อัลค์มาร์": "AZ Alkmaar",
    "ทเวนเต้": "Twente", "เอฟซี ทเวนเต้": "Twente",
    "อูเทร็คท์": "Utrecht", "เอฟซี อูเทร็คท์": "Utrecht",
    "ฮีเรนวีน": "Heerenveen", "สปาร์ต้า ร็อตเตอร์ดัม": "Sparta Rotterdam",
    "โก อเฮด อีเกิ้ลส์": "Go Ahead Eagles", "เอ็นอีซี ไนเมเก้น": "NEC Nijmegen",
    "ฟอร์ทูน่า ซิตตาร์ด": "Fortuna Sittard", "ซโวลล์": "PEC Zwolle",

    # Portugal Primeira Liga
    "เบนฟิก้า": "Benfica", "ปอร์โต้": "Porto", "เอฟซี ปอร์โต้": "Porto",
    "สปอร์ติ้ง ลิสบอน": "Sporting CP", "สปอร์ติ้ง": "Sporting CP",
    "บราก้า": "Sp Braga", "เอสซี บราก้า": "Sp Braga",
    "วิตอเรีย กิมาไรส์": "Guimaraes", "กิมาไรส์": "Guimaraes",
    "เบาวิสต้า": "Boavista", "ฟามาลิเกา": "Famalicao",
    "กิล วิเซนเต้": "Gil Vicente", "ริโอ อาฟ": "Rio Ave",
    "อารัวก้า": "Arouca", "เอสโตริล": "Estoril", "โมไรเรนเซ่": "Moreirense",

    # Belgium Pro League
    "คลับ บรูซ": "Club Brugge", "คลับ บรูกก์": "Club Brugge",
    "อันเดอร์เลชท์": "Anderlecht", "อาร์เอสซี อันเดอร์เลชท์": "Anderlecht",
    "เกนท์": "Gent", "เคเอเอ เกนท์": "Gent",
    "เกงค์": "Genk", "เคอาร์ซี เกงค์": "Genk",
    "ยูเนี่ยน แซงต์ กิลลัวส์": "Union SG", "แซงต์ กิลลัวส์": "Union SG",
    "สตองดาร์ ลีแอช": "Standard", "ลีแอช": "Standard",
    "อันท์เวิร์ป": "Antwerp", "รอยัล อันท์เวิร์ป": "Antwerp",
    "เซอร์เคิล บรูซ": "Cercle Brugge", "เมเชเลน": "Mechelen",
    "ชาร์เลอรอย": "Charleroi", "แซงต์ ทรุยด็อง": "St Truiden",
}


def _clean_odds(val: Any, default: float = 1.95) -> float:
    """Safe odds sanitizer ensuring finite float strictly > 1.0."""
    try:
        if val is None or pd.isna(val):
            return default
        f = float(val)
        return f if (np.isfinite(f) and f > 1.0) else default
    except (TypeError, ValueError):
        return default


def _clean_price(val: Any) -> float | None:
    """Return a usable decimal price, or None when no real quote exists.

    Unlike ``_clean_odds`` this never invents a price: a missing quote must stay
    missing so fair-odds/vig features become NaN instead of a fabricated market.
    """
    try:
        if val is None or pd.isna(val):
            return None
        f = float(val)
        return f if (np.isfinite(f) and f > 1.0) else None
    except (TypeError, ValueError):
        return None


def normalize_team_name(raw: str) -> str:
    """Normalize team name by stripping HTML, common affixes, and applying dictionary."""
    if not raw:
        return ""
    clean = re.sub(r"<[^>]+>", "", str(raw)).strip()
    clean = re.sub(r"\([^\)]+\)", "", clean).strip()

    # Check mapping
    for th_key, en_val in NAME_MAPPINGS.items():
        if th_key.lower() in clean.lower() or clean.lower() in th_key.lower():
            return en_val

    # Strip generic terms
    clean = re.sub(r"(เอฟซี|เอฟ ซี|ยูไนเต็ด|สโมสร|คลับ|ซีตี้|fc|united|club|city)", "", clean, flags=re.I).strip()
    return clean



class HistoricalFeatureStore:
    """Fast in-memory cache and historical statistical query engine."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.df_history = self._load_finished_matches()
        self.team_matches = self._build_team_index()

    def _load_finished_matches(self) -> pd.DataFrame:
        if not self.db_path.exists():
            return pd.DataFrame()

        con = sqlite3.connect(self.db_path)
        query = """
        SELECT match_id, league, kickoff_at, home_team, away_team,
               home_goals, away_goals, ht_home, ht_away,
               ah_open, ou_line_open
        FROM matches
        WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL
        ORDER BY kickoff_at ASC
        """
        df = pd.read_sql_query(query, con)
        con.close()

        if df.empty:
            return df

        df["dt"] = pd.to_datetime(df["kickoff_at"], utc=True, errors="coerce")
        df["home_norm"] = df["home_team"].apply(normalize_team_name)
        df["away_norm"] = df["away_team"].apply(normalize_team_name)
        return df

    def _build_team_index(self) -> dict[str, list[dict[str, Any]]]:
        index: dict[str, list[dict[str, Any]]] = {}
        if self.df_history.empty:
            return index

        for _, r in self.df_history.iterrows():
            h_norm = str(r["home_norm"]).lower()
            a_norm = str(r["away_norm"]).lower()
            dt = r["dt"]
            h_g = r["home_goals"]
            a_g = r["away_goals"]
            tot = h_g + a_g
            ou_line = float(r["ou_line_open"]) if pd.notna(r["ou_line_open"]) else 2.5

            if h_norm:
                if h_norm not in index:
                    index[h_norm] = []
                index[h_norm].append({
                    "dt": dt,
                    "gf": h_g,
                    "ga": a_g,
                    "total": tot,
                    "is_over": tot > ou_line,
                    "is_home": True,
                })

            if a_norm:
                if a_norm not in index:
                    index[a_norm] = []
                index[a_norm].append({
                    "dt": dt,
                    "gf": a_g,
                    "ga": h_g,
                    "total": tot,
                    "is_over": tot > ou_line,
                    "is_home": False,
                })
        return index

    def get_team_history(
        self, team_name: str, before_dt: datetime, window: int = 5
    ) -> list[dict[str, Any]]:
        """Fetch strictly previous matches of a team prior to before_dt."""
        norm = normalize_team_name(team_name).lower()
        if not norm or norm not in self.team_matches:
            return []

        all_m = self.team_matches[norm]
        prior = [m for m in all_m if m["dt"] < before_dt]
        return prior[-window:]

    def enrich_match(
        self,
        home_team: str,
        away_team: str,
        kickoff_iso: str,
        ah_open: float,
        ou_line_open: float,
        ah_fav_price: float = 1.95,
        ah_dog_price: float = 1.95,
        ou_over_price: float = 1.95,
        ou_under_price: float = 1.95,
        draw_ft_odds: float = 3.30,
        fav_side: str = "home",
        ah_close: float | None = None,
        ou_line_close: float | None = None,
        ah_fav_close_price: float | None = None,
        ah_dog_close_price: float | None = None,
        ou_over_close_price: float | None = None,
        ou_under_close_price: float | None = None,
        window: int = 5,
    ) -> dict[str, Any]:
        """Compute complete, leakage-free feature vector for a match."""
        from fbpredict.steam_engine import analyze_market_steam

        try:
            target_dt = pd.to_datetime(kickoff_iso, utc=True)
        except Exception:
            target_dt = datetime.now(timezone.utc)

        # Sanitize odds inputs without inventing a price for missing quotes
        ah_fav_price = _clean_price(ah_fav_price)
        ah_dog_price = _clean_price(ah_dog_price)
        ou_over_price = _clean_price(ou_over_price)
        ou_under_price = _clean_price(ou_under_price)
        draw_ft_odds = _clean_price(draw_ft_odds)

        h_hist = self.get_team_history(home_team, target_dt, window=window)
        a_hist = self.get_team_history(away_team, target_dt, window=window)

        h_n = len(h_hist)
        a_n = len(a_hist)
        has_sufficient = (h_n >= 3) and (a_n >= 3)


        # 1. Rolling Goals For / Against
        h_gf = float(np.mean([m["gf"] for m in h_hist])) if h_hist else np.nan
        h_ga = float(np.mean([m["ga"] for m in h_hist])) if h_hist else np.nan
        a_gf = float(np.mean([m["gf"] for m in a_hist])) if a_hist else np.nan
        a_ga = float(np.mean([m["ga"] for m in a_hist])) if a_hist else np.nan

        # 2. Advanced Form Differentials
        if pd.notna(h_gf) and pd.notna(h_ga) and pd.notna(a_gf) and pd.notna(a_ga):
            goals_avg_diff = (h_gf - h_ga) - (a_gf - a_ga)
            expected_total = (h_gf + a_ga + a_gf + h_ga) / 2.0
        else:
            goals_avg_diff = np.nan
            expected_total = np.nan

        # 3. Over / Under Tempo Rate
        h_over_rate = float(np.mean([1.0 if m["is_over"] else 0.0 for m in h_hist])) if h_hist else np.nan
        a_over_rate = float(np.mean([1.0 if m["is_over"] else 0.0 for m in a_hist])) if a_hist else np.nan

        # 4. Rest Days & Fatigue
        if h_hist:
            last_h_dt = h_hist[-1]["dt"]
            h_rest = min(14.0, max(1.0, (target_dt - last_h_dt).total_seconds() / 86400.0))
        else:
            h_rest = 7.0

        if a_hist:
            last_a_dt = a_hist[-1]["dt"]
            a_rest = min(14.0, max(1.0, (target_dt - last_a_dt).total_seconds() / 86400.0))
        else:
            a_rest = 7.0

        rest_diff = h_rest - a_rest

        # 5. Favourite vs Underdog Relative Orientation
        side_norm = str(fav_side).strip().lower() if fav_side is not None else ""
        if side_norm == "home":
            is_fav_home: int | None = 1
        elif side_norm == "away":
            is_fav_home = 0
        else:
            is_fav_home = None

        if is_fav_home is None:
            fav_gf = fav_ga = dog_gf = dog_ga = np.nan
            fav_form = dog_form = fav_dog_form_diff = np.nan
            fav_rest = dog_rest = np.nan
            fav_rest_diff = np.nan
            fav_over_rate = dog_over_rate = np.nan
        else:
            fav_gf = h_gf if is_fav_home else a_gf
            fav_ga = h_ga if is_fav_home else a_ga
            dog_gf = a_gf if is_fav_home else h_gf
            dog_ga = a_ga if is_fav_home else h_ga

            fav_form = (fav_gf - fav_ga) if (pd.notna(fav_gf) and pd.notna(fav_ga)) else np.nan
            dog_form = (dog_gf - dog_ga) if (pd.notna(dog_gf) and pd.notna(dog_ga)) else np.nan
            fav_dog_form_diff = (fav_form - dog_form) if (pd.notna(fav_form) and pd.notna(dog_form)) else np.nan

            fav_rest = h_rest if is_fav_home else a_rest
            dog_rest = a_rest if is_fav_home else h_rest
            fav_rest_diff = fav_rest - dog_rest

            fav_over_rate = h_over_rate if is_fav_home else a_over_rate
            dog_over_rate = a_over_rate if is_fav_home else h_over_rate

        # 6. Fair Odds & Market Overround (Vig-free)
        if (
            ah_fav_price is not None and ah_dog_price is not None
            and pd.notna(ah_fav_price) and pd.notna(ah_dog_price)
            and float(ah_fav_price) > 1.0 and float(ah_dog_price) > 1.0
        ):
            fair_fav, fair_dog = labels.fair_two_way(ah_fav_price, ah_dog_price)
            ah_overround = (1.0 / float(ah_fav_price) + 1.0 / float(ah_dog_price)) - 1.0
        else:
            fair_fav, fair_dog = np.nan, np.nan
            ah_overround = np.nan

        if (
            ou_over_price is not None and ou_under_price is not None
            and pd.notna(ou_over_price) and pd.notna(ou_under_price)
            and float(ou_over_price) > 1.0 and float(ou_under_price) > 1.0
        ):
            fair_over, fair_under = labels.fair_two_way(ou_over_price, ou_under_price)
            ou_overround = (1.0 / float(ou_over_price) + 1.0 / float(ou_under_price)) - 1.0
        else:
            fair_over, fair_under = np.nan, np.nan
            ou_overround = np.nan

        ah_abs = abs(float(ah_open)) if (ah_open is not None and pd.notna(ah_open)) else np.nan
        total_x_ah = (float(ou_line_open) * ah_abs) if (ou_line_open is not None and pd.notna(ou_line_open) and pd.notna(ah_abs)) else np.nan

        # 7. Steam & Market Line Movement
        steam_sig = analyze_market_steam(
            open_ah=ah_open,
            current_ah=ah_close,
            open_fav_price=ah_fav_price,
            open_dog_price=ah_dog_price,
            current_fav_price=ah_fav_close_price or ah_fav_price,
            current_dog_price=ah_dog_close_price or ah_dog_price,
            open_ou=ou_line_open,
            current_ou=ou_line_close,
            open_over_price=ou_over_price,
            open_under_price=ou_under_price,
            current_over_price=ou_over_close_price or ou_over_price,
            current_under_price=ou_under_close_price or ou_under_price,
        )

        def _f(val: Any) -> float:
            if val is None:
                return np.nan
            try:
                f = float(val)
                return f if math.isfinite(f) else np.nan
            except (ValueError, TypeError):
                return np.nan

        return {
            "has_sufficient_history": int(has_sufficient),
            "home_sample_size": h_n,
            "away_sample_size": a_n,
            "fav_side": fav_side,
            "is_fav_home": is_fav_home if is_fav_home is not None else np.nan,
            # Market odds
            "ah_open": _f(ah_open),
            "ah_abs": ah_abs,
            "ah_fav_price": _f(ah_fav_price),
            "ah_dog_price": _f(ah_dog_price),
            "fair_fav_ah": fair_fav,
            "fair_dog_ah": fair_dog,
            "ah_overround": ah_overround,
            "ou_line_open": _f(ou_line_open),
            "ou_over_price": _f(ou_over_price),
            "ou_under_price": _f(ou_under_price),
            "fair_over": fair_over,
            "fair_under": fair_under,
            "ou_overround": ou_overround,
            "total_x_ah": total_x_ah,
            "draw_ft_odds": _f(draw_ft_odds),
            # Rolling performance (Home/Away for OU)
            "home_gf_avg": h_gf,
            "home_ga_avg": h_ga,
            "away_gf_avg": a_gf,
            "away_ga_avg": a_ga,
            "goals_avg_diff": goals_avg_diff,
            "expected_total": expected_total,
            "home_over_rate": h_over_rate,
            "away_over_rate": a_over_rate,
            "home_rest_days": h_rest,
            "away_rest_days": a_rest,
            "rest_diff": rest_diff,
            # Rolling performance (Fav/Dog for AH)
            "fav_gf_avg": fav_gf,
            "fav_ga_avg": fav_ga,
            "dog_gf_avg": dog_gf,
            "dog_ga_avg": dog_ga,
            "fav_form": fav_form,
            "dog_form": dog_form,
            "fav_dog_form_diff": fav_dog_form_diff,
            "fav_rest_days": fav_rest,
            "dog_rest_days": dog_rest,
            "fav_rest_diff": fav_rest_diff,
            "fav_over_rate": fav_over_rate,
            "dog_over_rate": dog_over_rate,
            # Steam & Smart Money Features
            "ah_close": ah_close,
            "ou_line_close": ou_line_close,
            "ah_line_delta": steam_sig.ah_line_delta,
            "ou_line_delta": steam_sig.ou_line_delta,
            "fav_prob_drift": steam_sig.fav_prob_drift,
            "dog_prob_drift": steam_sig.dog_prob_drift,
            "over_prob_drift": steam_sig.over_prob_drift,
            "under_prob_drift": steam_sig.under_prob_drift,
            "is_rlm": int(steam_sig.is_rlm),
            "steam_direction": steam_sig.steam_direction,
            "steam_strength": steam_sig.steam_strength,
            "steam_description": steam_sig.description,
            "steam_signal": steam_sig,
        }

