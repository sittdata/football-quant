"""Thscore (thscore99.com) - **supplementary** source, opt-in only.

Why this one is treated carefully
---------------------------------
* ``robots.txt`` disallows ``/ajax/``, ``/odata/``, ``/bin/``, ``/gf/`` and
  ``/scripts/`` - which is where this site serves its data. Automated access is
  therefore a grey area, which is why this provider refuses to run unless you
  explicitly pass ``confirm_robots=True``.
* The schedule page only offers +/-7 days, so it cannot supply a historical
  backtest. It is useful for (a) eyeballing a match by hand and (b) a forward
  test where you collect a little each day.
* The HTML is a shell; the rows are injected by JavaScript, so a plain
  ``requests`` fetch returns nothing. This module drives a real browser.

Practical use: fetch one day, save the raw HTML as evidence, and read the
HT/FT scores (which are the hard-to-find part of the dataset).

Requires:  pip install playwright  &&  python -m playwright install chromium
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..config import Config
from ..db import to_iso_utc
from .base import BaseProvider, ProviderError

SITE = "https://www.thscore99.com"

#: day offsets the site exposes: f1..f7 = past, today, n1..n7 = future
VALID_DAYS = tuple([f"f{i}" for i in range(1, 8)]) + ("today",) + tuple(f"n{i}" for i in range(1, 8))


@dataclass
class ScrapedRow:
    match_id: str
    league: str | None
    kickoff_text: str | None
    home_team: str
    away_team: str
    ht_home: int | None
    ht_away: int | None
    home_goals: int | None
    away_goals: int | None
    state: str | None
    odds_spans: list[str]


_EXTRACT_JS = r"""
() => {
  const rows = [];
  const headers = [...document.querySelectorAll('div.group-title')];
  const leagueFor = (el) => {
    for (let i = headers.length - 1; i >= 0; i--) {
      const h = headers[i];
      if (h.compareDocumentPosition(el) & Node.DOCUMENT_POSITION_FOLLOWING) {
        const name = h.querySelector('.leaRow');
        return name ? name.textContent.trim() : null;
      }
    }
    return null;
  };
  const text = (id) => {
    const el = document.getElementById(id);
    return el ? el.textContent.trim() : null;
  };
  const num = (id) => {
    const raw = text(id);
    if (raw === null || raw === '') return null;
    const n = parseInt(raw, 10);
    return Number.isNaN(n) ? null : n;
  };
  document.querySelectorAll('div.matchDiv').forEach((el) => {
    const mid = el.getAttribute('data-mlid') || (el.id || '').replace('tb_', '');
    if (!mid) return;
    const odds = [];
    for (let i = 1; i <= 6; i++) {
      const v = text('o' + i + '_' + mid);
      if (v !== null) odds.push(v);
    }
    rows.push({
      match_id: mid,
      league: leagueFor(el),
      kickoff_text: text('mt_' + mid),
      home_team: text('ht_' + mid),
      away_team: text('gt_' + mid),
      ht_home: num('hht_' + mid),
      ht_away: num('ght_' + mid),
      home_goals: num('hsc_' + mid),
      away_goals: num('gsc_' + mid),
      state: text('state_' + mid),
      odds_spans: odds,
    });
  });
  return rows;
}
"""


class ThscoreProvider(BaseProvider):
    """Scrapes one schedule day. Deliberately slow and explicit."""

    def __init__(self, cfg: Config, con=None, use_cache: bool = False) -> None:
        super().__init__(name="thscore", cfg=cfg, con=con, use_cache=use_cache)

    # ------------------------------------------------------------------ #
    def _playwright(self):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover - depends on the machine
            raise ProviderError(
                "playwright is not installed. Run:\n"
                "  python -m pip install playwright\n"
                "  python -m playwright install chromium"
            ) from exc
        return sync_playwright

    def scrape_day(
        self,
        day: str = "today",
        *,
        confirm_robots: bool = False,
        date_hint: str | None = None,
        headless: bool = True,
    ) -> list[ScrapedRow]:
        """Return the rows of one schedule day.

        ``confirm_robots=True`` is required: it is your acknowledgement that you
        have read the site's robots.txt / terms and accept the grey area.
        """
        if not confirm_robots:
            raise ProviderError(
                "Thscore scraping is opt-in. Re-run with --i-understand-robots to "
                "acknowledge that robots.txt disallows the data endpoints."
            )
        if day not in VALID_DAYS:
            raise ProviderError(f"day must be one of {VALID_DAYS}")

        sync_playwright = self._playwright()
        url = f"{SITE}/football/schedule/{day}"
        kickoff_date = date_hint or datetime.now(timezone.utc).date().isoformat()

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=headless)
            try:
                page = browser.new_page(locale="th-TH")
                page.goto(url, wait_until="domcontentloaded", timeout=int(self.cfg.http_timeout * 1000))
                # the list is injected after an XHR; wait for it, then settle
                try:
                    page.wait_for_selector("div.matchDiv", timeout=20000)
                except Exception as exc:  # noqa: BLE001 - playwright raises its own type
                    raise ProviderError(
                        f"no match rows appeared for {url}; the site may have changed "
                        f"or blocked the request ({exc})"
                    ) from exc
                self._pause()

                html = page.content()
                if self.con is not None:
                    self._archive(f"/football/schedule/{day}", f"thscore-{day}-{kickoff_date}", html)

                raw_rows: list[dict[str, Any]] = page.evaluate(_EXTRACT_JS)
            finally:
                browser.close()

        rows: list[ScrapedRow] = []
        for item in raw_rows:
            if not item.get("home_team") or not item.get("away_team"):
                continue
            rows.append(
                ScrapedRow(
                    match_id=str(item.get("match_id")),
                    league=item.get("league"),
                    kickoff_text=item.get("kickoff_text"),
                    home_team=item["home_team"],
                    away_team=item["away_team"],
                    ht_home=item.get("ht_home"),
                    ht_away=item.get("ht_away"),
                    home_goals=item.get("home_goals"),
                    away_goals=item.get("away_goals"),
                    state=item.get("state"),
                    odds_spans=list(item.get("odds_spans") or []),
                )
            )
        return rows

    # ------------------------------------------------------------------ #
    def to_match_rows(
        self,
        rows: list[ScrapedRow],
        *,
        date_hint: str | None = None,
        fetched_at: str | None = None,
    ) -> list[dict]:
        """Convert scraped rows into ``matches`` rows (results only).

        Odds are intentionally **not** written: which of the six odds columns is
        which depends on the visitor's own display setting, so they are returned
        separately for manual checking instead of being guessed.
        """
        fetched_at = fetched_at or to_iso_utc(datetime.now(timezone.utc))
        day = date_hint or datetime.now(timezone.utc).date().isoformat()
        out: list[dict] = []

        for row in rows:
            kickoff = f"{day}T00:00:00+00:00"
            if row.kickoff_text:
                parts = row.kickoff_text.replace(":", " ").split()
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    kickoff = f"{day}T{int(parts[0]):02d}:{int(parts[1]):02d}:00+00:00"

            out.append(
                {
                    "match_id": f"thscore:{day}:{row.match_id}",
                    "provider": "thscore",
                    "league": row.league,
                    "season": day[:4],
                    "kickoff_at": kickoff,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "home_goals": row.home_goals,
                    "away_goals": row.away_goals,
                    "ht_home": row.ht_home,
                    "ht_away": row.ht_away,
                    "result_source": "thscore",
                    "result_fetched_at": fetched_at,
                }
            )
        return out


def odds_span_report(rows: list[ScrapedRow], limit: int = 15) -> str:
    """Readable dump so a human can decide what the six odds columns mean."""
    lines = [
        "Odds columns seen on the Thscore schedule page.",
        "Which market each column shows depends on the site's own display setting,",
        "so verify by hand before writing these into the database.",
        "",
    ]
    for row in rows[:limit]:
        lines.append(f"{row.home_team} vs {row.away_team}  ({row.state or '?'})")
        lines.append(f"    columns: {row.odds_spans or '(none shown)'}")
    if len(rows) > limit:
        lines.append(f"... {len(rows) - limit} more rows omitted")
    return "\n".join(lines)
