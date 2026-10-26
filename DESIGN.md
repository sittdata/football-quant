# Design System: Institutional Quant Hedge Fund Terminal

## Visual World: Institutional Quant Trading Desk
The aesthetic of tier-1 quantitative trading operations (Bloomberg Terminal, Citadel, Millennium). High-density tabular data, unambiguous signal hierarchy, razor-sharp typography, and disciplined restraint. Every pixel serves informational clarity.

## Mode: Operate
Primary objective is rapid scanability, split-second decision making, and real-time capital oversight. Zero decorative distraction.

## Color Tokens
- Ground (Base Canvas): `#080c14`
- Surface 1 (Cards, Ledger Panels): `#0e1626`
- Surface 2 (Hover, Sub-cards, Headers): `#141f36`
- Surface Highlight (Active tabs, focus rings): `#1e2d4d`
- Borders:
  - Primary Border: `rgba(255, 255, 255, 0.08)`
  - Active/Focused Border: `rgba(255, 255, 255, 0.18)`
  - Gold Accent Border: `rgba(245, 158, 11, 0.35)`
- Typography Colors (WCAG AA Compliant):
  - Text Primary: `#f8fafc` (Contrast 14.8:1 on base)
  - Text Secondary: `#cbd5e1` (Contrast 9.4:1 on base)
  - Text Muted: `#8492a6` (Contrast 5.1:1 on base, fully compliant)
- Semantic Signals:
  - Alpha / Edge: Amber Gold `#f59e0b`
  - Win / Positive P&L: Emerald `#10b981`
  - Loss / Drawdown: Crimson Rose `#f43f5e`
  - Volume / Allocation: Cyan `#06b6d4`
  - Risk / Money Management: Indigo `#6366f1`

## Typography
- Display / Headers: `'Outfit', sans-serif`, weight 600/700, tracking `-0.025em`
- UI Controls & Labels: `'Inter', sans-serif`, weight 500/600
- Numerical Values & Telemetry: `'JetBrains Mono', monospace`, `font-variant-numeric: tabular-nums`

## Components & Anatomy
- **No Emojis:** Replace all inline emoji markers with crisp, authored SVG icons (16px and 20px).
- **No Gradient Text:** All numbers and labels use solid colors calibrated for contrast.
- **No Zero-Offset Glows:** Elevation is handled via clean 1px structural borders and directional soft drop-shadows (`box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4)`).
- **Status Badges:** Compact uppercase badges with solid background tints and 1px matching borders.
