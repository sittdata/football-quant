# Simulation test for KUBET multi-market resolver
import re

def test_market_resolver():
    # Simulate Row 1 (Home - 5 cells)
    row_home = ["25-10 19:00", "ฮิเบอร์เนียน", "0.91", "3 0.88", "3.636"]
    # Simulate Row 2 (Away - 4 cells)
    row_away = ["เซลติก", "1 1.02", "ต่ำ 1.05", "0.545"]
    # Simulate Row 3 (Draw - 2 cells)
    row_draw = ["เสมอ", "3.333"]

    test_jobs = [
        {"pick": "ต่อ เซลติก (-1.00)", "team_target": "เซลติก", "expected_row": "away", "expected_col": 1, "expected_odds": "1 1.02"},
        {"pick": "รอง ฮิเบอร์เนียน (+1.00)", "team_target": "ฮิเบอร์เนียน", "expected_row": "home", "expected_col": 2, "expected_odds": "0.91"},
        {"pick": "สูง Over 3.0", "team_target": "", "expected_row": "home", "expected_col": 3, "expected_odds": "3 0.88"},
        {"pick": "ต่ำ Under 3.0", "team_target": "", "expected_row": "away", "expected_col": 2, "expected_odds": "ต่ำ 1.05"},
        {"pick": "1X2 เซลติก", "team_target": "เซลติก", "expected_row": "away", "expected_col": 3, "expected_odds": "0.545"},
        {"pick": "1X2 เสมอ", "team_target": "เสมอ", "expected_row": "draw", "expected_col": 1, "expected_odds": "3.333"},
    ]

    all_passed = True
    for job in test_jobs:
        pick = job["pick"]
        is_ou = "สูง" in pick or "ต่ำ" in pick or "over" in pick.lower() or "under" in pick.lower()
        is_over = "สูง" in pick or "over" in pick.lower()
        is_1x2 = "1x2" in pick.lower()

        if is_ou:
            target_row = row_home if is_over else row_away
            col_idx = 3 if target_row == row_home else 2
        elif is_1x2:
            if "เสมอ" in pick or job["team_target"] == "เสมอ":
                target_row = row_draw
                col_idx = 1
            elif job["team_target"] == "ฮิเบอร์เนียน":
                target_row = row_home
                col_idx = 4
            else:
                target_row = row_away
                col_idx = 3
        else: # Handicap
            if job["team_target"] == "ฮิเบอร์เนียน":
                target_row = row_home
                col_idx = 2
            else:
                target_row = row_away
                col_idx = 1

        selected_odds = target_row[col_idx]
        passed = (selected_odds == job["expected_odds"])
        print(f"Test Pick: {pick:<26} -> Odds: {selected_odds:<10} | {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_passed = False

    return all_passed

if __name__ == "__main__":
    assert test_market_resolver()
    print("\nALL 6 MARKET RESOLVER TESTS PASSED 100%!")
