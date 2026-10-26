"""Unit tests for the Steam & Line Movement Engine.
Author: cook45
"""

import unittest
from fbpredict.steam_engine import (
    SteamSignal,
    analyze_market_steam,
    calculate_ah_drift,
    calculate_ou_drift,
    evaluate_steam_confirmation,
)


class TestSteamEngine(unittest.TestCase):
    def test_line_deltas(self):
        # Favorite handicap moves from -0.50 to -0.75 (Favorite steam)
        sig = analyze_market_steam(
            open_ah=-0.50,
            current_ah=-0.75,
            open_fav_price=1.95,
            open_dog_price=1.95,
            current_fav_price=1.95,
            current_dog_price=1.95,
        )
        self.assertAlmostEqual(sig.ah_line_delta, -0.25)
        self.assertEqual(sig.steam_direction, "STEAM_FAV")
        self.assertFalse(sig.is_rlm)

    def test_reverse_line_movement(self):
        # Line moves from -0.75 to -0.50 (easier for favorite -> underdog steam)
        sig = analyze_market_steam(
            open_ah=-0.75,
            current_ah=-0.50,
            open_fav_price=1.95,
            open_dog_price=1.95,
            current_fav_price=1.95,
            current_dog_price=1.95,
        )
        self.assertAlmostEqual(sig.ah_line_delta, 0.25)
        self.assertEqual(sig.steam_direction, "STEAM_DOG")
        self.assertTrue(sig.is_rlm)

    def test_price_drift_without_line_change(self):
        # Line stays at -0.50, but Underdog price drops from 2.05 to 1.82 (heavy buy)
        sig = analyze_market_steam(
            open_ah=-0.50,
            current_ah=-0.50,
            open_fav_price=1.85,
            open_dog_price=2.05,
            current_fav_price=2.10,
            current_dog_price=1.82,
        )
        self.assertAlmostEqual(sig.ah_line_delta, 0.0)
        self.assertGreater(sig.dog_prob_drift, 0.04)
        self.assertEqual(sig.steam_direction, "STEAM_DOG")
        self.assertTrue(sig.is_rlm)

    def test_ou_line_steam(self):
        # Over/Under moves from 2.5 to 2.75
        sig = analyze_market_steam(
            open_ah=0.0,
            current_ah=0.0,
            open_ou=2.50,
            current_ou=2.75,
            open_over_price=1.95,
            open_under_price=1.95,
            current_over_price=1.95,
            current_under_price=1.95,
        )
        self.assertAlmostEqual(sig.ou_line_delta, 0.25)
        self.assertEqual(sig.steam_direction, "STEAM_OVER")

    def test_evaluate_steam_confirmation(self):
        sig_dog = analyze_market_steam(
            open_ah=-0.75,
            current_ah=-0.50,
        )
        # Model picks AH_dog and Steam is STEAM_DOG -> Confirmed!
        conf = evaluate_steam_confirmation("AH_dog", sig_dog)
        self.assertTrue(conf["is_confirmed"])
        self.assertEqual(conf["conviction_tier"], "TIER_1_STEAM_CONFIRMED")
        self.assertEqual(conf["badge_label"], "🔥 STEAM CONFIRMED")

        # Model picks AH_fav when Steam is STEAM_DOG -> Not Confirmed
        unconf = evaluate_steam_confirmation("AH_fav", sig_dog)
        self.assertFalse(unconf["is_confirmed"])
        self.assertEqual(unconf["conviction_tier"], "TIER_2_MODEL_ONLY")

    def test_safe_fallbacks_with_nans(self):
        sig = analyze_market_steam(
            open_ah=None,
            current_ah=None,
            open_fav_price=None,
            open_dog_price=None,
        )
        self.assertEqual(sig.ah_line_delta, 0.0)
    def test_compute_synergy_kelly(self):
        from fbpredict.steam_engine import compute_synergy_kelly
        
        # Test 1: Steam confirmed -> 1.25x scaling
        sig_dog = analyze_market_steam(open_ah=-0.75, current_ah=-0.50)
        stake = compute_synergy_kelly(base_stake=1.0, market_pick="AH_dog", steam_signal=sig_dog)
        self.assertAlmostEqual(stake, 1.25)
        
        # Test 2: Steam opposed -> 0.50x scaling
        stake_opp = compute_synergy_kelly(base_stake=1.0, market_pick="AH_fav", steam_signal=sig_dog)
        self.assertAlmostEqual(stake_opp, 0.50)
        
        # Test 3: Neutral steam -> 1.0x scaling
        sig_neutral = analyze_market_steam(open_ah=-0.50, current_ah=-0.50)
        stake_neutral = compute_synergy_kelly(base_stake=1.0, market_pick="AH_dog", steam_signal=sig_neutral)
        self.assertAlmostEqual(stake_neutral, 1.00)

        # Test 4: Clamping to max_stake=2.5
        stake_max = compute_synergy_kelly(base_stake=2.2, market_pick="AH_dog", steam_signal=sig_dog)
        self.assertLessEqual(stake_max, 2.5)


if __name__ == "__main__":
    unittest.main()

