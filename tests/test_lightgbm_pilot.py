"""Synthetic checks verify implementation, never count as real C03 evidence."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from m5.lightgbm_pilot import make_features, make_model, predict_nonnegative


def fixture(end=100, horizon=28):
    dates = pd.date_range('2011-01-29', periods=end+horizon)
    calendar = pd.DataFrame(dict(d=[f'd_{i}' for i in range(1,end+horizon+1)],
        wm_yr_wk=np.arange(end+horizon)//7, wday=dates.day_of_week+1,
        month=dates.month, year=dates.year, snap_CA=0))
    for column in ['event_name_1','event_name_2','event_type_1','event_type_2']:
        calendar[column] = None
    calendar.loc[end:, 'event_name_1'] = 'future_event'
    prices = pd.DataFrame({'wm_yr_wk':np.arange((end+horizon)//7+1), 'sell_price':2.5})
    return np.arange(1,end+1, dtype=float), calendar, prices


class LightGBMPilotTest(unittest.TestCase):
    def test_fixed_origin_lags_and_windows(self):
        history, calendar, prices = fixture()
        train, target, future = make_features(history,calendar,prices,state_id='CA',train_end=100)
        self.assertEqual(len(train),45)
        np.testing.assert_array_equal(target,np.arange(56,101))
        np.testing.assert_array_equal(future['lag_28'],np.arange(73,101))
        self.assertEqual(future.iloc[0]['rolling_mean_7'],np.mean(np.arange(67,74)))
        self.assertEqual(future.iloc[-1]['rolling_mean_28'],np.mean(np.arange(73,101)))
        self.assertAlmostEqual(future.iloc[-1]['rolling_std_28'],np.std(np.arange(73,101),ddof=1))
        self.assertEqual(list(train['event_name_1'].cat.categories),list(future['event_name_1'].cat.categories))

    def test_target_does_not_enter_its_own_features(self):
        history, calendar, prices = fixture()
        before = make_features(history,calendar,prices,state_id='CA',train_end=100)
        history[80:] += 9999
        after = make_features(history,calendar,prices,state_id='CA',train_end=100)
        # Day 101 only sees sales through day 73.
        pd.testing.assert_series_equal(before[2].iloc[0],after[2].iloc[0])
        pd.testing.assert_frame_equal(before[0],after[0])

    def test_missing_prices_are_not_backfilled(self):
        history, calendar, prices = fixture()
        prices = prices.iloc[10:]
        train, _, _ = make_features(history,calendar,prices,state_id='CA',train_end=100)
        self.assertTrue(pd.isna(train.iloc[0]['sell_price']))
        self.assertEqual(train.iloc[0]['price_missing'],1)

    def test_invalid_data_and_duplicate_keys(self):
        history, calendar, prices = fixture()
        for bad in [history[:-1], np.r_[history[:-1],np.nan], -history]:
            with self.assertRaises(ValueError):
                make_features(bad,calendar,prices,state_id='CA',train_end=100)
        with self.assertRaises(ValueError):
            make_features(history,calendar,prices,state_id='CA',train_end=100,horizon=29)
        for bad_calendar,bad_prices in [(pd.concat([calendar,calendar.iloc[:1]]),prices),
                                       (calendar,pd.concat([prices,prices.iloc[:1]]))]:
            with self.assertRaises(ValueError):
                make_features(history,bad_calendar,bad_prices,state_id='CA',train_end=100)

    def test_fit_predict_repeatable_and_all_zero_series(self):
        history, calendar, prices = fixture()
        for y in [history,np.zeros(100)]:
            train,target,future = make_features(y,calendar,prices,state_id='CA',train_end=100)
            params = dict(n_estimators=10,n_jobs=1,random_state=42,deterministic=True,force_col_wise=True,verbosity=-1)
            predictions = [predict_nonnegative(make_model(params).fit(train,target),future) for _ in range(2)]
            np.testing.assert_array_equal(*predictions)
            self.assertEqual(predictions[0].shape,(28,))
            self.assertTrue(np.isfinite(predictions[0]).all())
            self.assertTrue((predictions[0]>=0).all())


if __name__ == '__main__':
    unittest.main()
