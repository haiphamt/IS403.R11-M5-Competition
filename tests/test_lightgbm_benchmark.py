"""End-to-end CLI regression check using explicitly SYNTHETIC data only."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkTest(unittest.TestCase):
    def test_ten_series_exports_and_unread_validation_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root/'raw'
            raw.mkdir()
            records = []
            for i in range(11):
                record = dict(id=f'SYNTHETIC_{i}',item_id=f'item_{i}',store_id='CA_1',state_id='CA')
                record.update({f'd_{d}':(d+i)%7 for d in range(1,1886)})
                # Non-numeric poison labels must not be read or used.
                record.update({f'd_{d}':'DO_NOT_READ_VALIDATION' for d in range(1886,1914)})
                records.append(record)
            pd.DataFrame(records).to_csv(raw/'sales_train_validation.csv',index=False)
            dates = pd.date_range('2011-01-29',periods=1913)
            calendar = pd.DataFrame(dict(d=[f'd_{d}' for d in range(1,1914)],
                wm_yr_wk=np.arange(1913)//7,wday=dates.day_of_week+1,
                month=dates.month,year=dates.year,snap_CA=0))
            for c in ['event_name_1','event_type_1','event_name_2','event_type_2']:
                calendar[c] = None
            calendar.to_csv(raw/'calendar.csv',index=False)
            pd.DataFrame([dict(store_id='CA_1',item_id=f'item_{i}',wm_yr_wk=w,sell_price=2.5)
                for i in range(11) for w in range(274)]).to_csv(raw/'sell_prices.csv',index=False)
            # Test/submission files are deliberately absent: C03 only needs three
            # CSVs and must not require unused labels or a submission template.
            result = subprocess.run([sys.executable,str(ROOT/'scripts/benchmark_lightgbm.py'),
                '--data-dir',str(raw),'--repeats','1','--output-dir',str(root/'output')],
                capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            runs = list((root/'output').glob('c03_lightgbm_*'))
            self.assertEqual(len(runs),1)
            summary = json.loads((runs[0]/'summary.json').read_text(encoding='utf-8'))
            timings = pd.read_csv(runs[0]/'runtime_repeats.csv')
            forecasts = pd.read_csv(runs[0]/'validation_forecasts.csv')
            self.assertEqual(summary['series_ids'],[f'SYNTHETIC_{i}' for i in range(10)])
            self.assertEqual(len(timings),10)
            self.assertEqual(len(forecasts),280)
            self.assertTrue(np.isfinite(forecasts.prediction).all())
            self.assertFalse(summary['validation_labels_used'])
            self.assertFalse(summary['test_labels_used'])
            self.assertEqual(summary['approval'],'pending; do not auto-mark Xong')
            self.assertTrue(timings.total_seconds.gt(0).all())
            self.assertEqual(summary['effective_training_target_range'],[56,1885])


if __name__ == '__main__':
    unittest.main()
