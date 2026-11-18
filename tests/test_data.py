import unittest

import pandas as pd

from churn.config import FEATURES, TARGET
from churn.data import clean
from tests.synthetic import make_telco_like


class CleanTests(unittest.TestCase):
    def setUp(self):
        self.raw = make_telco_like(500)
        self.df = clean(self.raw)

    def test_blank_total_charges_fixed(self):
        self.assertTrue((self.raw["TotalCharges"].str.strip() == "").any())  # the quirk exists in the raw data
        self.assertFalse(self.df["TotalCharges"].isna().any())
        self.assertTrue((self.df.loc[self.df["tenure"] == 0, "TotalCharges"] == 0).all())
        self.assertTrue(pd.api.types.is_float_dtype(self.df["TotalCharges"]))

    def test_target_and_id(self):
        self.assertEqual(set(self.df[TARGET].unique()), {0, 1})
        self.assertNotIn("customerID", self.df.columns)
        self.assertEqual(self.df[TARGET].sum(), (self.raw["Churn"] == "Yes").sum())

    def test_senior_citizen_mapped(self):
        self.assertEqual(set(self.df["SeniorCitizen"].unique()), {"No", "Yes"})

    def test_all_features_present_and_input_untouched(self):
        self.assertTrue(set(FEATURES) <= set(self.df.columns))
        self.assertEqual(self.raw["Churn"].iloc[0] in ("Yes", "No"), True)  # raw not mutated

    def test_wrong_file_raises(self):
        with self.assertRaises(ValueError):
            clean(pd.DataFrame({"a": [1]}))


if __name__ == "__main__":
    unittest.main()
