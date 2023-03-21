from typing import List, Union

import holidays
import pandas as pd

from ..utils.list import wrap_in_list
from .base import Transformation, fit_noop


class AddHolidayFeatures(Transformation):
    """
    Adds holiday features for given regions as a column.
    """

    properties = Transformation.Properties()

    def __init__(self, country_codes: Union[List[str], str]) -> None:
        self.country_codes = wrap_in_list(country_codes)
        self.name = f"AddRegions-{self.country_codes}"
        self.holidays = [
            holidays.country_holidays(country_code)
            for country_code in self.country_codes
        ]

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return pd.concat(
            [
                X.copy(),
                pd.DataFrame(
                    {
                        country.country: X.index.map(lambda d: d in country).to_list()
                        for country in self.holidays
                    },
                    index=X.index,
                ),
            ],
            axis=1,
        )

    fit = fit_noop
    update = fit_noop
