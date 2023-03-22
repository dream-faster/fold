from enum import Enum
from typing import List, Optional, Union

import holidays
import pandas as pd

from ..utils.list import wrap_in_list
from .base import Transformation, fit_noop


def get_weekends(dates: pd.DatetimeIndex) -> pd.Series:
    return pd.Series((dates.weekday > 4).astype(int), index=dates)


def encode_series(series: pd.Series) -> pd.Series:
    return series.astype("category").cat.codes


def get_multi_holidays(
    dates: pd.DatetimeIndex, country_codes: List[str], encode: Optional[bool] = False
) -> pd.DataFrame:
    series = [
        pd.Series(
            holidays.country_holidays(
                country_code,
                years=dates.year.unique().to_list(),
                language="en_US",
            ),
            index=dates.date,
            name=country_code,
        ).reset_index(drop=True)
        for country_code in country_codes
    ]
    for serie in series:
        serie.index = dates

    df = pd.concat(series, axis="columns").fillna(0.0)

    if encode:
        # Turn individual holiday names into category than reindex with the original dates
        df = df.apply(encode_series).add(1).reindex(dates)

    return df


class LabelingMethod(Enum):
    holiday_binary = "holiday_binary"
    weekday_weekend_holiday = "weekday_weekend_holiday"
    weekday_weekend_uniqueholiday = "weekday_weekend_uniqueholiday"

    @staticmethod
    def from_str(value: Union[str, "LabelingMethod"]) -> "LabelingMethod":
        if isinstance(value, LabelingMethod):
            return value
        for strategy in LabelingMethod:
            if strategy.value == value:
                return strategy
        else:
            raise ValueError(f"Unknown HolidayType: {value}")


class AddHolidayFeatures(Transformation):
    """
    Adds holiday features for given regions as multiple columns.


    Parameters
    ----------
    country_codes: List of Country codes (eg.: `US`, `DE`) for each of which to create a column according to type
    type: How the holidays should be calculated
        - holiday_binary: Non-special days = 0.0 | holidays = 1.0
        - weekday_weekend_holiday: Non-special days = 0.0 | Weekends = 1.0 | holidays == 2.0 | holidays + weekend == 3.0
        - weekday_weekend_uniqueholiday: Non-special days = 0.0 | Weekends = 1.0 |
                - encode_holidays = True: seperate int (>1.0) for each holiday
                - encode_holidays = False: unlabeled, raw holiday name strings returned

    Returns
    ----------
    pd.DataFrame with the original X concatinated with the holiday DataFrame, that has len(columns) == country_codes
    """

    properties = Transformation.Properties()

    def __init__(
        self,
        country_codes: Union[List[str], str],
        type: Union[str, LabelingMethod] = LabelingMethod.weekday_weekend_holiday,
        encode_holidays: bool = False,
    ) -> None:
        self.country_codes = wrap_in_list(country_codes)
        self.name = f"AddHolidayFeatures-{self.country_codes}"
        self.type = LabelingMethod.from_str(type)
        self.encode_holidays = encode_holidays

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        if self.type is LabelingMethod.holiday_binary:
            extra_holiday_features = get_multi_holidays(X.index, self.country_codes)
            extra_holiday_features[extra_holiday_features != 0.0] = 1.0

        elif self.type is LabelingMethod.weekday_weekend_holiday:
            extra_holiday_features = get_multi_holidays(X.index, self.country_codes)

            # All values that are bigger than 0 are holidays, but we don't want to differentiate between them
            extra_holiday_features[extra_holiday_features != 0.0] = 1.0

            extra_holiday_features = extra_holiday_features.mul(2).add(
                get_weekends(X.index), axis="index"
            )

        elif self.type is LabelingMethod.weekday_weekend_uniqueholiday:
            extra_holiday_features = get_multi_holidays(
                X.index, self.country_codes, encode=self.encode_holidays
            )

            extra_holiday_features.apply(
                lambda x: x[x == 0.0].add(get_weekends(X.index), axis="index")
            )
            # Pandas can only add to integer values, filter for non-holidays and add 1
            extra_holiday_features[
                extra_holiday_features.str.isnumeric()
                # type(extra_holiday_features) == (int, float)
            ] = extra_holiday_features[extra_holiday_features.str.isnumeric()].add(
                get_weekends(X.index), axis="index"
            )

        return pd.concat(
            [X, extra_holiday_features.add_suffix(f"_{self.type.value}")],
            axis="columns",
        )

    fit = fit_noop
    update = fit_noop
