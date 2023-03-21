from enum import Enum
from typing import List, Union

import holidays
import pandas as pd

from ..utils.list import wrap_in_list
from .base import Transformation, fit_noop


def get_weekends(dates: pd.DatetimeIndex) -> pd.Series:
    return pd.Series((dates.weekday > 4).astype(int), index=dates)


def encode_series(series: pd.Series) -> pd.Series:
    return series.astype("category").cat.codes


def get_multi_holidays(
    dates: pd.DatetimeIndex, country_codes: List[str]
) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(
        {
            country_code: holidays.country_holidays(
                country_code,
                years=dates.year.unique().to_list(),
                language="en_US",
            )
            for country_code in country_codes
        },
    )

    # Turn individual holiday names into category than reindex with the original dates
    df = df.apply(encode_series).add(1).reindex(dates)

    # When frequency is not daily we have to group and forward fill so that each hour/day/minute of the day has the same value
    df["datetime"] = df.index
    df = df.groupby(df["datetime"].dt.date).ffill().fillna(0.0)
    df = df.drop("datetime", axis="columns")
    return df


class HolidayTypes(Enum):
    holiday_binary = "holiday_binary"
    holiday_weekend = "holiday_weekend"
    holidays_differentiated = "holidays_differentiated"

    @staticmethod
    def from_str(value: Union[str, "HolidayTypes"]) -> "HolidayTypes":
        if isinstance(value, HolidayTypes):
            return value
        for strategy in HolidayTypes:
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
        - holiday_weekend: Non-special days = 0.0 | Weekends = 1.0 | holidays == 2.0 | holidays + weekend == 3.0
        - holidays_differentiated: Non-special days = 0.0 | Weekends = 1.0 | holidays = 2.0 + according to holiday type.

    Returns
    ----------
    pd.DataFrame with the original X concatinated with the holiday DataFrame, that has len(columns) == country_codes
    """

    properties = Transformation.Properties()

    def __init__(
        self,
        country_codes: Union[List[str], str],
        type: Union[str, HolidayTypes] = HolidayTypes.holiday_weekend,
    ) -> None:
        self.country_codes = wrap_in_list(country_codes)
        self.name = f"AddHolidayFeatures-{self.country_codes}"
        self.type = HolidayTypes.from_str(type)

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        if self.type is HolidayTypes.holiday_binary:
            extra_holiday_features = get_multi_holidays(X.index, self.country_codes)
            extra_holiday_features[extra_holiday_features != 0.0] = 1.0

        elif self.type is HolidayTypes.holiday_weekend:
            extra_holiday_features = get_multi_holidays(X.index, self.country_codes)

            # All values that are bigger than 0 are holidays, but we don't want to differentiate between them
            extra_holiday_features[extra_holiday_features != 0.0] = 1.0

            extra_holiday_features = extra_holiday_features.mul(2).add(
                get_weekends(X.index), axis="index"
            )

        elif self.type is HolidayTypes.holidays_differentiated:
            extra_holiday_features = get_multi_holidays(
                X.index, self.country_codes
            ).add(get_weekends(X.index), axis="index")

        return pd.concat(
            [X, extra_holiday_features.add_suffix(f"_{self.type.value}")],
            axis="columns",
        )

    fit = fit_noop
    update = fit_noop
