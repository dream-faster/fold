from enum import Enum
from typing import List, Union

import holidays
import pandas as pd

from ..utils.list import wrap_in_list
from .base import Transformation, fit_noop


def get_weekends(dates: pd.DatetimeIndex) -> pd.Series:
    return pd.Series((dates.weekday > 4).astype(int), index=dates)


# def get_holidays(dates: pd.DatetimeIndex, country_codes: List[str]) -> pd.DataFrame:
#     pd.DataFrame.from_dict(
#         holidays.country_holidays(country_code, years=dates.year.unique().to_list())
#     )
#     return pd.DataFrame(
#         {
#             country_code: dates.normalize()
#             .isin(
#                 holidays.country_holidays(
#                     country_code, years=dates.year.unique().to_list()
#                 )
#             )
#             .astype(int)
#             for country_code in country_codes
#         },
#         index=dates,
#     )


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
    Adds holiday features for given regions as a column.
    For each passed country_code it produces a column of Boolean if the given index is a holiday or not.
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

        elif self.type is HolidayTypes.holiday_weekend:
            holiday_df = get_multi_holidays(X.index, self.country_codes)

            # All values that are bigger than 0 are holidays, but we don't want to differentiate between them
            holiday_df[holiday_df != 0.0] = 1.0

            extra_holiday_features = holiday_df.mul(2).add(
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
