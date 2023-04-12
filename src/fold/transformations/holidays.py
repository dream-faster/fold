from datetime import date
from enum import Enum
from typing import List, Union

import pandas as pd

from ..base import Transformation, fit_noop
from ..utils.list import swap_tuples, wrap_in_list


class LabelingMethod(Enum):
    holiday_binary = "holiday_binary"
    """Workdays = 0 | National Holidays = 1"""
    weekday_weekend_holiday = "weekday_weekend_holiday"
    """Workdays = 0 | Weekends = 1 | National Holidays == 2"""
    weekday_weekend_uniqueholiday = "weekday_weekend_uniqueholiday"
    """Workdays = 0 | Weekends = 1 | National Holidays == Unique int (>1)"""
    weekday_weekend_uniqueholiday_string = "weekday_weekend_uniqueholiday_string"
    """Workdays = 0 | Weekends = 1 | National Holidays == string)"""

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
    Adds holiday features for given region(s) as new column(s).
    It uses the pattern "holiday_{country_code}" for naming the columns.


    Parameters
    ----------

    country_codes: List[str]
        List of country codes  (eg.: `US`, `DE`) for which to add holiday features.

    labeling: LabelingMethod
        How to label the holidays. Possible values:
        - holiday_binary: Workdays = 0 | National Holidays = 1
        - weekday_weekend_holiday: Workdays = 0 | Weekends = 1 | National Holidays == 2
        - weekday_weekend_uniqueholiday: Workdays = 0 | Weekends = 1 | National Holidays == Unique int (>1)
        - weekday_weekend_uniqueholiday_string: Workdays = 0 | Weekends = 1 | National Holidays == string

    """

    properties = Transformation.Properties(requires_X=False)

    def __init__(
        self,
        country_codes: Union[List[str], str],
        labeling: Union[str, LabelingMethod] = LabelingMethod.weekday_weekend_holiday,
    ) -> None:
        self.country_codes = [
            country_code.upper() for country_code in wrap_in_list(country_codes)
        ]
        self.name = f"AddHolidayFeatures-{self.country_codes}"
        self.type = LabelingMethod.from_str(labeling)
        from holidays import country_holidays, list_supported_countries

        all_supported_countries = list_supported_countries()

        assert all(
            [
                country_code in all_supported_countries
                for country_code in self.country_codes
            ]
        ), f"Country code not supported: {country_codes}"

        self.holiday_to_int_maps = [
            dict(
                swap_tuples(
                    enumerate(
                        sorted(
                            set(
                                country_holidays(
                                    country_code,
                                    years=list(range(1900, date.today().year)),
                                    language="en_US",
                                ).values()
                            )
                        ),
                        start=1,
                    )
                )
            )
            for country_code in self.country_codes
        ]

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        if self.type == LabelingMethod.holiday_binary:
            holidays = _get_holidays(
                X.index, self.country_codes, self.holiday_to_int_maps, encode=True
            )
            holidays[holidays != 0] = 1

            return pd.concat([X, holidays], axis="columns")
        elif self.type == LabelingMethod.weekday_weekend_holiday:
            holidays = _get_holidays(
                X.index, self.country_codes, self.holiday_to_int_maps, encode=True
            )

            # All values that are (bigger than 0 or a string) are holidays, but we don't want to differentiate between them
            holidays[holidays != 0] = 2

            holidays[holidays == 0] = holidays[holidays == 0].add(
                _get_weekends(X.index), axis="index"
            )
            return pd.concat([X, holidays], axis="columns")
        elif (
            self.type == LabelingMethod.weekday_weekend_uniqueholiday
            or self.type == LabelingMethod.weekday_weekend_uniqueholiday_string
        ):
            holidays = _get_holidays(
                X.index,
                self.country_codes,
                self.holiday_to_int_maps,
                encode=self.type == LabelingMethod.weekday_weekend_uniqueholiday,
            )

            weekends = _get_weekends(X.index)
            holidays[holidays == 0] = holidays[holidays == 0].add(
                weekends, axis="index"
            )
            return pd.concat([X, holidays], axis="columns")
        else:
            raise ValueError(f"Unknown HolidayType: {self.type}")

    fit = fit_noop
    update = fit_noop


def _get_weekends(dates: pd.DatetimeIndex) -> pd.Series:
    return pd.Series((dates.weekday > 4).astype(int), index=dates)


def _get_holidays(
    dates: pd.DatetimeIndex,
    country_codes: List[str],
    holiday_to_int_maps: List[dict],
    encode: bool,
) -> pd.DataFrame:
    from holidays import country_holidays

    series = [
        pd.Series(
            country_holidays(
                country_code,
                years=dates.year.unique().to_list(),
                language="en_US",
            ),
            index=dates.date,
            name=f"holiday_{country_code}",
        ).set_axis(dates)
        for country_code in country_codes
    ]

    df = pd.concat(series, axis="columns").fillna(0)

    if encode:
        for country_code, holiday_to_int_map in zip(country_codes, holiday_to_int_maps):
            col = df[f"holiday_{country_code}"]
            if not col.dtype == "object":
                continue  # we don't use strings, there's nothing to encode
            df[f"holiday_{country_code}"] = (
                col.map(holiday_to_int_map).fillna(0).astype(int)
            )

    return df
