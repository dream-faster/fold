from datetime import date
from enum import Enum
from typing import List, Union

import pandas as pd

from ..utils.list import swap_tuples, wrap_in_list
from .base import Transformation, fit_noop


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
    Adds holiday features for given region(s) as new column(s).


    Parameters
    ----------

    country_codes: List[str]
        List of country codes  (eg.: `US`, `DE`) for which to add holiday features.

    labeling: LabelingMethod
        How to label the holidays. Possible values:
        - holiday_binary: Workdays = 0 | National Holidays = 1
        - weekday_weekend_holiday: Workdays = 0 | Weekends = 1 | National Holidays == 2
        - weekday_weekend_uniqueholiday: Workdays = 0 | Weekends = 1 |
                - if `label_encode` == True: seperate int (>1) for each holiday
                - if `label_encode` == False: raw holiday names (as string)

    label_encode: bool (default=True)
        If True, national holidays are encoded as integers as well. If False, national holidays are returned as strings.

    """

    properties = Transformation.Properties()

    def __init__(
        self,
        country_codes: Union[List[str], str],
        labeling: Union[str, LabelingMethod] = LabelingMethod.weekday_weekend_holiday,
        label_encode: bool = True,
    ) -> None:
        self.country_codes = wrap_in_list(country_codes)
        self.name = f"AddHolidayFeatures-{self.country_codes}"
        self.type = LabelingMethod.from_str(labeling)
        self.label_encode = label_encode
        from holidays import country_holidays, list_supported_countries

        all_supported_countries = list_supported_countries()

        assert all(
            [country_code in all_supported_countries for country_code in country_codes]
        ), f"Country code not supported: {country_codes}"

        self.holiday_to_int_maps = [
            dict(
                swap_tuples(
                    enumerate(
                        sorted(
                            list(
                                set(
                                    country_holidays(
                                        country_code,
                                        years=list(range(1900, date.today().year)),
                                        language="en_US",
                                    ).values()
                                )
                            )
                        ),
                        start=1,
                    )
                )
            )
            for country_code in self.country_codes
        ]
        print(self.holiday_to_int_maps)

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        if self.type is LabelingMethod.holiday_binary:
            holidays = _get_holidays(
                X.index, self.country_codes, self.holiday_to_int_maps
            )
            holidays[holidays != 0] = 1
            return pd.concat([X, holidays], axis="columns")
        elif self.type is LabelingMethod.weekday_weekend_holiday:
            holidays = _get_holidays(
                X.index, self.country_codes, self.holiday_to_int_maps
            )

            # All values that are (bigger than 0 or a string) are holidays, but we don't want to differentiate between them
            holidays[holidays != 0] = 2

            holidays[holidays == 0] = holidays[holidays == 0].add(
                _get_weekends(X.index), axis="index"
            )
            return pd.concat([X, holidays], axis="columns")
        elif self.type is LabelingMethod.weekday_weekend_uniqueholiday:
            holidays = _get_holidays(
                X.index,
                self.country_codes,
                self.holiday_to_int_maps,
                encode=self.label_encode,
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
    encode: bool = False,
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
            name=f"{country_code}_holiday",
        )
        for country_code in country_codes
    ]
    for s in series:
        s.index = dates

    df = pd.concat(series, axis="columns").fillna(0)

    if encode:
        for country_code, holiday_to_int_map in zip(country_codes, holiday_to_int_maps):
            df[f"{country_code}_holiday"] = df[f"{country_code}_holiday"].map(
                holiday_to_int_map
            )

    return df
