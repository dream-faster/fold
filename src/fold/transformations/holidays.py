# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from datetime import date
from enum import Enum
from typing import List, Optional, Tuple, Union

import pandas as pd

from ..base import Artifact, Transformation, Tunable, fit_noop
from ..utils.list import swap_tuples, wrap_in_list


class LabelingMethod(Enum):
    """
    Parameters
    ----------
    holiday_binary: string
        * Workdays = 0
        * National Holidays = 1
    weekday_weekend_holiday: string
        * Workdays = 0
        * Weekends = 1
        * National Holidays == 2
    weekday_weekend_uniqueholiday: string
        * Workdays = 0
        * Weekends = 1
        * National Holidays == Unique int (>1)
    weekday_weekend_uniqueholiday_string: string
        * Workdays = 0
        * Weekends = 1
        * National Holidays == string
    """

    holiday_binary = "holiday_binary"
    weekday_weekend_holiday = "weekday_weekend_holiday"
    weekday_weekend_uniqueholiday = "weekday_weekend_uniqueholiday"
    weekday_weekend_uniqueholiday_string = "weekday_weekend_uniqueholiday_string"

    @staticmethod
    def from_str(value: Union[str, "LabelingMethod"]) -> "LabelingMethod":
        if isinstance(value, LabelingMethod):
            return value
        for strategy in LabelingMethod:
            if strategy.value == value:
                return strategy
        else:
            raise ValueError(f"Unknown HolidayType: {value}")


class AddHolidayFeatures(Transformation, Tunable):
    """
    Adds holiday features for given region(s) as new column(s).
    It uses the pattern "holiday_{country_code}" for naming the columns.


    Parameters
    ----------

    country_codes: List[str]
        List of country codes  (eg.: `US`, `DE`) for which to add holiday features.
    labeling: LabelingMethod
        * holiday_binary: Workdays = 0 | National Holidays = 1
        * weekday_weekend_holiday: Workdays = 0 | Weekends = 1 | National Holidays == 2
        * weekday_weekend_uniqueholiday: Workdays = 0 | Weekends = 1 | National Holidays == Unique int (>1)
        * weekday_weekend_uniqueholiday_string: Workdays = 0 | Weekends = 1 | National Holidays == string

    """

    properties = Transformation.Properties(requires_X=False)

    def __init__(
        self,
        country_codes: Union[List[str], str],
        labeling: Union[str, LabelingMethod] = LabelingMethod.weekday_weekend_holiday,
        params_to_try: Optional[dict] = None,
    ) -> None:
        self.country_codes = [
            country_code.upper() for country_code in wrap_in_list(country_codes)
        ]
        self.name = f"AddHolidayFeatures-{self.country_codes}"
        self.labeling = LabelingMethod.from_str(labeling)
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
        self.params_to_try = params_to_try
        super().__init__()

    def transform(
        self, X: pd.DataFrame, in_sample: bool
    ) -> Tuple[pd.DataFrame, Optional[Artifact]]:
        if self.labeling == LabelingMethod.holiday_binary:
            holidays = _get_holidays(
                X.index, self.country_codes, self.holiday_to_int_maps, encode=True
            )
            holidays[holidays != 0] = 1

            return pd.concat([X, holidays], axis="columns"), None
        elif self.labeling == LabelingMethod.weekday_weekend_holiday:
            holidays = _get_holidays(
                X.index, self.country_codes, self.holiday_to_int_maps, encode=True
            )

            # All values that are (bigger than 0 or a string) are holidays, but we don't want to differentiate between them
            holidays[holidays != 0] = 2

            holidays[holidays == 0] = holidays[holidays == 0].add(
                _get_weekends(X.index), axis="index"
            )
            return pd.concat([X, holidays], axis="columns"), None
        elif (
            self.labeling == LabelingMethod.weekday_weekend_uniqueholiday
            or self.labeling == LabelingMethod.weekday_weekend_uniqueholiday_string
        ):
            holidays = _get_holidays(
                X.index,
                self.country_codes,
                self.holiday_to_int_maps,
                encode=self.labeling == LabelingMethod.weekday_weekend_uniqueholiday,
            )

            weekends = _get_weekends(X.index)
            holidays[holidays == 0] = holidays[holidays == 0].add(
                weekends, axis="index"
            )
            return pd.concat([X, holidays], axis="columns"), None
        else:
            raise ValueError(f"Unknown HolidayType: {self.labeling}")

    fit = fit_noop
    update = fit_noop

    def get_params(self) -> dict:
        return {
            "country_codes": self.country_codes,
            "labeling": self.labeling,
        }


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
