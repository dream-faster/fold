# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.

from __future__ import annotations

from datetime import date
from typing import List, Optional, Union

import pandas as pd

from ..base import Transformation, Tunable, fit_noop
from ..utils.enums import ParsableEnum
from ..utils.list import swap_tuples, wrap_in_list


class LabelingMethod(ParsableEnum):
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

    def __init__(
        self,
        country_codes: Union[List[str], str],
        labeling: Union[str, LabelingMethod] = LabelingMethod.weekday_weekend_holiday,
        keep_original: bool = True,
        name: Optional[str] = None,
        params_to_try: Optional[dict] = None,
    ) -> None:
        self.country_codes = [
            country_code.upper() for country_code in wrap_in_list(country_codes)
        ]
        self.labeling = LabelingMethod.from_str(labeling)
        from holidays import country_holidays, list_supported_countries

        all_supported_countries = list(list_supported_countries().keys())

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
        self.keep_original = keep_original
        self.name = name or f"AddHolidayFeatures-{self.country_codes}"
        self.properties = Transformation.Properties(requires_X=False)

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        if self.labeling == LabelingMethod.holiday_binary:
            holidays = _get_holidays(
                X.index, self.country_codes, self.holiday_to_int_maps, encode=True
            )
            holidays[holidays != 0] = 1

        elif self.labeling == LabelingMethod.weekday_weekend_holiday:
            holidays = _get_holidays(
                X.index, self.country_codes, self.holiday_to_int_maps, encode=True
            )

            # All values that are (bigger than 0 or a string) are holidays, but we don't want to differentiate between them
            holidays[holidays != 0] = 2

            holidays[holidays == 0] = holidays[holidays == 0].add(
                _get_weekends(X.index), axis="index"
            )

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
        else:
            raise ValueError(f"Unknown HolidayType: {self.labeling}")

        concatenated = (
            pd.concat([X, holidays], axis="columns") if self.keep_original else holidays
        )
        return concatenated

    fit = fit_noop
    update = fit_noop


class AddExchangeHolidayFeatures(Transformation, Tunable):
    """
    Adds holiday features for given exchange(s) as new column(s).
    It uses the pattern "holiday_{exchange}" for naming the columns.


    Parameters
    ----------

    exchange_codes: List[str]
        List of exchange codes  (eg.: `NYSE`) for which to add holiday features.
    labeling: LabelingMethod
        * holiday_binary: Workdays = 0 | National Holidays = 1
        * weekday_weekend_holiday: Workdays = 0 | Weekends = 1 | National Holidays == 2
    """

    def __init__(
        self,
        exchange_codes: Union[List[str], str],
        labeling: Union[str, LabelingMethod] = LabelingMethod.weekday_weekend_holiday,
        keep_original: bool = True,
        name: Optional[str] = None,
        params_to_try: Optional[dict] = None,
    ) -> None:
        self.exchange_codes = wrap_in_list(exchange_codes)
        self.labeling = LabelingMethod.from_str(labeling)
        import pandas_market_calendars as mcal

        all_supported_exchanges = mcal.get_calendar_names()

        if "all" in self.exchange_codes:
            self.exchange_codes = all_supported_exchanges

        assert self.labeling in [
            LabelingMethod.holiday_binary,
            LabelingMethod.weekday_weekend_holiday,
        ], "Only holiday_binary and weekday_weekend_holiday are supported for stockexchanges"
        assert all(
            [
                country_code in all_supported_exchanges
                for country_code in self.exchange_codes
            ]
        ), f"Exchange code not supported: {exchange_codes}"

        self.params_to_try = params_to_try
        self.keep_original = keep_original
        self.name = name or f"AddExchangeHolidayFeatures-{self.exchange_codes}"
        self.properties = Transformation.Properties(requires_X=False)

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        if self.labeling == LabelingMethod.holiday_binary:
            holidays = _get_exchange_holidays(X.index, self.exchange_codes)
            holidays[holidays != 0] = 1

        elif self.labeling == LabelingMethod.weekday_weekend_holiday:
            holidays = _get_exchange_holidays(X.index, self.exchange_codes)

            # All values that are (bigger than 0 or a string) are holidays, but we don't want to differentiate between them
            holidays[holidays != 0] = 2

            holidays[holidays == 0] = holidays[holidays == 0].add(
                _get_weekends(X.index), axis="index"
            )

        else:
            raise ValueError(f"Unknown HolidayType: {self.labeling}")

        concatenated = (
            pd.concat([X, holidays], axis="columns") if self.keep_original else holidays
        )
        return concatenated

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


def _get_exchange_holidays(
    dates: pd.DatetimeIndex, exchange_codes: List[str]
) -> pd.DataFrame:
    import pandas_market_calendars as mcal

    return pd.concat(
        [
            mcal.get_calendar(exchange)
            .schedule(start_date=dates[0], end_date=dates[-1])
            .iloc[:, 0]
            .reindex(pd.date_range(dates[0], dates[-1]))
            .isna()
            .astype(int)
            .rename(f"holiday_{exchange}")
            for exchange in exchange_codes
        ],
        axis="columns",
    )
