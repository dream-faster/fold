from finml_utils.enums import ParsableEnum


class DateTimeFeature(ParsableEnum):
    second = "second"
    minute = "minute"
    hour = "hour"
    day_of_week = "day_of_week"
    day_of_month = "day_of_month"
    day_of_year = "day_of_year"
    week = "week"
    week_of_year = "week_of_year"
    month = "month"
    quarter = "quarter"
    year = "year"
