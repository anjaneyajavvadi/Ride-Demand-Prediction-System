# src/features/schema.py

FEATURE_SCHEMA = {
    "hour_of_day":      {"dtype": "int8",   "min": 0,  "max": 23},
    "day_of_week":      {"dtype": "int8",   "min": 0,  "max": 6},
    "month":            {"dtype": "int8",   "min": 1,  "max": 12},
    "is_weekend":       {"dtype": "bool",   "min": 0,  "max": 1},
    "is_rush_hour":     {"dtype": "bool",   "min": 0,  "max": 1},
    "lag_1h":           {"dtype": "float64","min": 0,  "max": None},
    "lag_24h":          {"dtype": "float64","min": 0,  "max": None},
    "lag_168h":         {"dtype": "float64","min": 0,  "max": None},
    "rolling_mean_3h":  {"dtype": "float64","min": 0,  "max": None},
    "rolling_mean_24h": {"dtype": "float64","min": 0,  "max": None},
    "PULocationID":     {"dtype": "int32",  "min": 1,  "max": 263},
}

FEATURE_COLUMNS = list(FEATURE_SCHEMA.keys())
