"""AQI category helpers."""


def aqi_category(aqi: float) -> tuple[str, str]:
    bands = [
        (50, "Good", "#00e400"),
        (100, "Moderate", "#ffff00"),
        (150, "Unhealthy (Sensitive)", "#ff7e00"),
        (200, "Unhealthy", "#ff0000"),
        (300, "Very Unhealthy", "#8f3f97"),
        (10**9, "Hazardous", "#7e0023"),
    ]
    for hi, label, color in bands:
        if aqi <= hi:
            return label, color
    return "Hazardous", "#7e0023"
