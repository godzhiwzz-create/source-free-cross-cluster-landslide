"""Paper protocol constants."""

CLUSTERS = (
    "Africa",
    "Americas",
    "CentralAsia",
    "Europe",
    "Oceania",
    "SoutheastAsia",
)

# Ten Sentinel-2 bands plus DEM. Sentinel-1 (10, 11) and SCL (13) are excluded.
SELECTED_CHANNELS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12)
LANDSLIDE_PIXEL_MIN = 50
DEFAULT_THRESHOLDS = tuple(round(i * 0.05, 2) for i in range(1, 20))
