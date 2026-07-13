"""conn_country → IANA-region map (#90 AC2, decision 498eb9d1).

The map must hold *real* IANA regions, never fixed-offset ``Etc/GMT`` keys: a
fixed offset erases DST and historical transitions, so the KZ 2024-03-01
UTC+6→+5 change (the motivating case) would be invisible. zoneinfo needs the
region name to apply the offset that was in force on the play's own date.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

from music_intel_mcp.timezones import COUNTRY_ZONES, zone_for


def test_no_fixed_offset_zones():
    for country, zone in COUNTRY_ZONES.items():
        assert not zone.startswith("Etc/"), f"{country} → {zone} is a fixed offset"


def test_every_zone_is_a_loadable_iana_region():
    for zone in COUNTRY_ZONES.values():
        ZoneInfo(zone)  # raises ZoneInfoNotFoundError if not a real key


def test_kazakhstan_maps_to_almaty():
    # The motivating market — must resolve to a DST/transition-aware region.
    assert zone_for("KZ") == "Asia/Almaty"


def test_zone_for_is_case_insensitive_and_null_safe():
    assert zone_for("kz") == "Asia/Almaty"
    assert zone_for("KZ") == zone_for("kz")
    assert zone_for(None) is None
    assert zone_for("") is None
    assert zone_for("ZZ") is None  # unmapped country → no zone (caller falls back)
