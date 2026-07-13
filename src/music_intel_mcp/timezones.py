"""conn_country → IANA-region map for local-wall-clock bucketize (#90, AC2).

The Spotify Extended Streaming History source carries an ISO-3166 alpha-2
``conn_country`` per play. To bucketize a play by the listener's *local* hour we
shift its UTC timestamp into the region that country implies. This module is the
country → region lookup; the shift itself and the fallback chain live in
``analyzer._temporal_plays``.

Two deliberate constraints (decision ``498eb9d1``):

- **Real IANA regions only, never ``Etc/GMT`` fixed offsets.** A fixed offset
  freezes one UTC delta forever, erasing DST and historical transitions. The
  motivating case is Kazakhstan's 2024-03-01 UTC+6→+5 change: only a region
  (``Asia/Almaty``) lets zoneinfo pick the offset in force on the play's date.
- **One representative region per country.** For multi-zone countries we take the
  most-populous zone. This is an approximation — a play from western Kazakhstan
  or eastern Russia lands in the wrong sub-zone — but day-part bucketize tolerates
  a ±1-2h skew far better than the fixed-offset alternative, and per-city zone
  inference is out of scope for V1. Unmapped/multi-zone-ambiguous countries fall
  back (in the caller) to the user's modal zone, so no play is stranded.

Not exhaustive: ~80 countries covering the major music markets plus all of
Europe. An unmapped country returns ``None`` from :func:`zone_for` and the caller
applies the modal-zone fallback rather than guessing.
"""

from __future__ import annotations

# ISO-3166 alpha-2 (upper) → representative IANA region (most-populous zone).
COUNTRY_ZONES: dict[str, str] = {
    # --- Central Asia / motivating market -------------------------------- #
    "KZ": "Asia/Almaty",
    "KG": "Asia/Bishkek",
    "UZ": "Asia/Tashkent",
    "TJ": "Asia/Dushanbe",
    "TM": "Asia/Ashgabat",
    "AZ": "Asia/Baku",
    "GE": "Asia/Tbilisi",
    "AM": "Asia/Yerevan",
    # --- Europe ---------------------------------------------------------- #
    "GB": "Europe/London",
    "IE": "Europe/Dublin",
    "PT": "Europe/Lisbon",
    "ES": "Europe/Madrid",
    "FR": "Europe/Paris",
    "BE": "Europe/Brussels",
    "NL": "Europe/Amsterdam",
    "LU": "Europe/Luxembourg",
    "DE": "Europe/Berlin",
    "CH": "Europe/Zurich",
    "AT": "Europe/Vienna",
    "IT": "Europe/Rome",
    "MT": "Europe/Malta",
    "DK": "Europe/Copenhagen",
    "NO": "Europe/Oslo",
    "SE": "Europe/Stockholm",
    "FI": "Europe/Helsinki",
    "IS": "Atlantic/Reykjavik",
    "PL": "Europe/Warsaw",
    "CZ": "Europe/Prague",
    "SK": "Europe/Bratislava",
    "HU": "Europe/Budapest",
    "SI": "Europe/Ljubljana",
    "HR": "Europe/Zagreb",
    "BA": "Europe/Sarajevo",
    "RS": "Europe/Belgrade",
    "ME": "Europe/Podgorica",
    "MK": "Europe/Skopje",
    "AL": "Europe/Tirane",
    "GR": "Europe/Athens",
    "BG": "Europe/Sofia",
    "RO": "Europe/Bucharest",
    "MD": "Europe/Chisinau",
    "UA": "Europe/Kyiv",
    "BY": "Europe/Minsk",
    "LT": "Europe/Vilnius",
    "LV": "Europe/Riga",
    "EE": "Europe/Tallinn",
    "RU": "Europe/Moscow",
    "TR": "Europe/Istanbul",
    "CY": "Asia/Nicosia",
    # --- Middle East ----------------------------------------------------- #
    "IL": "Asia/Jerusalem",
    "AE": "Asia/Dubai",
    "SA": "Asia/Riyadh",
    "QA": "Asia/Qatar",
    "KW": "Asia/Kuwait",
    "IR": "Asia/Tehran",
    "IQ": "Asia/Baghdad",
    "JO": "Asia/Amman",
    "LB": "Asia/Beirut",
    # --- Asia ------------------------------------------------------------ #
    "IN": "Asia/Kolkata",
    "PK": "Asia/Karachi",
    "BD": "Asia/Dhaka",
    "LK": "Asia/Colombo",
    "NP": "Asia/Kathmandu",
    "TH": "Asia/Bangkok",
    "VN": "Asia/Ho_Chi_Minh",
    "MY": "Asia/Kuala_Lumpur",
    "SG": "Asia/Singapore",
    "ID": "Asia/Jakarta",
    "PH": "Asia/Manila",
    "HK": "Asia/Hong_Kong",
    "TW": "Asia/Taipei",
    "CN": "Asia/Shanghai",
    "JP": "Asia/Tokyo",
    "KR": "Asia/Seoul",
    # --- Oceania --------------------------------------------------------- #
    "AU": "Australia/Sydney",
    "NZ": "Pacific/Auckland",
    # --- Africa ---------------------------------------------------------- #
    "EG": "Africa/Cairo",
    "MA": "Africa/Casablanca",
    "DZ": "Africa/Algiers",
    "TN": "Africa/Tunis",
    "NG": "Africa/Lagos",
    "GH": "Africa/Accra",
    "KE": "Africa/Nairobi",
    "ZA": "Africa/Johannesburg",
    # --- Americas -------------------------------------------------------- #
    "US": "America/New_York",
    "CA": "America/Toronto",
    "MX": "America/Mexico_City",
    "GT": "America/Guatemala",
    "CO": "America/Bogota",
    "PE": "America/Lima",
    "EC": "America/Guayaquil",
    "VE": "America/Caracas",
    "BR": "America/Sao_Paulo",
    "AR": "America/Argentina/Buenos_Aires",
    "CL": "America/Santiago",
    "UY": "America/Montevideo",
    "PY": "America/Asuncion",
    "BO": "America/La_Paz",
    "CR": "America/Costa_Rica",
    "PA": "America/Panama",
    "DO": "America/Santo_Domingo",
    "PR": "America/Puerto_Rico",
}


def zone_for(country: str | None) -> str | None:
    """Representative IANA region for an ISO-3166 alpha-2 country code.

    Case-insensitive. Returns ``None`` for a null/blank/unmapped country so the
    caller can apply its own fallback (the user's modal zone) rather than this
    module inventing a default.
    """
    if not country:
        return None
    return COUNTRY_ZONES.get(country.strip().upper())
