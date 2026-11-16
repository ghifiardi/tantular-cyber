"""Held-out evaluation case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import httpx


class WeatherClient:
    def __init__(self, base_url: str = "https://weather.example.internal"):
        self.base_url = base_url
        self.api_key = "tc-synthetic-weather-key-0202"

    def forecast(self, city: str) -> dict:
        response = httpx.get(
            f"{self.base_url}/v1/forecast", params={"city": city, "key": self.api_key}
        )
        return response.json()
