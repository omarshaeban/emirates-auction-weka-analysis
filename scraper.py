"""Reconstruction of source pasted by the student in the 2025 coursework chat.

The original used a local chromedriver path and an exact set of CSS/HTML selectors.
This version preserves those selectors and provides CLI flags, offline HTML support,
and explicit failure if a scrape returns no vehicle cards.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from bs4 import BeautifulSoup

SOURCE_URL = "https://www.emiratesauction.com/ar/motors"
OUTPUT_FIELDS = ["title", "sequence", "time_left", "bids", "price"]


def parse_html(html: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.find_all("div", class_="list-card-container")
    records: list[dict[str, str]] = []
    for card in cards:
        title_tag = card.find("h2", id=re.compile(r"^title_"))
        title = title_tag.get_text(" ", strip=True) if title_tag else ""
        if not title:
            titled = card.find(attrs={"aria-label": True})
            title = (titled.get("aria-label", "") if titled else "").strip()
        full_text = card.get_text(separator=" ", strip=True)
        seq_match = re.search(r"التسلسل\s*:?\s*(\d+)", full_text)
        seq = seq_match.group(1) if seq_match else ""
        time_match = re.search(r"الوقت\s*المتبقي\s*:?\s*(.*?)(?=مزايدات\s*:|$)", full_text)
        time_left = time_match.group(1).strip() if time_match else ""
        bids_match = re.search(r"مزايدات\s*:?\s*(\d+)", full_text)
        bids = bids_match.group(1) if bids_match else ""
        price = ""
        price_div = card.find("div", id=f"CARD_PRICE_{seq}") if seq else None
        if price_div:
            for span in price_div.find_all("span"):
                classes = span.get("class", [])
                if "text-decoration-line-through" in classes:
                    continue
                raw = span.get_text(" ", strip=True)
                match = re.search(r"[\d,٬]+", raw)
                if match:
                    price = match.group().replace(",", "").replace("٬", "")
                    break
        records.append({"title": title, "sequence": seq,
                        "time_left": time_left, "bids": bids, "price": price})
    return records


def save_csv(records: list[dict[str, str]], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, delimiter=";")
        writer.writeheader()
        writer.writerows(records)


def fetch_browser_html(url: str, timeout: int) -> str:
    # Imported only when live scraping; offline mode and tests need no Chrome.
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    options = webdriver.ChromeOptions()
    with webdriver.Chrome(options=options) as driver:
        driver.get(url)
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CLASS_NAME, "list-card-container"))
        )
        return driver.page_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover vehicle listing fields from Emirates Auction")
    parser.add_argument("--html", help="Local HTML file for offline extraction and selector debugging")
    parser.add_argument("--url", default=SOURCE_URL)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--output", default="emirates_auction_data.csv")
    args = parser.parse_args()
    html = Path(args.html).read_text(encoding="utf-8") if args.html else fetch_browser_html(args.url, args.timeout)
    records = parse_html(html)
    if not records:
        raise SystemExit("No vehicle cards found: check page structure; CSV NOT saved.")
    save_csv(records, args.output)
    print(f"Saved {len(records)} listings to {args.output}")


if __name__ == "__main__":
    main()
