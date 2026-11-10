"""Create a verified page-mapped gold set without changing the original labels."""

from __future__ import annotations

import json
from pathlib import Path


CORRECTIONS = {
    "q03": ("amfi_mutual_funds_made_simple_2026.pdf", 15),
    "q06": ("common_booklet_securities_market_2021.pdf", 14),
    "q073": ("sebi_how_to_invest_in_ipo_2022.pdf", 12),
    "q14": ("nse_basic_faqs_secondary_market_2023.pdf", 8),
    "q157": ("the_candlestick_trading_bible_kohanfx.pdf", 14),
    "q299": (
        "sebi_study_trading_behaviour_of_individual_traders_(fy25_26)_2026.pdf",
        17,
    ),
}


def main() -> None:
    source = Path("Golden_set_300.json")
    output = Path("Golden_set_300_mapped.json")
    rows = json.loads(source.read_text(encoding="utf-8"))
    changed = []
    for row in rows:
        correction = CORRECTIONS.get(row.get("id"))
        if correction is None:
            continue
        source_name, page = correction
        row["relevant_pages"] = [{"source": source_name, "page": page}]
        row["label_mapping_note"] = (
            "Verified answer-bearing page replaces a legacy TOC/front-matter "
            "or introduction-page mapping."
        )
        changed.append(row["id"])
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"corrected={len(changed)} ids={changed}")


if __name__ == "__main__":
    main()
