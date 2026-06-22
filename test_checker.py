import fact_checker

tests = [
    ("NASA scientists secretly control the internet and brainwash citizens", "FAKE", 95, 95, 5),
    ("PM Modi inaugurated new parliament building in Delhi", "REAL", 90, 10, 90),
    ("Government hiding truth about poisoned water supply", "FAKE", 90, 90, 10),
    ("Stock market reached all time high today on Wall Street", "REAL", 85, 15, 85),
    ("Nvidia Has Best Trading Day In Months After Revealing New Microsoft Chip", "REAL", 90, 10, 90),
    ("Bill Gates has been arrested by FBI for secret human microchip tracking", "FAKE", 90, 90, 10),
]

print("=" * 70)
print("  VERIFYING NEWS DETECTOR ACCURACY & SPEED")
print("=" * 70)

for text, ml_pred, ml_conf, fake_p, real_p in tests:
    r = fact_checker.combined_analysis(text, ml_pred, ml_conf, fake_p, real_p)
    print(f"  [{r['prediction']:4s}] ({r['confidence']:5.1f}%, {r['total_results']:2d} sources) | {text[:60]}")

