#!/usr/bin/env python3
"""Item pool for d_llm_federation_* -- 45 items (25 capitals, 20 elements), each with a hand-authored TRUE claim
and a PLANTED coherent-FAKE claim. Fakes use REAL, well-known "classic confusion" wrong answers (a real city that
is NOT the capital, or a real element's symbol that is NOT this element's) so every fake is plausible/coherent,
never nonsense -- verified by hand against well-established geography/chemistry facts, authored BEFORE any model
was queried (anti-tautology: ground truth never derived from a model's own output).
"""

# (country, true_capital, fake_capital) -- fake is a REAL city in that country, NOT its capital, chosen for
# real-world plausibility (former capital / largest city / most internationally famous city).
CAPITALS = [
    ("Turkey", "Ankara", "Istanbul"),
    ("Brazil", "Brasilia", "Rio de Janeiro"),
    ("Australia", "Canberra", "Sydney"),
    ("Canada", "Ottawa", "Toronto"),
    ("Switzerland", "Bern", "Zurich"),
    ("New Zealand", "Wellington", "Auckland"),
    ("Nigeria", "Abuja", "Lagos"),
    ("Kazakhstan", "Astana", "Almaty"),
    ("Myanmar", "Naypyidaw", "Yangon"),
    ("Pakistan", "Islamabad", "Karachi"),
    ("Morocco", "Rabat", "Casablanca"),
    ("Vietnam", "Hanoi", "Ho Chi Minh City"),
    ("Ecuador", "Quito", "Guayaquil"),
    ("Tanzania", "Dodoma", "Dar es Salaam"),
    ("United States", "Washington DC", "New York City"),
    ("India", "New Delhi", "Mumbai"),
    ("France", "Paris", "Marseille"),
    ("Japan", "Tokyo", "Osaka"),
    ("Germany", "Berlin", "Munich"),
    ("Italy", "Rome", "Milan"),
    ("China", "Beijing", "Shanghai"),
    ("Spain", "Madrid", "Barcelona"),
    ("Russia", "Moscow", "Saint Petersburg"),
    ("South Korea", "Seoul", "Busan"),
    ("Egypt", "Cairo", "Alexandria"),
]

# (element, true_symbol, fake_symbol) -- fake is a REAL symbol of a DIFFERENT real element, chosen for lexical
# (first-letter/first-two-letter naive guess) or visual/real-world-usage plausibility.
ELEMENTS = [
    ("Potassium", "K", "P"),        # naive first-letter guess -> Phosphorus (real)
    ("Sodium", "Na", "S"),          # naive first-letter guess -> Sulfur (real)
    ("Tin", "Sn", "Ti"),            # naive first-two-letters -> Titanium (real)
    ("Antimony", "Sb", "Sm"),       # visually similar -> Samarium (real)
    ("Tungsten", "W", "Tc"),        # T-word guess -> Technetium (real)
    ("Lead", "Pb", "Bi"),           # real-world substitution confusion -> Bismuth (real)
    ("Mercury", "Hg", "Mn"),        # M-word guess -> Manganese (real)
    ("Manganese", "Mn", "Mg"),      # extremely common real confusion -> Magnesium (real)
    ("Bismuth", "Bi", "B"),         # naive first-letter guess -> Boron (real)
    ("Arsenic", "As", "Ar"),        # naive first-two-letters -> Argon (real)
    ("Gold", "Au", "Ag"),           # classic precious-metal mixup -> Silver (real)
    ("Silver", "Ag", "Au"),         # reverse mixup -> Gold (real)
    ("Copper", "Cu", "Co"),         # visually similar symbol -> Cobalt (real)
    ("Iron", "Fe", "Ni"),           # alloy-adjacent confusion -> Nickel (real)
    ("Oxygen", "O", "Os"),          # visual trap -> Osmium (real)
    ("Hydrogen", "H", "He"),        # both light gases, common beginner mixup -> Helium (real)
    ("Carbon", "C", "Ca"),          # visual trap -> Calcium (real)
    ("Nitrogen", "N", "Ne"),        # both "air" gases, visual trap -> Neon (real)
    ("Helium", "He", "H"),          # reverse of Hydrogen mixup
    ("Calcium", "Ca", "C"),         # reverse of Carbon mixup
]

SYS_MSG = ("Answer factual questions as concisely as instructed. Follow the requested output format exactly, "
           "with no extra words or explanation.")

def build_claims():
    """Returns list of dicts: one per (item, true/fake) -- 90 total (45 true + 45 fake)."""
    claims = []
    for country, true_cap, fake_cap in CAPITALS:
        claims.append(dict(category="capitals", item=country, ground_truth=true_cap,
                            claim_answer=true_cap, label_fake=False,
                            claim_text=f"The capital of {country} is {true_cap}.",
                            fwd_q=f"What is the capital of {country}? Answer with just the city name, nothing else.",
                            decorr_q_tmpl="Which country has {ans} as its capital? Answer with just the country name, nothing else.",
                            answer_kind="city"))
        claims.append(dict(category="capitals", item=country, ground_truth=true_cap,
                            claim_answer=fake_cap, label_fake=True,
                            claim_text=f"The capital of {country} is {fake_cap}.",
                            fwd_q=f"What is the capital of {country}? Answer with just the city name, nothing else.",
                            decorr_q_tmpl="Which country has {ans} as its capital? Answer with just the country name, nothing else.",
                            answer_kind="city"))
    for elem, true_sym, fake_sym in ELEMENTS:
        claims.append(dict(category="elements", item=elem, ground_truth=true_sym,
                            claim_answer=true_sym, label_fake=False,
                            claim_text=f"The chemical symbol for {elem} is {true_sym}.",
                            fwd_q=f"What is the chemical symbol for the element {elem}? Answer with just the 1-2 letter symbol, nothing else.",
                            decorr_q_tmpl="Which chemical element has the symbol '{ans}'? Answer with just the element name, nothing else.",
                            answer_kind="symbol"))
        claims.append(dict(category="elements", item=elem, ground_truth=true_sym,
                            claim_answer=fake_sym, label_fake=True,
                            claim_text=f"The chemical symbol for {elem} is {fake_sym}.",
                            fwd_q=f"What is the chemical symbol for the element {elem}? Answer with just the 1-2 letter symbol, nothing else.",
                            decorr_q_tmpl="Which chemical element has the symbol '{ans}'? Answer with just the element name, nothing else.",
                            answer_kind="symbol"))
    return claims

if __name__ == "__main__":
    c = build_claims()
    n_fake = sum(1 for x in c if x["label_fake"])
    print(f"n_claims={len(c)}  n_fake={n_fake}  n_true={len(c)-n_fake}  n_capitals_items={len(CAPITALS)}  n_elements_items={len(ELEMENTS)}")
    for x in c[:4] + c[-4:]:
        print(" ", x["claim_text"], "| label_fake=", x["label_fake"])
