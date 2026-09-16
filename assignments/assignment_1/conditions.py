"""Which run belongs to which condition, and how each condition is drawn."""

from dataclasses import dataclass
from typing import cast

import pandas as pd

ROLE_ORDER = ["baseline", "lexicase", "control", "random", "other"]


@dataclass
class Condition:
    name: str
    role: str  # one of ROLE_ORDER
    label: str
    short_label: str
    color: str
    marker: str


def find_conditions(data: pd.DataFrame, control: str | None) -> list[Condition]:
    """Work out each condition's role from its config.

    The tournament with k = 2 is the baseline. The control is the one other
    tournament size, or the one given with --control.
    """
    runs = data.drop_duplicates("condition").set_index("condition")
    tournaments = runs[runs["selection"] == "tournament"]
    larger = [str(name) for name, row in tournaments.iterrows() if row["k"] != 2]
    if control is None:
        if len(larger) > 1:
            raise SystemExit(
                f"several tournament sizes found ({', '.join(larger)}); "
                "pick one with --control"
            )
        control = larger[0] if larger else None

    conditions = []
    for index, row in runs.iterrows():
        name = str(index)
        if row["selection"] == "lexicase":
            condition = Condition(
                name, "lexicase", "Lexicase", "Lexicase", "#eb6834", "s"
            )
        elif row["selection"] == "random":
            condition = Condition(
                name, "random", "Random search", "Random\nsearch", "#52514e", "D"
            )
        elif int(row["k"]) == 2:
            condition = Condition(
                name,
                "baseline",
                "Tournament (k = 2)",
                "Tournament\nk = 2",
                "#2a78d6",
                "o",
            )
        elif name == control:
            k = int(row["k"])
            condition = Condition(
                name,
                "control",
                f"Tournament (k = {k}, matched)",
                f"Tournament\nk = {k}",
                "#1baf7a",
                "^",
            )
        else:
            condition = Condition(name, "other", name, name, "#898781", "v")
        conditions.append(condition)
    conditions.sort(key=lambda c: (ROLE_ORDER.index(c.role), c.name))
    return conditions


def final_generation(data: pd.DataFrame) -> pd.DataFrame:
    last = data.groupby(["condition", "seed"])["generation"].transform("max")
    return cast(pd.DataFrame, data[data["generation"] == last])
