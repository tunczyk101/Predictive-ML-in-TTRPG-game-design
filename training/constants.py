RANDOM_STATE = 42

DATASET_FILES = [
    "abomination-vaults-bestiary.json",
    "age-of-ashes-bestiary.json",
    "agents-of-edgewatch-bestiary.json",
    # "april-fools-bestiary.json",
    "blog-bestiary.json",
    "blood-lords-bestiary.json",
    "book-of-the-dead-bestiary.json",
    "crown-of-the-kobold-king-bestiary.json",
    "extinction-curse-bestiary.json",
    "fall-of-plaguestone.json",
    "fists-of-the-ruby-phoenix-bestiary.json",
    "gatewalkers-bestiary.json",
    "impossible-lands-bestiary.json",
    "kingmaker-bestiary.json",
    "malevolence-bestiary.json",
    "menace-under-otari-bestiary.json",
    "monsters-of-myth-bestiary.json",
    "mwangi-expanse-bestiary.json",
    "night-of-the-gray-death-bestiary.json",
    "npc-gallery.json",
    "one-shot-bestiary.json",
    "outlaws-of-alkenstar-bestiary.json",
    "pathfinder-bestiary-2.json",
    "pathfinder-bestiary-3.json",
    "pathfinder-bestiary.json",
    "pathfinder-dark-archive.json",
    "pfs-introductions-bestiary.json",
    "pfs-season-1-bestiary.json",
    "pfs-season-2-bestiary.json",
    "pfs-season-3-bestiary.json",
    "pfs-season-4-bestiary.json",
    "quest-for-the-frozen-flame-bestiary.json",
    "shadows-at-sundown-bestiary.json",
    "strength-of-thousands-bestiary.json",
    "the-slithering-bestiary.json",
    "travel-guide-bestiary.json",
    "troubles-in-otari-bestiary.json",
]

FEATURES = [
    "cha",
    "con",
    "dex",
    "int",
    "str",
    "wis",
    "ac",
    "hp",
    "perception",
    "fortitude",
    "reflex",
    "will",
    "focus",
    "land_speed",
    "num_immunities",
    "fly",
    "swim",
    "climb",
    "fire_resistance",
    "cold_resistance",
    "electricity_resistance",
    "acid_resistance",
    "piercing_resistance",
    "slashing_resistance",
    "physical_resistance",
    "bludgeoning_resistance",
    "mental_resistance",
    "poison_resistance",
    "all-damage_resistance",
    "cold-iron_weakness",
    "good_weakness",
    "fire_weakness",
    "cold_weakness",
    "area-damage_weakness",
    "splash-damage_weakness",
    "evil_weakness",
    "slashing_weakness",
    "melee",
    "ranged",
    "spells",
]

FEATURES_NO_ITEMS = [
    "cha",
    "con",
    "dex",
    "int",
    "str",
    "wis",
    "ac",
    "hp",
    "perception",
    "fortitude",
    "reflex",
    "will",
    "focus",
    "land_speed",
    "num_immunities",
    "fly",
    "swim",
    "climb",
    "fire_resistance",
    "cold_resistance",
    "electricity_resistance",
    "acid_resistance",
    "piercing_resistance",
    "slashing_resistance",
    "physical_resistance",
    "bludgeoning_resistance",
    "mental_resistance",
    "poison_resistance",
    "all-damage_resistance",
    "cold-iron_weakness",
    "good_weakness",
    "fire_weakness",
    "cold_weakness",
    "area-damage_weakness",
    "splash-damage_weakness",
    "evil_weakness",
    "slashing_weakness",
]

ORDERED_CHARACTERISTICS_BASIC = ["str", "dex", "con", "int", "wis", "cha", "ac", "hp"]
ORDERED_CHARACTERISTICS_EXPANDED = [
    "str",
    "dex",
    "con",
    "int",
    "wis",
    "cha",
    "ac",
    "hp",
    "perception",
    "fortitude",
    "reflex",
    "will",
    "focus",
]
ORDERED_CHARACTERISTICS_FULL = [
    "str",
    "dex",
    "con",
    "int",
    "wis",
    "cha",
    "ac",
    "hp",
    "perception",
    "fortitude",
    "reflex",
    "will",
    "focus",
    "num_immunities",
    "land_speed",
    "fly",
    "climb",
    "swim",
    "spells_nr_lvl_1",
    "spells_nr_lvl_2",
    "spells_nr_lvl_3",
    "spells_nr_lvl_4",
    "spells_nr_lvl_5",
    "spells_nr_lvl_6",
    "spells_nr_lvl_7",
    "spells_nr_lvl_8",
    "spells_nr_lvl_9",
    "melee_max_bonus",
    "avg_melee_dmg",
    "ranged_max_bonus",
    "avg_ranged_dmg",
    "acid_resistance",
    "all-damage_resistance",
    "bludgeoning_resistance",
    "cold_resistance",
    "electricity_resistance",
    "fire_resistance",
    "mental_resistance",
    "physical_resistance",
    "piercing_resistance",
    "poison_resistance",
    "slashing_resistance",
    "area-damage_weakness",
    "cold_weakness",
    "cold-iron_weakness",
    "evil_weakness",
    "fire_weakness",
    "good_weakness",
    "slashing_weakness",
    "splash-damage_weakness",
]

ORDERED_CHARACTERISTICS_FULL_NO_ITEMS = [
    "str",
    "dex",
    "con",
    "int",
    "wis",
    "cha",
    "ac",
    "hp",
    "perception",
    "fortitude",
    "reflex",
    "will",
    "focus",
    "num_immunities",
    "land_speed",
    "fly",
    "climb",
    "swim",
    "acid_resistance",
    "all-damage_resistance",
    "bludgeoning_resistance",
    "cold_resistance",
    "electricity_resistance",
    "fire_resistance",
    "mental_resistance",
    "physical_resistance",
    "piercing_resistance",
    "poison_resistance",
    "slashing_resistance",
    "area-damage_weakness",
    "cold_weakness",
    "cold-iron_weakness",
    "evil_weakness",
    "fire_weakness",
    "good_weakness",
    "slashing_weakness",
    "splash-damage_weakness",
]
THRESHOLD = 0.33
