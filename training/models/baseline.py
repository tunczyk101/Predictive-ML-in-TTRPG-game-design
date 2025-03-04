import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

BRUTE = "brute"
MAGICAL_STRIKER = "magical_striker"
SKILL_PARAGON = "skill_paragon"
SKIRMISHER = "skirmisher"
SNIPER = "sniper"
SOLDIER = "soldier"
SPELLCASTER = "spellcaster"

ABILITIES = ["str", "dex", "int", "wis", "cha"]

BASIC_STATS = ABILITIES + [
    "con",
    "hp",
    "ac",
    "melee_max_bonus",
    "avg_melee_dmg",
    "ranged_max_bonus",
    "avg_ranged_dmg",
    "spell_dc",
    "spell_attack",
]

mental_attributes = {
    "Extreme": {SPELLCASTER, SKILL_PARAGON},
    "High": {SPELLCASTER, MAGICAL_STRIKER, SKILL_PARAGON},
    "Moderate": {MAGICAL_STRIKER},
    "Low": {BRUTE},
}
attack_bonus = {
    "High": {MAGICAL_STRIKER, BRUTE, SOLDIER, SNIPER},
    "Moderate": {BRUTE},
    "Low": {SPELLCASTER},
}
damage = {
    "Extreme": {BRUTE},
    "High": {BRUTE},
    "Moderate": {MAGICAL_STRIKER},
    "Low": {SPELLCASTER},
}

ATTRIBUTES = {
    "str": {
        "Extreme": {BRUTE},
        "High": {BRUTE, SOLDIER},
    },
    "con": {
        "High": {BRUTE},
        "Moderate": {BRUTE},
    },
    "dex": {
        "Extreme": {SKILL_PARAGON},
        "High": {SKIRMISHER, SNIPER, SKILL_PARAGON},
        "Low": {BRUTE},
    },
    "int": mental_attributes,
    "wis": mental_attributes,
    "cha": mental_attributes,
    "perception": {
        "High": {SNIPER},
        "Low": {BRUTE},
    },
    "ac": {
        "Extreme": {SOLDIER},
        "High": {SOLDIER},
        "Moderate": {BRUTE},
        "Low": {BRUTE},
    },
    "fortitude": {
        "High": {BRUTE, SOLDIER},
        "Low": {SKIRMISHER, SNIPER, SPELLCASTER, SKILL_PARAGON},
    },
    "reflex": {
        "High": {SKIRMISHER, SNIPER, SKILL_PARAGON},
        "Low": {BRUTE},
    },
    "will": {
        "High": {SPELLCASTER, SKILL_PARAGON},
        "Low": {BRUTE},
    },
    "hp": {
        "High": {BRUTE},
        "Moderate": {SNIPER},
        "Low": {SNIPER, SPELLCASTER},
    },
    "spell_attack": attack_bonus,
    "melee_max_bonus": attack_bonus,
    "ranged_max_bonus": attack_bonus,
    "spell_dc": {
        "Extreme": {SPELLCASTER},
        "High": {MAGICAL_STRIKER, SPELLCASTER},
        "Moderate": {MAGICAL_STRIKER},
    },
    "avg_melee_dmg": damage,
    "avg_ranged_dmg": damage,
}

archetypes = [
    BRUTE,
    MAGICAL_STRIKER,
    SKILL_PARAGON,
    SKIRMISHER,
    SNIPER,
    SOLDIER,
    SPELLCASTER,
]

max_scores_per_archetype = {
    "brute": 17,
    "magical_striker": 9,
    "skill_paragon": 7,
    "skirmisher": 3,
    "sniper": 8,
    "soldier": 6,
    "spellcaster": 12,
}


class BaselineModel:
    def fit(self, X, y):
        self.bestiary = X.copy()
        self.bestiary["level"] = y
        self.bestiary["archetype"] = self.bestiary.apply(
            self.classify_creatures_archetype, axis=1
        )

        # filter bestiaries by archetype
        all_archetypes = self.bestiary["archetype"].unique()
        self.filtered_bestiaries = {
            archetype: self.bestiary.loc[self.bestiary["archetype"] == archetype]
            for archetype in all_archetypes
        }

        # create knn for each archetype
        self.filtered_knns = {
            archetype: KNeighborsRegressor(n_neighbors=3, n_jobs=-1, metric="manhattan")
            if len(self.filtered_bestiaries[archetype]) != 0
            else None
            for archetype in all_archetypes
        }
        for archetype in all_archetypes:
            if self.filtered_knns[archetype] is None:
                continue
            self.filtered_knns[archetype].fit(
                self.filtered_bestiaries[archetype][BASIC_STATS],
                self.filtered_bestiaries[archetype]["level"],
            )

        # create knn for all bestiaries
        self.knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1, metric="manhattan")
        self.knn.fit(self.bestiary[BASIC_STATS], self.bestiary["level"])

    def predict(self, X):
        return self.filtered_knn(X)

    def __init__(self):
        # load stat distributions
        self.ability_dist = pd.read_csv(
            "../dataset/attribute_distributions/modifiers.csv", index_col="Level"
        )
        self.ac_dist = pd.read_csv(
            "../dataset/attribute_distributions/ac.csv", index_col="Level"
        )
        self.hp_dist = pd.read_csv(
            "../dataset/attribute_distributions/hp.csv", index_col="Level"
        )
        self.perception_dist = pd.read_csv(
            "../dataset/attribute_distributions/perception.csv", index_col="Level"
        )
        self.saving_throws_dist = pd.read_csv(
            "../dataset/attribute_distributions/saving_throws.csv", index_col="Level"
        )
        self.spell_attack_dist = pd.read_csv(
            "../dataset/attribute_distributions/spell_attack_bonus.csv", index_col="Level"
        )
        self.spell_dc_dist = pd.read_csv(
            "../dataset/attribute_distributions/spell_save_dc.csv", index_col="Level"
        )
        self.strike_attack_dist = pd.read_csv(
            "../dataset/attribute_distributions/strike_attack_bonus.csv", index_col="Level"
        )
        self.strike_damage_dist = pd.read_csv(
            "../dataset/attribute_distributions/strike_damage.csv", index_col="Level"
        )

        self.modifier_distributions = {
            "str": self.ability_dist,
            "dex": self.ability_dist,
            "con": self.ability_dist,
            "int": self.ability_dist,
            "wis": self.ability_dist,
            "cha": self.ability_dist,
            "ac": self.ac_dist,
            "hp": self.hp_dist,
            "perception": self.perception_dist,
            "fortitude": self.saving_throws_dist,
            "reflex": self.saving_throws_dist,
            "will": self.saving_throws_dist,
            "spell_attack": self.spell_attack_dist,
            "spell_dc": self.spell_dc_dist,
            "melee_max_bonus": self.strike_attack_dist,
            "ranged_max_bonus": self.strike_attack_dist,
            "avg_melee_dmg": self.strike_damage_dist,
            "avg_ranged_dmg": self.strike_damage_dist,
        }

    def find_best_abilities(self, row):
        best_abilities = set()

        max_modifier = max(row[ABILITIES])
        for ability in ABILITIES:
            if row[ability] == max_modifier:
                best_abilities.add(ability)

        return best_abilities

    def assumed_lvl_range(self, row, modifiers):
        min_lvl = -1
        max_lvl = 21
        max_modifier = max(row[ABILITIES])

        for level in range(-1, 22):
            extreme = modifiers.iloc[level]["Extreme"]
            if max_modifier <= extreme:
                min_lvl = level
                break

        for level in range(21, 1, -1):
            moderate = modifiers.iloc[level]["Moderate"]
            if max_modifier >= moderate:
                max_lvl = level
                break

        return (min_lvl, max_lvl)

    def find_spellcaster_lvl(self, row):
        max_spell_lvl = 0
        for spell_lvl in range(9, 0, -1):
            if row["spells_nr_lvl_" + str(spell_lvl)] > 0:
                max_spell_lvl = spell_lvl

        spellcaster_lvl = max_spell_lvl * 2 - 1
        return spellcaster_lvl

    def assess_stat_height(self, stat_val, stat_name, assumed_lvl):
        # get stat distribution at assumed level
        stat_distribution = self.modifier_distributions[stat_name]
        stat_distribution = stat_distribution.iloc[assumed_lvl].to_frame().T

        # calculate how far the stat is from being extreme, high, moderate or low at assumed level
        dist_diff = stat_distribution - stat_val
        dist_diff = dist_diff.abs()

        # find index with the smallest difference
        closest_stat = dist_diff.idxmin(axis=1)

        # return the name of the closest stat height
        return closest_stat.values[0]

    def classify_creatures_archetype(self, row, use_lvl=True):
        possible_archetypes = set()

        min_lvl, max_lvl = self.assumed_lvl_range(row, self.ability_dist)

        # during training we know the level and can use it
        if use_lvl and "level" in row and not np.isnan(row["level"]):
            assumed_lvl = row["level"]
        # otherwise, we use base knn to predict level
        else:
            assumed_lvl = self.knn.predict([row[BASIC_STATS]])[0]

        assumed_lvl = int(assumed_lvl)

        # preliminary check for possible archetypes
        best_abilities = self.find_best_abilities(row)
        spellcaster_lvl = self.find_spellcaster_lvl(row)

        if "str" in best_abilities:
            possible_archetypes.add(BRUTE)
            possible_archetypes.add(SOLDIER)
        if "dex" in best_abilities:
            possible_archetypes.add(SKILL_PARAGON)
            if row["ranged_max_bonus"] > 0:
                possible_archetypes.add(SNIPER)
            if row["land_speed"] > 25 or row["fly"] > 40 or row["swim"] > 30:
                possible_archetypes.add(SKIRMISHER)
        if len(best_abilities & {"int", "wis", "cha"}) > 0:
            possible_archetypes.add(SKILL_PARAGON)
            if min_lvl <= spellcaster_lvl <= max_lvl:
                possible_archetypes.add(SPELLCASTER)
            if min_lvl <= spellcaster_lvl + 1 <= max_lvl:
                possible_archetypes.add(MAGICAL_STRIKER)

        # out of possible archetypes, find the one with the highest probability
        archetype_probabilities = {archetype: 0 for archetype in possible_archetypes}

        for attribute in ATTRIBUTES:
            stat_height = self.assess_stat_height(
                row[attribute], attribute, assumed_lvl=assumed_lvl
            )
            for archetype in ATTRIBUTES[attribute].get(stat_height, {}):
                if archetype in possible_archetypes:
                    archetype_probabilities[archetype] += 1

        for archetype in possible_archetypes:
            archetype_probabilities[archetype] /= max_scores_per_archetype[archetype]

        best_archetype = max(archetype_probabilities, key=archetype_probabilities.get)

        return best_archetype

    def filtered_knn(self, X_test):
        # predict archetypes for each row
        predicted_archetypes = [
            self.classify_creatures_archetype(row, use_lvl=False)
            for index, row in X_test.iterrows()
        ]

        # predict levels for each row based on archetype
        results = []
        for i, (_, row) in enumerate(X_test.iterrows()):
            archetype = predicted_archetypes[i]
            knn_for_archetype = self.filtered_knns[archetype]
            if knn_for_archetype is None:
                prediction = self.knn.predict([row[BASIC_STATS]])[0]

            prediction = knn_for_archetype.predict([row[BASIC_STATS]])[0]

            results.append(prediction)

        return np.array(results)
