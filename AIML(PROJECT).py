import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import math
import uuid
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Set random seeds
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Simulation parameters
GENERATIONS = 120
K_TOTAL = 5000
K_OPPONENTS = 8
COMBAT_PHASE_RATIO = 0.4
REST_PHASE_RATIO = 0.3
MATING_PHASE_RATIO = 0.3
PAIR_COMBAT_CHANCE = 0.4
XP_WIN = 10.0
XP_PER_LEVEL = 100.0
LEVEL_GROWTH = 0.02
SIGMOID_BETA = 0.08
P_TOTAL = 50000.0
RESOURCE_MAX = 40.0
R0 = 10.0
FOOD_COST = 0.2
GAMMA_RESOURCE_XP = 0.15
PREDATION_BIOMASS_FACTOR = 0.8
MUT_PROB = 0.01
MUT_SIGMA = 1.0
MAX_LEVEL_NORMAL = 100
MAX_LEVEL_LEGENDARY = 150
LEVEL_DIFF_XP_BONUS = 10
GENETIC_DRIFT_RATE = 0.02

# Real Gen 1 Pokémon data (all 151)
POKEMON_DATA = [
    # Format: [Name, HP, Atk, Def, SpA, SpD, Speed, Type1, Type2, EggGroup1, EggGroup2]
    ["Bulbasaur", 45, 49, 49, 65, 65, 45, "Grass", "Poison", "Monster", "Grass"],
    ["Ivysaur", 60, 62, 63, 80, 80, 60, "Grass", "Poison", "Monster", "Grass"],
    ["Venusaur", 80, 82, 83, 100, 100, 80, "Grass", "Poison", "Monster", "Grass"],
    ["Charmander", 39, 52, 43, 60, 50, 65, "Fire", None, "Monster", "Dragon"],
    ["Charmeleon", 58, 64, 58, 80, 65, 80, "Fire", None, "Monster", "Dragon"],
    ["Charizard", 78, 84, 78, 109, 85, 100, "Fire", "Flying", "Monster", "Dragon"],
    ["Squirtle", 44, 48, 65, 50, 64, 43, "Water", None, "Monster", "Water1"],
    ["Wartortle", 59, 63, 80, 65, 80, 58, "Water", None, "Monster", "Water1"],
    ["Blastoise", 79, 83, 100, 85, 105, 78, "Water", None, "Monster", "Water1"],
    ["Caterpie", 45, 30, 35, 20, 20, 45, "Bug", None, "Bug", None],
    ["Metapod", 50, 20, 55, 25, 25, 30, "Bug", None, "Bug", None],
    ["Butterfree", 60, 45, 50, 90, 80, 70, "Bug", "Flying", "Bug", None],
    ["Weedle", 40, 35, 30, 20, 20, 50, "Bug", "Poison", "Bug", None],
    ["Kakuna", 45, 25, 50, 25, 25, 35, "Bug", "Poison", "Bug", None],
    ["Beedrill", 65, 90, 40, 45, 80, 75, "Bug", "Poison", "Bug", None],
    ["Pidgey", 40, 45, 40, 35, 35, 56, "Normal", "Flying", "Flying", None],
    ["Pidgeotto", 63, 60, 55, 50, 50, 71, "Normal", "Flying", "Flying", None],
    ["Pidgeot", 83, 80, 75, 70, 70, 101, "Normal", "Flying", "Flying", None],
    ["Rattata", 30, 56, 35, 25, 35, 72, "Normal", None, "Field", None],
    ["Raticate", 55, 81, 60, 50, 70, 97, "Normal", None, "Field", None],
    ["Spearow", 40, 60, 30, 31, 31, 70, "Normal", "Flying", "Flying", None],
    ["Fearow", 65, 90, 65, 61, 61, 100, "Normal", "Flying", "Flying", None],
    ["Ekans", 35, 60, 44, 40, 54, 55, "Poison", None, "Field", "Dragon"],
    ["Arbok", 60, 95, 69, 65, 79, 80, "Poison", None, "Field", "Dragon"],
    ["Pikachu", 35, 55, 40, 50, 50, 90, "Electric", None, "Field", "Fairy"],
    ["Raichu", 60, 90, 55, 90, 80, 110, "Electric", None, "Field", "Fairy"],
    ["Sandshrew", 50, 75, 85, 20, 30, 40, "Ground", None, "Field", None],
    ["Sandslash", 75, 100, 110, 45, 55, 65, "Ground", None, "Field", None],
    ["Nidoran♀", 55, 47, 52, 40, 40, 41, "Poison", None, "Monster", "Field"],
    ["Nidorina", 70, 62, 67, 55, 55, 56, "Poison", None, "Undiscovered", None],
    ["Nidoqueen", 90, 92, 87, 75, 85, 76, "Poison", "Ground", "Undiscovered", None],
    ["Nidoran♂", 46, 57, 40, 40, 40, 50, "Poison", None, "Monster", "Field"],
    ["Nidorino", 61, 72, 57, 55, 55, 65, "Poison", None, "Monster", "Field"],
    ["Nidoking", 81, 102, 77, 85, 75, 85, "Poison", "Ground", "Monster", "Field"],
    ["Clefairy", 70, 45, 48, 60, 65, 35, "Fairy", None, "Fairy", None],
    ["Clefable", 95, 70, 73, 95, 90, 60, "Fairy", None, "Fairy", None],
    ["Vulpix", 38, 41, 40, 50, 65, 65, "Fire", None, "Field", None],
    ["Ninetales", 73, 76, 75, 81, 100, 100, "Fire", None, "Field", None],
    ["Jigglypuff", 115, 45, 20, 45, 25, 20, "Fairy", None, "Fairy", None],
    ["Wigglytuff", 140, 70, 45, 85, 50, 45, "Fairy", None, "Fairy", None],
    ["Zubat", 40, 45, 35, 30, 40, 55, "Poison", "Flying", "Flying", None],
    ["Golbat", 75, 80, 70, 65, 75, 90, "Poison", "Flying", "Flying", None],
    ["Oddish", 45, 50, 55, 75, 65, 30, "Grass", "Poison", "Grass", None],
    ["Gloom", 60, 65, 70, 85, 75, 40, "Grass", "Poison", "Grass", None],
    ["Vileplume", 75, 80, 85, 110, 90, 50, "Grass", "Poison", "Grass", None],
    ["Paras", 35, 70, 55, 45, 55, 25, "Bug", "Grass", "Bug", "Grass"],
    ["Parasect", 60, 95, 80, 60, 80, 30, "Bug", "Grass", "Bug", "Grass"],
    ["Venonat", 60, 55, 50, 40, 55, 45, "Bug", "Poison", "Bug", None],
    ["Venomoth", 70, 65, 60, 90, 75, 90, "Bug", "Poison", "Bug", None],
    ["Diglett", 10, 55, 25, 35, 45, 95, "Ground", None, "Field", None],
    ["Dugtrio", 35, 100, 50, 50, 70, 120, "Ground", None, "Field", None],
    ["Meowth", 40, 45, 35, 40, 40, 90, "Normal", None, "Field", None],
    ["Persian", 65, 70, 60, 65, 65, 115, "Normal", None, "Field", None],
    ["Psyduck", 50, 52, 48, 65, 50, 55, "Water", None, "Water1", "Field"],
    ["Golduck", 80, 82, 78, 95, 80, 85, "Water", None, "Water1", "Field"],
    ["Mankey", 40, 80, 35, 35, 45, 70, "Fighting", None, "Field", None],
    ["Primeape", 65, 105, 60, 60, 70, 95, "Fighting", None, "Field", None],
    ["Growlithe", 55, 70, 45, 70, 50, 60, "Fire", None, "Field", None],
    ["Arcanine", 90, 110, 80, 100, 80, 95, "Fire", None, "Field", None],
    ["Poliwag", 40, 50, 40, 40, 40, 90, "Water", None, "Water1", None],
    ["Poliwhirl", 65, 65, 65, 50, 50, 90, "Water", None, "Water1", None],
    ["Poliwrath", 90, 95, 95, 70, 90, 70, "Water", "Fighting", "Water1", None],
    ["Abra", 25, 20, 15, 105, 55, 90, "Psychic", None, "Human-Like", None],
    ["Kadabra", 40, 35, 30, 120, 70, 105, "Psychic", None, "Human-Like", None],
    ["Alakazam", 55, 50, 45, 135, 95, 120, "Psychic", None, "Human-Like", None],
    ["Machop", 70, 80, 50, 35, 35, 35, "Fighting", None, "Human-Like", None],
    ["Machoke", 80, 100, 70, 50, 60, 45, "Fighting", None, "Human-Like", None],
    ["Machamp", 90, 130, 80, 65, 85, 55, "Fighting", None, "Human-Like", None],
    ["Bellsprout", 50, 75, 35, 70, 30, 40, "Grass", "Poison", "Grass", None],
    ["Weepinbell", 65, 90, 50, 85, 45, 55, "Grass", "Poison", "Grass", None],
    ["Victreebel", 80, 105, 65, 100, 70, 70, "Grass", "Poison", "Grass", None],
    ["Tentacool", 40, 40, 35, 50, 100, 70, "Water", "Poison", "Water3", None],
    ["Tentacruel", 80, 70, 65, 80, 120, 100, "Water", "Poison", "Water3", None],
    ["Geodude", 40, 80, 100, 30, 30, 20, "Rock", "Ground", "Mineral", None],
    ["Graveler", 55, 95, 115, 45, 45, 35, "Rock", "Ground", "Mineral", None],
    ["Golem", 80, 120, 130, 55, 65, 45, "Rock", "Ground", "Mineral", None],
    ["Ponyta", 50, 85, 55, 65, 65, 90, "Fire", None, "Field", None],
    ["Rapidash", 65, 100, 70, 80, 80, 105, "Fire", None, "Field", None],
    ["Slowpoke", 90, 65, 65, 40, 40, 15, "Water", "Psychic", "Monster", "Water1"],
    ["Slowbro", 95, 75, 110, 100, 80, 30, "Water", "Psychic", "Monster", "Water1"],
    ["Magnemite", 25, 35, 70, 95, 55, 45, "Electric", "Steel", "Mineral", None],
    ["Magneton", 50, 60, 95, 120, 70, 70, "Electric", "Steel", "Mineral", None],
    ["Farfetchd", 52, 90, 55, 58, 62, 60, "Normal", "Flying", "Flying", "Field"],
    ["Doduo", 35, 85, 45, 35, 35, 75, "Normal", "Flying", "Flying", None],
    ["Dodrio", 60, 110, 70, 60, 60, 110, "Normal", "Flying", "Flying", None],
    ["Seel", 65, 45, 55, 45, 70, 45, "Water", None, "Water1", "Field"],
    ["Dewgong", 90, 70, 80, 70, 95, 70, "Water", "Ice", "Water1", "Field"],
    ["Grimer", 80, 80, 50, 40, 50, 25, "Poison", None, "Amorphous", None],
    ["Muk", 105, 105, 75, 65, 100, 50, "Poison", None, "Amorphous", None],
    ["Shellder", 30, 65, 100, 45, 25, 40, "Water", None, "Water3", None],
    ["Cloyster", 50, 95, 180, 85, 45, 70, "Water", "Ice", "Water3", None],
    ["Gastly", 30, 35, 30, 100, 35, 80, "Ghost", "Poison", "Amorphous", None],
    ["Haunter", 45, 50, 45, 115, 55, 95, "Ghost", "Poison", "Amorphous", None],
    ["Gengar", 60, 65, 60, 130, 75, 110, "Ghost", "Poison", "Amorphous", None],
    ["Onix", 35, 45, 160, 30, 45, 70, "Rock", "Ground", "Mineral", None],
    ["Drowzee", 60, 48, 45, 43, 90, 42, "Psychic", None, "Human-Like", None],
    ["Hypno", 85, 73, 70, 73, 115, 67, "Psychic", None, "Human-Like", None],
    ["Krabby", 30, 105, 90, 25, 25, 50, "Water", None, "Water3", None],
    ["Kingler", 55, 130, 115, 50, 50, 75, "Water", None, "Water3", None],
    ["Voltorb", 40, 30, 50, 55, 55, 100, "Electric", None, "Mineral", None],
    ["Electrode", 60, 50, 70, 80, 80, 150, "Electric", None, "Mineral", None],
    ["Exeggcute", 60, 40, 80, 60, 45, 40, "Grass", "Psychic", "Grass", None],
    ["Exeggutor", 95, 95, 85, 125, 75, 55, "Grass", "Psychic", "Grass", None],
    ["Cubone", 50, 50, 95, 40, 50, 35, "Ground", None, "Monster", None],
    ["Marowak", 60, 80, 110, 50, 80, 45, "Ground", None, "Monster", None],
    ["Hitmonlee", 50, 120, 53, 35, 110, 87, "Fighting", None, "Human-Like", None],
    ["Hitmonchan", 50, 105, 79, 35, 110, 76, "Fighting", None, "Human-Like", None],
    ["Lickitung", 90, 55, 75, 60, 75, 30, "Normal", None, "Monster", None],
    ["Koffing", 40, 65, 95, 60, 45, 35, "Poison", None, "Amorphous", None],
    ["Weezing", 65, 90, 120, 85, 70, 60, "Poison", None, "Amorphous", None],
    ["Rhyhorn", 80, 85, 95, 30, 30, 25, "Ground", "Rock", "Monster", "Field"],
    ["Rhydon", 105, 130, 120, 45, 45, 40, "Ground", "Rock", "Monster", "Field"],
    ["Chansey", 250, 5, 5, 35, 105, 50, "Normal", None, "Fairy", None],
    ["Tangela", 65, 55, 115, 100, 40, 60, "Grass", None, "Grass", None],
    ["Kangaskhan", 105, 95, 80, 40, 80, 90, "Normal", None, "Monster", None],
    ["Horsea", 30, 40, 70, 70, 25, 60, "Water", None, "Water1", "Dragon"],
    ["Seadra", 55, 65, 95, 95, 45, 85, "Water", None, "Water1", "Dragon"],
    ["Goldeen", 45, 67, 60, 35, 50, 63, "Water", None, "Water2", None],
    ["Seaking", 80, 92, 65, 65, 80, 68, "Water", None, "Water2", None],
    ["Staryu", 30, 45, 55, 70, 55, 85, "Water", None, "Water3", None],
    ["Starmie", 60, 75, 85, 100, 85, 115, "Water", "Psychic", "Water3", None],
    ["MrMime", 40, 45, 65, 100, 120, 90, "Psychic", "Fairy", "Human-Like", None],
    ["Scyther", 70, 110, 80, 55, 80, 105, "Bug", "Flying", "Bug", None],
    ["Jynx", 65, 50, 35, 115, 95, 95, "Ice", "Psychic", "Human-Like", None],
    ["Electabuzz", 65, 83, 57, 95, 85, 105, "Electric", None, "Human-Like", None],
    ["Magmar", 65, 95, 57, 100, 85, 93, "Fire", None, "Human-Like", None],
    ["Pinsir", 65, 125, 100, 55, 70, 85, "Bug", None, "Bug", None],
    ["Tauros", 75, 100, 95, 40, 70, 110, "Normal", None, "Field", None],
    ["Magikarp", 20, 10, 55, 15, 20, 80, "Water", None, "Water2", "Dragon"],
    ["Gyarados", 95, 125, 79, 60, 100, 81, "Water", "Flying", "Water2", "Dragon"],
    ["Lapras", 130, 85, 80, 85, 95, 60, "Water", "Ice", "Monster", "Water1"],
    ["Ditto", 48, 48, 48, 48, 48, 48, "Normal", None, "Ditto", None],
    ["Eevee", 55, 55, 50, 45, 65, 55, "Normal", None, "Field", None],
    ["Vaporeon", 130, 65, 60, 110, 95, 65, "Water", None, "Field", None],
    ["Jolteon", 65, 65, 60, 110, 95, 130, "Electric", None, "Field", None],
    ["Flareon", 65, 130, 60, 95, 110, 65, "Fire", None, "Field", None],
    ["Porygon", 65, 60, 70, 85, 75, 40, "Normal", None, "Mineral", None],
    ["Omanyte", 35, 40, 100, 90, 55, 35, "Rock", "Water", "Water1", "Water3"],
    ["Omastar", 70, 60, 125, 115, 70, 55, "Rock", "Water", "Water1", "Water3"],
    ["Kabuto", 30, 80, 90, 55, 45, 55, "Rock", "Water", "Water1", "Water3"],
    ["Kabutops", 60, 115, 105, 65, 70, 80, "Rock", "Water", "Water1", "Water3"],
    ["Aerodactyl", 80, 105, 65, 60, 75, 130, "Rock", "Flying", "Flying", None],
    ["Snorlax", 160, 110, 65, 65, 110, 30, "Normal", None, "Monster", None],
    ["Articuno", 90, 85, 100, 95, 125, 85, "Ice", "Flying", "Undiscovered", None],
    ["Zapdos", 90, 90, 85, 125, 90, 100, "Electric", "Flying", "Undiscovered", None],
    ["Moltres", 90, 100, 90, 125, 85, 90, "Fire", "Flying", "Undiscovered", None],
    ["Dratini", 41, 64, 45, 50, 50, 50, "Dragon", None, "Water1", "Dragon"],
    ["Dragonair", 61, 84, 65, 70, 70, 70, "Dragon", None, "Water1", "Dragon"],
    ["Dragonite", 91, 134, 95, 100, 100, 80, "Dragon", "Flying", "Water1", "Dragon"],
    ["Mewtwo", 106, 110, 90, 154, 90, 130, "Psychic", None, "Undiscovered", None],
    ["Mew", 100, 100, 100, 100, 100, 100, "Psychic", None, "Undiscovered", None]
]
prototypes = pd.DataFrame(POKEMON_DATA, columns=[
    "name", "HP", "Atk", "Def", "SpA", "SpD", "Speed", 
    "type1", "type2", "egg_group1", "egg_group2"
])
prototypes['species_id'] = range(1, len(prototypes) + 1)

types = ["Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison",
         "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Steel", "Fairy"]
type_to_idx = {t: i for i, t in enumerate(types)}

NUM_TYPES = len(types)
rng = np.random.RandomState(SEED)
type_adv = rng.normal(loc=0.0, scale=0.12, size=(NUM_TYPES, NUM_TYPES))
type_adv = np.clip(type_adv, -0.3, 0.3)
np.fill_diagonal(type_adv, 0.0)
def type_bonus(typeA_idx, typeB_idx):
    return type_adv[typeA_idx, typeB_idx]
def assign_diet(row):
    type1 = row['type1']
    name = row['name']    
    if name in ['Mewtwo', 'Mew', 'Articuno', 'Zapdos', 'Moltres', 'Dragonite']:
        return 'carnivore'
    if type1 in ['Poison'] or name in ['Paras', 'Parasect', 'Venonat', 'Venomoth']:
        return 'parasite'    
    if type1 in ['Dragon', 'Ghost', 'Dark'] or row['type2'] in ['Dragon', 'Ghost']:
        return 'carnivore'
    if name in ['Gyarados', 'Arcanine', 'Charizard', 'Gengar', 'Alakazam']:
        return 'carnivore'
    if type1 == 'Fighting' or row['Atk'] > 100:
        return 'carnivore'
    if type1 in ['Grass', 'Bug'] or (type1 == 'Normal' and row['Atk'] < 80):
        return 'herbivore'
        r = random.random()
    if r < 0.15:
        return 'parasite'
    elif r < 0.45:
        return 'carnivore'
    else:
        return 'herbivore'

prototypes['diet'] = prototypes.apply(assign_diet, axis=1)
def create_individual(proto_row, individual_uid=None):
    if individual_uid is None:
        individual_uid = str(uuid.uuid4())
    
    base_stats = np.array([
        proto_row['HP'], proto_row['Atk'], proto_row['Def'],
        proto_row['SpA'], proto_row['SpD'], proto_row['Speed']
    ], dtype=float)
    
    is_legendary = proto_row['name'] in ['Mewtwo', 'Mew', 'Articuno', 'Zapdos', 'Moltres']
    
    return {
        "uid": individual_uid,
        "species_id": int(proto_row['species_id']),
        "name": proto_row['name'],
        "base_stats": base_stats.copy(),
        "current_stats": base_stats.copy(),
        "hp": base_stats[0],
        "level": 1,
        "xp": 0.0,
        "resource": R0,
        "diet": proto_row['diet'],
        "type_idx": type_to_idx.get(proto_row['type1'], 0),
        "egg_group1": proto_row['egg_group1'],
        "egg_group2": proto_row['egg_group2'],
        "age": 0,
        "alive": True,
        "is_legendary": is_legendary,
        "max_level": MAX_LEVEL_LEGENDARY if is_legendary else MAX_LEVEL_NORMAL,
        "rest_energy": 100.0,
        "mating_readiness": 50.0,
        "generation_born": 0
    }

# Initialize population (3-5 of each species)
population = []
for _, proto in prototypes.iterrows():
    # Legendaries only get 1, regular Pokémon get 3-5
    count = 1 if proto['name'] in ['Mewtwo', 'Mew', 'Articuno', 'Zapdos', 'Moltres'] else random.randint(3, 5)
    for _ in range(count):
        population.append(create_individual(proto))

print(f"Initial population size: {len(population)}")
print(f"Species count: {len(prototypes)}")
print(f"Diet distribution: {prototypes['diet'].value_counts().to_dict()}")

# Helper functions
def effective_stats(ind):
    return ind['base_stats'] * (1.0 + LEVEL_GROWTH * (ind['level'] - 1))

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-np.clip(x, -500, 500)))
def battle_prob(indA, indB, allyA=None, allyB=None):
    """Battle probability with type advantages and ally support"""
    A_stats = effective_stats(indA)
    B_stats = effective_stats(indB)
    w_atk, w_spa, w_spd = 0.5, 0.4, 0.1
    s = (w_atk * (A_stats[1] - B_stats[2]) + 
         w_spa * (A_stats[3] - B_stats[4]) + 
         w_spd * (A_stats[5] - B_stats[5]))
    type_mult = type_bonus(indA['type_idx'], indB['type_idx'])
    s += type_mult * 40.0
    s += (indA['rest_energy'] / 100.0) * 10.0    
    if allyA:
        ally_stats = effective_stats(allyA)
        s += (ally_stats[1] + ally_stats[3]) * 0.15
        s += type_bonus(allyA['type_idx'], indB['type_idx']) * 15.0
    
    if allyB:
        opp_stats = effective_stats(allyB)
        s -= (opp_stats[1] + opp_stats[3]) * 0.15
        s -= type_bonus(allyB['type_idx'], indA['type_idx']) * 15.0
    
    return np.clip(sigmoid(SIGMOID_BETA * s), 0.05, 0.95)
# Egg group compatibility
def can_breed(ind1, ind2):
    """Check if two individuals can breed based on egg groups"""
    if ind1['species_id'] == ind2['species_id']:
        return True
    # Undiscovered (legendaries) cannot breed normally
    if ind1['egg_group1'] == 'Undiscovered' or ind2['egg_group1'] == 'Undiscovered':
        return False
    # Ditto can breed with anyone (except Undiscovered)
    if ind1['name'] == 'Ditto' or ind2['name'] == 'Ditto':
        return True
    # Mew can breed with anyone (except Undiscovered)
    if ind1['name'] == 'Mew' or ind2['name'] == 'Mew':
        return True
    # Check egg group compatibility
    groups1 = {ind1['egg_group1'], ind1['egg_group2']} - {None}
    groups2 = {ind2['egg_group1'], ind2['egg_group2']} - {None}
    return len(groups1 & groups2) > 0
# Ecology functions
def forage_plants(pop):
    """Herbivores gather plant resources"""
    herbivores = [ind for ind in pop if ind['alive'] and ind['diet'] == 'herbivore']
    if not herbivores:
        return
    scores = np.array([effective_stats(ind)[5] + 0.1 * effective_stats(ind)[3] 
                      for ind in herbivores])
    scores = np.maximum(scores, 0.1)
    shares = (scores / scores.sum()) * P_TOTAL
    for ind, share in zip(herbivores, shares):
        ind['resource'] = min(RESOURCE_MAX, ind['resource'] + share / 100.0)
def attempt_predation(pred, prey, ally_pred=None, ally_prey=None):
    """Carnivore hunts prey"""
    p_kill = battle_prob(pred, prey, ally_pred, ally_prey)
    if np.random.rand() < p_kill:
        # Successful hunt
        biomass_gain = PREDATION_BIOMASS_FACTOR * prey['base_stats'][0]
        pred['resource'] += biomass_gain / 100.0
        if ally_pred and ally_pred['alive']:
            ally_pred['resource'] += biomass_gain / 200.0
        xp_gain = XP_WIN * 2.0
        if prey['level'] - pred['level'] >= LEVEL_DIFF_XP_BONUS:
            xp_gain *= 2.0
        pred['xp'] += xp_gain
        if ally_pred and ally_pred['alive']:
            ally_pred['xp'] += xp_gain * 0.6
        prey['alive'] = False
        return True
    else:
        # Prey fights back
        damage = max(0.5, 0.01 * prey['base_stats'][1])
        pred['hp'] -= damage
        if pred['hp'] <= 0:
            pred['alive'] = False
        if ally_pred and ally_pred['alive'] and np.random.rand() < 0.3:
            ally_pred['hp'] -= damage * 0.5
            if ally_pred['hp'] <= 0:
                ally_pred['alive'] = False
        
        return False
def parasite_action(parasite, host):
    """Parasite drains resources from host"""
    p_attach = battle_prob(parasite, host) * 0.6
    if np.random.rand() < p_attach and host['alive']:
        drain = min(0.5 + 0.02 * parasite['level'], host['resource'])
        host['resource'] -= drain
        parasite['resource'] += drain
        parasite['xp'] += drain * 0.5
        host['resource'] = max(0, host['resource'])

def rest_phase(pop):
    """Individuals rest and recover"""
    for ind in pop:
        if not ind['alive']:
            continue
        
        base_recovery = 10.0
        resource_factor = min(1.0, ind['resource'] / RESOURCE_MAX)
        recovery = base_recovery * (0.5 + 0.5 * resource_factor)
        
        ind['rest_energy'] = min(100.0, ind['rest_energy'] + recovery)
        
        if ind['rest_energy'] > 80.0:
            ind['hp'] = min(ind['current_stats'][0], ind['hp'] + 0.5)

def mating_phase(pop):
    """Individuals build mating readiness"""
    for ind in pop:
        if not ind['alive']:
            continue
        
        base_gain = 8.0
        resource_factor = min(1.0, ind['resource'] / RESOURCE_MAX)
        energy_factor = ind['rest_energy'] / 100.0
        gain = base_gain * (0.3 + 0.4 * resource_factor + 0.3 * energy_factor)
        
        ind['mating_readiness'] = min(100.0, ind['mating_readiness'] + gain)

# Main generation loop
def run_generation(pop, gen_num):
    # Phase 1: Foraging
    forage_plants(pop)
    
    # Phase 2: Combat
    alive_combat = [ind for ind in pop if ind['alive']]
    combat_pairs_formed = set()
    
    for ind in alive_combat:
        if not ind['alive'] or ind['uid'] in combat_pairs_formed:
            continue
        
        # Find ally
        ally = None
        if np.random.rand() < PAIR_COMBAT_CHANCE:
            potential_allies = [
                a for a in alive_combat
                if a['alive'] and a['uid'] != ind['uid']
                and a['uid'] not in combat_pairs_formed
                and (a['species_id'] == ind['species_id'] or a['diet'] == ind['diet'])
            ]
            if potential_allies:
                ally = random.choice(potential_allies)
                combat_pairs_formed.add(ind['uid'])
                combat_pairs_formed.add(ally['uid'])
        
        # Combat encounters
        encounters = min(int(K_OPPONENTS * COMBAT_PHASE_RATIO), len(alive_combat))
        opponents = random.sample(alive_combat, k=encounters)
        
        ind['rest_energy'] = max(0.0, ind['rest_energy'] - 5.0 * encounters)
        if ally:
            ally['rest_energy'] = max(0.0, ally['rest_energy'] - 5.0 * encounters)
        
        expected_xp = 0.0
        for opp in opponents:
            if not opp['alive'] or opp['uid'] == ind['uid']:
                continue
            if ally and opp['uid'] == ally['uid']:
                continue
            
            # Opponent ally
            opp_ally = None
            if np.random.rand() < PAIR_COMBAT_CHANCE * 0.7:
                potential_opp_allies = [
                    a for a in alive_combat
                    if a['alive'] and a['uid'] not in [opp['uid'], ind['uid']]
                    and (not ally or a['uid'] != ally['uid'])
                    and (a['species_id'] == opp['species_id'] or a['diet'] == opp['diet'])
                ]
                if potential_opp_allies:
                    opp_ally = random.choice(potential_opp_allies)
            
            # Diet-based interactions
            if ind['diet'] == 'carnivore' and opp['diet'] in ['herbivore', 'parasite']:
                attempt_predation(ind, opp, ally, opp_ally)
            elif ind['diet'] == 'parasite' and opp['diet'] != 'parasite':
                parasite_action(ind, opp)
            
            # Calculate XP from battle
            p_win = battle_prob(ind, opp, ally, opp_ally)
            xp_from_battle = XP_WIN * p_win
            if opp['level'] - ind['level'] >= LEVEL_DIFF_XP_BONUS:
                xp_from_battle *= 2.0
            if ally:
                xp_from_battle *= 1.15
            expected_xp += xp_from_battle
            if ally and ally['alive']:
                ally['xp'] += xp_from_battle * 0.5
        # Apply XP with resource bonus
        rfrac = min(1.0, ind['resource'] / RESOURCE_MAX)
        ind['xp'] += expected_xp * (1.0 + GAMMA_RESOURCE_XP * rfrac)
    # Phase 3: Rest
    rest_phase(pop)
    # Phase 4: Mating/Socializing
    mating_phase(pop)
    # Update stats, level up, maintenance costs
    for ind in pop:
        if not ind['alive']:
            continue
        # Level up
        new_level = int(ind['xp'] / XP_PER_LEVEL) + 1
        ind['level'] = max(1, min(new_level, ind['max_level']))
        ind['current_stats'] = effective_stats(ind)
        # HP recovery
        ind['hp'] = min(ind['current_stats'][0], ind['hp'] + min(0.1 * ind['resource'], 0.5))
        # Food cost
        ind['resource'] -= FOOD_COST
        if ind['resource'] < 0:
            ind['hp'] -= 0.5 + abs(ind['resource']) * 0.1
            ind['xp'] *= 0.995
        # Exhaustion penalty
        if ind['rest_energy'] < 30.0:
            ind['hp'] -= 0.2
        # Death check
        if ind['hp'] <= 0:
            ind['alive'] = False
        ind['age'] += 1
    # Phase 5: Reproduction with egg groups
    species_to_inds = defaultdict(list)
    for ind in pop:
        if ind['alive']:
            species_to_inds[ind['species_id']].append(ind)
    offspring = []
    # Process each species
    for sid, inds in species_to_inds.items():
        if len(inds) == 0:
            continue
        # Legendaries don't breed
        if inds[0]['is_legendary'] and inds[0]['name'] not in ['Ditto', 'Mew']:
            continue
        viable_breeders = [ind for ind in inds if ind['mating_readiness'] >= 40.0]
        if len(viable_breeders) < 2:
            continue
        # Create breeding pairs considering egg groups
        for _ in range(len(viable_breeders) // 2):
            if not viable_breeders:
                break
            # Select first parent weighted by fitness
            scores = np.array([
                max(0.1, ind['resource'] * ind['level'] * (ind['mating_readiness'] / 100.0))
                for ind in viable_breeders
            ])
            if scores.sum() <= 0:
                break
            p1_idx = np.random.choice(len(viable_breeders), p=scores / scores.sum())
            p1 = viable_breeders.pop(p1_idx)
            
            if not viable_breeders:
                break
            # Find compatible mate
            compatible = [ind for ind in viable_breeders if can_breed(p1, ind)]
            if not compatible:
                continue
            # Select second parent
            scores2 = np.array([
                max(0.1, ind['resource'] * ind['level'] * (ind['mating_readiness'] / 100.0))
                for ind in compatible
            ])
            p2_idx = np.random.choice(len(compatible), p=scores2 / scores2.sum())
            p2 = compatible[p2_idx]
            viable_breeders.remove(p2)
            # Breeding success
            p1['mating_readiness'] = max(0.0, p1['mating_readiness'] - 20.0)
            p2['mating_readiness'] = max(0.0, p2['mating_readiness'] - 20.0)
            # Determine offspring species
            if p1['name'] == 'Ditto':
                # Ditto breeds -> 40% Ditto, 60% other parent
                offspring_species = p2['species_id'] if np.random.rand() > 0.4 else p1['species_id']
            elif p2['name'] == 'Ditto':
                offspring_species = p1['species_id'] if np.random.rand() > 0.4 else p2['species_id']
            elif p1['name'] == 'Mew':
                offspring_species = p2['species_id'] if np.random.rand() > 0.4 else p1['species_id']
            elif p2['name'] == 'Mew':
                offspring_species = p1['species_id'] if np.random.rand() > 0.4 else p2['species_id']
            else:
                # Normal breeding - offspring is parent 1's species
                offspring_species = p1['species_id']
            
            # Create offspring
            proto_row = prototypes[prototypes['species_id'] == offspring_species].iloc[0]
            child = create_individual(proto_row)
            
            # Genetic drift and mutation
            if np.random.rand() < MUT_PROB:
                child['base_stats'] += np.random.normal(0, MUT_SIGMA, size=6)
                child['base_stats'] = np.maximum(child['base_stats'], 1.0)
            
            # Random drift
            if np.random.rand() < GENETIC_DRIFT_RATE:
                drift = np.random.normal(0, 0.5, size=6)
                child['base_stats'] += drift
                child['base_stats'] = np.maximum(child['base_stats'], 1.0)
            
            child['resource'] = R0
            child['xp'] = 0.0
            child['level'] = 1
            child['hp'] = child['base_stats'][0]
            child['generation_born'] = gen_num
            
            offspring.append(child)
    
    pop.extend(offspring)
    
    # Carrying capacity
    alive_inds = [ind for ind in pop if ind['alive']]
    if len(alive_inds) > K_TOTAL:
        alive_inds.sort(key=lambda x: x['xp'])
        to_kill = alive_inds[:len(alive_inds) - K_TOTAL]
        for ind in to_kill:
            ind['alive'] = False
    
    # Aggregate statistics
    species_agg = {}
    for sid in prototypes['species_id'].values:
        members = [ind for ind in pop if ind['alive'] and ind['species_id'] == sid]
        if not members:
            continue
        
        stat_sums = np.sum([ind['current_stats'] for ind in members], axis=0)
        total_biomass = stat_sums.sum()
        total_xp = sum(ind['xp'] for ind in members)
        mean_level = np.mean([ind['level'] for ind in members])
        max_level = max(ind['level'] for ind in members)
        
        species_agg[sid] = {
            "count": len(members),
            "total_biomass": total_biomass,
            "total_xp": total_xp,
            "mean_level": mean_level,
            "max_level": max_level,
            "HP": stat_sums[0],
            "Atk": stat_sums[1],
            "Def": stat_sums[2],
            "SpA": stat_sums[3],
            "SpD": stat_sums[4],
            "Speed": stat_sums[5]
        }
    
    return species_agg

# Run simulation
print("\n🎮 Starting Pokémon Evolution Simulation\n")

gen_logs = []
species_time_series = defaultdict(list)
species_count_series = defaultdict(list)
extinction_events = {}

for g in range(GENERATIONS):
    agg = run_generation(population, g)
    
    total_pop = sum(v['count'] for v in agg.values()) if agg else 0
    species_richness = len(agg)
    
    # Track extinctions
    for sid in prototypes['species_id'].values:
        if sid not in agg and sid not in extinction_events:
            extinction_events[sid] = g
    
    # Log generation
    gen_logs.append({
        "gen": g,
        "total_pop": total_pop,
        "species_richness": species_richness
    })
    
    # Time series
    for sid in prototypes['species_id'].values:
        if sid in agg:
            species_time_series[sid].append(agg[sid]['total_biomass'])
            species_count_series[sid].append(agg[sid]['count'])
        else:
            species_time_series[sid].append(0.0)
            species_count_series[sid].append(0)
    
    if (g + 1) % 20 == 0 or g == 0:
        print(f"Gen {g+1}/{GENERATIONS} - Pop: {total_pop}, Species: {species_richness}")

print("\n✅ Simulation Complete!\n")

# Final analysis
final_agg = {}
for sid in prototypes['species_id'].values:
    members = [ind for ind in population if ind['alive'] and ind['species_id'] == sid]
    if not members:
        continue
    
    stat_sums = np.sum([ind['current_stats'] for ind in members], axis=0)
    proto = prototypes[prototypes['species_id'] == sid].iloc[0]
    
    final_agg[sid] = {
        "species_id": sid,
        "name": proto['name'],
        "count": len(members),
        "total_biomass": stat_sums.sum(),
        "total_xp": sum(ind['xp'] for ind in members),
        "max_level": max(ind['level'] for ind in members),
        "diet": proto['diet'],
        "egg_group": proto['egg_group1']
    }

final_df = pd.DataFrame(list(final_agg.values()))
final_df = final_df.sort_values("count", ascending=False).reset_index(drop=True)

# === VISUALIZATIONS ===

# 1. Top 20 Species by Population
plt.figure(figsize=(14, 6))
top20 = final_df.head(20)
colors = ['green' if d == 'herbivore' else 'red' if d == 'carnivore' else 'purple' 
          for d in top20['diet']]
plt.bar(range(len(top20)), top20['count'], color=colors)
plt.xticks(range(len(top20)), top20['name'], rotation=45, ha='right')
plt.title("🏆 Top 20 Species by Population (Final Generation)", fontsize=14, fontweight='bold')
plt.ylabel("Population Count")
plt.xlabel("Species")
plt.legend(handles=[
    plt.Rectangle((0,0),1,1, color='green', label='Herbivore'),
    plt.Rectangle((0,0),1,1, color='red', label='Carnivore'),
    plt.Rectangle((0,0),1,1, color='purple', label='Parasite')
])
plt.tight_layout()
plt.show()

# 2. Species Richness Over Time
gens = [log['gen'] for log in gen_logs]
richness = [log['species_richness'] for log in gen_logs]

plt.figure(figsize=(10, 5))
plt.plot(gens, richness, linewidth=2, color='blue')
plt.fill_between(gens, richness, alpha=0.3)
plt.title("📊 Species Richness Over Generations", fontsize=14, fontweight='bold')
plt.xlabel("Generation")
plt.ylabel("Number of Living Species")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. Population Dynamics of Top 10 Final Species
plt.figure(figsize=(12, 6))
top10_ids = final_df.head(10)['species_id'].tolist()
for sid in top10_ids:
    name = prototypes[prototypes['species_id'] == sid].iloc[0]['name']
    plt.plot(gens, species_count_series[sid], label=name, linewidth=2)

plt.title("📈 Population Dynamics: Top 10 Final Species", fontsize=14, fontweight='bold')
plt.xlabel("Generation")
plt.ylabel("Population Count")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# 4. Extinction Timeline
plt.figure(figsize=(12, 6))
extinct_data = []
for sid, gen in extinction_events.items():
    name = prototypes[prototypes['species_id'] == sid].iloc[0]['name']
    extinct_data.append((gen, name))
extinct_data.sort()
if extinct_data:
    extinction_gens, extinction_names = zip(*extinct_data)
    plt.scatter(extinction_gens, range(len(extinction_gens)), alpha=0.6, s=50)
    plt.title("💀 Species Extinction Timeline", fontsize=14, fontweight='bold')
    plt.xlabel("Generation")
    plt.ylabel("Cumulative Extinctions")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
# 5. Game Theory: Predator-Prey Dynamics
herbivore_pops = []
carnivore_pops = []
parasite_pops = []
for g in range(GENERATIONS):
    herb_count = 0
    carn_count = 0
    para_count = 0
    for sid in prototypes['species_id'].values:
        count = species_count_series[sid][g]
        diet = prototypes[prototypes['species_id'] == sid].iloc[0]['diet']
        if diet == 'herbivore':
            herb_count += count
        elif diet == 'carnivore':
            carn_count += count
        else:
            para_count += count
    herbivore_pops.append(herb_count)
    carnivore_pops.append(carn_count)
    parasite_pops.append(para_count)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
# Time series
ax1.plot(gens, herbivore_pops, label='Herbivores (Prey)', color='green', linewidth=2)
ax1.plot(gens, carnivore_pops, label='Carnivores (Predators)', color='red', linewidth=2)
ax1.plot(gens, parasite_pops, label='Parasites', color='purple', linewidth=2)
ax1.set_title("🎮 Game Theory: Predator-Prey-Parasite Dynamics", fontsize=14, fontweight='bold')
ax1.set_xlabel("Generation")
ax1.set_ylabel("Population Count")
ax1.legend()
ax1.grid(True, alpha=0.3)
# Phase space (Predator vs Prey)
ax2.plot(herbivore_pops, carnivore_pops, alpha=0.6, linewidth=1)
ax2.scatter(herbivore_pops[0], carnivore_pops[0], c='blue', s=100, label='Start', zorder=5)
ax2.scatter(herbivore_pops[-1], carnivore_pops[-1], c='red', s=100, label='End', zorder=5)
ax2.set_title("Phase Space: Carnivore vs Herbivore", fontsize=12, fontweight='bold')
ax2.set_xlabel("Herbivore Population")
ax2.set_ylabel("Carnivore Population")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# 6. PCA Analysis
print("\n🔬 Running PCA Analysis...\n")
# Prepare data for PCA
pca_data = []
pca_labels = []
pca_colors = []

for sid in final_df['species_id'].values:
    members = [ind for ind in population if ind['alive'] and ind['species_id'] == sid]
    if members:
        for ind in members[:5]:  # Sample up to 5 individuals per species
            pca_data.append(ind['current_stats'])
            pca_labels.append(ind['name'])
            diet = ind['diet']
            if diet == 'herbivore':
                pca_colors.append('green')
            elif diet == 'carnivore':
                pca_colors.append('red')
            else:
                pca_colors.append('purple')

if len(pca_data) > 10:
    pca_data = np.array(pca_data)
    # Standardize
    scaler = StandardScaler()
    pca_data_scaled = scaler.fit_transform(pca_data)
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(pca_data_scaled)
    # Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                         c=pca_colors, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    
    plt.title("🧬 PCA: Pokémon Trait Space\n(HP, Atk, Def, SpA, SpD, Speed)", 
              fontsize=14, fontweight='bold')
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                   markersize=10, label='Herbivore'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=10, label='Carnivore'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', 
                   markersize=10, label='Parasite')
    ])
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # PCA loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    stat_names = ['HP', 'Atk', 'Def', 'SpA', 'SpD', 'Speed']
    plt.figure(figsize=(10, 6))
    for i, stat in enumerate(stat_names):
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                 head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        plt.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, stat, 
                fontsize=12, fontweight='bold')
    
    plt.title("PCA Loadings: Contribution of Each Stat", fontsize=14, fontweight='bold')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
# === FINAL SUMMARY ===
print("\n" + "="*60)
print("📊 FINAL SIMULATION RESULTS")
print("="*60)

print(f"\n🌍 Total Living Population: {final_df['count'].sum()}")
print(f"🦋 Species Surviving: {len(final_df)} / {len(prototypes)}")
print(f"💀 Species Extinct: {len(extinction_events)}")

print("\n🏆 TOP 10 MOST DOMINANT SPECIES:")
print("-" * 60)
for i, row in final_df.head(10).iterrows():
    print(f"{i+1:2d}. {row['name']:15s} | Pop: {row['count']:4d} | "
          f"Diet: {row['diet']:10s} | Max Lvl: {row['max_level']:3.0f}")

print("\n🍃 DIET DISTRIBUTION (Final):")
diet_counts = final_df.groupby('diet')['count'].sum()
for diet, count in diet_counts.items():
    pct = (count / final_df['count'].sum()) * 100
    print(f"  {diet.capitalize():12s}: {count:4d} ({pct:5.1f}%)")

print("\n🥚 EGG GROUP DISTRIBUTION (Top 5):")
egg_counts = final_df.groupby('egg_group')['count'].sum().sort_values(ascending=False)
for egg_group, count in egg_counts.head(5).items():
    pct = (count / final_df['count'].sum()) * 100
    print(f"  {egg_group:15s}: {count:4d} ({pct:5.1f}%)")

# Special Pokémon status
special_pokemon = ['Mewtwo', 'Mew', 'Ditto', 'Dragonite', 'Charizard']
print("\n✨ SPECIAL POKÉMON STATUS:")
print("-" * 60)
for name in special_pokemon:
    row = final_df[final_df['name'] == name]
    if len(row) > 0:
        r = row.iloc[0]
        print(f"  {name:12s}: ✅ ALIVE | Pop: {r['count']:3d} | Level: {r['max_level']:3.0f}")
    else:
        gen = extinction_events.get(
            prototypes[prototypes['name'] == name]['species_id'].values[0], 
            'Unknown'
        )
        print(f"  {name:12s}: 💀 EXTINCT (Gen {gen})")

print("\n" + "="*60)
print("✅ Analysis Complete!")
print("="*60)