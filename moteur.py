"""Moteur algorithmique du démonstrateur Streamlit (Projet 7).

Ce module est en Python pur : il n'importe ni Streamlit ni Plotly, ce qui
permet de le tester seul (``python -m unittest``) et de le réutiliser.

Contenu
-------
* ``HashTableChaining``  table de hachage à chaînage (vraies listes chaînées),
  instrumentée : insertions, collisions, longueurs de chaînes, comparaisons.
* Fonctions de hachage :
    - ``ModuloHash``          h(k) = k mod m                         (déterministe)
    - ``KnuthHash``           h(k) = ⌊m · (k·A mod 1)⌋, A = (√5 − 1)/2  (déterministe)
    - ``CarterWegmanHash``    h_{a,b}(k) = ((a·k + b) mod p) mod m    (2-universelle)
    - ``RandomShiftHash``     h_b(k) = (k + b) mod m   (aléatoire mais NON universelle,
                              contre-exemple pédagogique)
* Générateurs de jeux de clés : uniforme, séquentiel, adverse (k_i = i·m + c),
  et adversaire « adaptatif » qui connaît la fonction tirée.
* Outils théoriques : test de primalité, probabilité EXACTE de collision de la
  famille de Carter-Wegman, coûts attendus des différentes structures.

Conventions : toutes les clés sont des entiers naturels ``0 <= k < p``.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Optional, Protocol, Sequence

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------

#: Nombre premier de Mersenne 2^61 − 1 : p par défaut de la famille universelle.
#: Il dépasse toutes les clés manipulées par le démonstrateur.
P_MERSENNE_61: int = (1 << 61) - 1

#: Taille de mot w utilisée pour l'implémentation exacte (en virgule fixe) de
#: la méthode multiplicative de Knuth.
KNUTH_W: int = 32

#: A_w = ⌊2^w · (√5 − 1)/2⌋ : la constante de Knuth en virgule fixe (impaire).
KNUTH_A_FIXED: int = 2_654_435_769

#: Valeur réelle de A, pour l'affichage.
KNUTH_A: float = (math.sqrt(5) - 1) / 2

#: Borne supérieure (exclue) de l'univers des clés aléatoires générées.
UNIVERSE_DEFAULT: int = 10**9


# --------------------------------------------------------------------------
# Arithmétique : primalité
# --------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    """Test de primalité de Miller-Rabin, déterministe pour n < 3,3·10^24.

    Les 12 premières bases premières suffisent pour tous les entiers de
    moins de 3,3·10^24, ce qui couvre largement l'usage du démonstrateur.
    """
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for q in small_primes:
        if n % q == 0:
            return n == q
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in small_primes:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def next_prime(n: int) -> int:
    """Plus petit nombre premier strictement supérieur à ``n``."""
    candidate = max(2, n + 1)
    while not is_prime(candidate):
        candidate += 1
    return candidate


# --------------------------------------------------------------------------
# Fonctions de hachage
# --------------------------------------------------------------------------

class HashFunction(Protocol):
    """Interface commune : un objet appelable clé -> numéro d'alvéole."""

    name: str
    m: int
    randomized: bool
    universal: bool

    def __call__(self, key: int) -> int: ...

    def formula(self) -> str: ...

    def explain(self, key: int) -> str: ...

    def redraw(self, rng: Optional[random.Random] = None) -> None: ...


class ModuloHash:
    """Fonction déterministe naïve : h(k) = k mod m.

    Connue de tous, elle est trivialement attaquable : les clés
    k_i = i·m + c tombent toutes dans l'alvéole c.
    """

    name: str = "Modulo  (k mod m)"
    randomized: bool = False
    universal: bool = False

    def __init__(self, m: int) -> None:
        if m < 1:
            raise ValueError("m doit être >= 1")
        self.m: int = m

    def __call__(self, key: int) -> int:
        return key % self.m

    def formula(self) -> str:
        return rf"h(k) = k \bmod {self.m}"

    def explain(self, key: int) -> str:
        return rf"h({key}) = {key} \bmod {self.m} = {self(key)}"

    def redraw(self, rng: Optional[random.Random] = None) -> None:
        """Rien à tirer : la fonction est fixée une fois pour toutes."""


class KnuthHash:
    """Méthode multiplicative de Knuth : h(k) = ⌊m · (k·A mod 1)⌋.

    Implémentation EXACTE en virgule fixe sur w = 32 bits (comme Knuth,
    TAOCP vol. 3, §6.4) : avec A_w = ⌊2^w·A⌋,
        k·A mod 1  ≈  (k·A_w mod 2^w) / 2^w,
    d'où h(k) = ⌊m · (k·A_w mod 2^w) / 2^w⌋. On évite ainsi les erreurs
    d'arrondi des flottants pour les grandes clés.
    """

    name: str = "Multiplicative (Knuth)"
    randomized: bool = False
    universal: bool = False

    def __init__(self, m: int) -> None:
        if m < 1:
            raise ValueError("m doit être >= 1")
        self.m: int = m

    def __call__(self, key: int) -> int:
        frac = (key * KNUTH_A_FIXED) & ((1 << KNUTH_W) - 1)
        return (frac * self.m) >> KNUTH_W

    def formula(self) -> str:
        return (rf"h(k) = \left\lfloor {self.m}\cdot (k\,A \bmod 1) \right\rfloor,"
                rf"\quad A = \tfrac{{\sqrt5-1}}{{2}} \approx {KNUTH_A:.6f}")

    def explain(self, key: int) -> str:
        frac = ((key * KNUTH_A_FIXED) & ((1 << KNUTH_W) - 1)) / (1 << KNUTH_W)
        return (rf"{key}\cdot A \bmod 1 \approx {frac:.6f} \;\Rightarrow\; "
                rf"h({key}) = \lfloor {self.m} \times {frac:.6f} \rfloor = {self(key)}")

    def redraw(self, rng: Optional[random.Random] = None) -> None:
        """Rien à tirer : la fonction est fixée une fois pour toutes."""


class CarterWegmanHash:
    """Famille 2-universelle de Carter et Wegman (1979).

        h_{a,b}(k) = ((a·k + b) mod p) mod m,
        p premier, p > max(clés, m),  a ∈ {1..p−1},  b ∈ {0..p−1}.

    Théorème : pour toutes clés x ≠ y dans {0..p−1},
        Pr_{(a,b)}[ h_{a,b}(x) = h_{a,b}(y) ] ≤ 1/m.
    """

    name: str = "Universelle (Carter-Wegman)"
    randomized: bool = True
    universal: bool = True

    def __init__(self, m: int, p: int = P_MERSENNE_61,
                 rng: Optional[random.Random] = None,
                 a: Optional[int] = None, b: Optional[int] = None) -> None:
        if m < 1:
            raise ValueError("m doit être >= 1")
        if not is_prime(p):
            raise ValueError(f"p = {p} n'est pas premier")
        if p <= m:
            raise ValueError("p doit être strictement supérieur à m")
        self.m: int = m
        self.p: int = p
        self.a: int = 1
        self.b: int = 0
        self.redraw(rng)
        if a is not None:
            if not 1 <= a < p:
                raise ValueError("a doit appartenir à {1, …, p−1}")
            self.a = a
        if b is not None:
            if not 0 <= b < p:
                raise ValueError("b doit appartenir à {0, …, p−1}")
            self.b = b

    def redraw(self, rng: Optional[random.Random] = None) -> None:
        """Tire une nouvelle fonction (a, b) uniformément dans la famille."""
        r = rng or random.SystemRandom()
        self.a = r.randrange(1, self.p)
        self.b = r.randrange(0, self.p)

    def __call__(self, key: int) -> int:
        return ((self.a * key + self.b) % self.p) % self.m

    def formula(self) -> str:
        return rf"h_{{a,b}}(k) = \big((a\,k + b) \bmod p\big) \bmod {self.m}"

    def explain(self, key: int) -> str:
        inner = (self.a * key + self.b) % self.p
        return (rf"(a\cdot{key} + b) \bmod p = {inner} \;\Rightarrow\; "
                rf"h({key}) = {inner} \bmod {self.m} = {self(key)}")


class RandomShiftHash:
    """Contre-exemple : h_b(k) = (k + b) mod m, b aléatoire.

    Il y a du hasard, mais la famille n'est PAS universelle : si
    y = x + m, alors h_b(x) = h_b(y) pour TOUT b (probabilité 1 ≫ 1/m).
    """

    name: str = "Décalage aléatoire (non universel)"
    randomized: bool = True
    universal: bool = False

    def __init__(self, m: int, rng: Optional[random.Random] = None) -> None:
        self.m: int = m
        self.b: int = 0
        self.redraw(rng)

    def redraw(self, rng: Optional[random.Random] = None) -> None:
        self.b = (rng or random.SystemRandom()).randrange(self.m)

    def __call__(self, key: int) -> int:
        return (key + self.b) % self.m

    def formula(self) -> str:
        return rf"h_b(k) = (k + b) \bmod {self.m}"

    def explain(self, key: int) -> str:
        return rf"h({key}) = ({key} + {self.b}) \bmod {self.m} = {self(key)}"


#: Noms affichés -> constructeurs (m, rng) -> fonction de hachage.
HASH_FAMILIES: dict[str, Callable[[int, Optional[random.Random]], HashFunction]] = {
    ModuloHash.name: lambda m, rng: ModuloHash(m),
    KnuthHash.name: lambda m, rng: KnuthHash(m),
    CarterWegmanHash.name: lambda m, rng: CarterWegmanHash(m, rng=rng),
}


def make_hash(name: str, m: int, rng: Optional[random.Random] = None) -> HashFunction:
    """Instancie la fonction de hachage ``name`` pour une table de taille m."""
    return HASH_FAMILIES[name](m, rng)


# --------------------------------------------------------------------------
# Table de hachage à chaînage, instrumentée
# --------------------------------------------------------------------------

class Node:
    """Maillon d'une liste simplement chaînée."""

    __slots__ = ("key", "next")

    def __init__(self, key: int, next_node: Optional["Node"] = None) -> None:
        self.key: int = key
        self.next: Optional[Node] = next_node


class LinkedList:
    """Liste simplement chaînée : une alvéole (bucket) de la table."""

    __slots__ = ("head", "size")

    def __init__(self) -> None:
        self.head: Optional[Node] = None
        self.size: int = 0

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[int]:
        node = self.head
        while node is not None:
            yield node.key
            node = node.next

    def find(self, key: int) -> tuple[bool, int]:
        """Parcourt la chaîne. Renvoie (trouvée, nombre de comparaisons)."""
        comparisons = 0
        node = self.head
        while node is not None:
            comparisons += 1
            if node.key == key:
                return True, comparisons
            node = node.next
        return False, comparisons

    def push_front(self, key: int) -> None:
        """Insertion en tête, O(1)."""
        self.head = Node(key, self.head)
        self.size += 1

    def remove(self, key: int) -> tuple[bool, int]:
        """Supprime la clé. Renvoie (supprimée, nombre de comparaisons)."""
        comparisons = 0
        prev: Optional[Node] = None
        node = self.head
        while node is not None:
            comparisons += 1
            if node.key == key:
                if prev is None:
                    self.head = node.next
                else:
                    prev.next = node.next
                self.size -= 1
                return True, comparisons
            prev, node = node, node.next
        return False, comparisons


@dataclass
class OperationRecord:
    """Trace d'une opération sur la table (pour le journal de l'interface)."""

    operation: str          # "insertion", "recherche", "suppression"
    key: int
    bucket: int
    comparisons: int        # comparaisons de clés effectuées
    success: bool
    chain_length: int       # longueur de la chaîne APRÈS l'opération
    collision: bool = False  # insertion dans une alvéole déjà occupée


@dataclass
class TableStats:
    """Instantané des métriques de la table."""

    n: int
    m: int
    load_factor: float
    max_chain: int
    mean_nonempty_chain: float
    empty_ratio: float
    colliding_pairs: int            # Σ C(L_i, 2)
    insert_collisions: int          # insertions tombées dans une alvéole non vide
    expected_success_cost: float    # coût moyen exact d'une recherche fructueuse
    expected_failure_cost: float    # coût moyen exact d'une recherche infructueuse


@dataclass
class HashTableChaining:
    """Table de hachage à chaînage séparé, instrumentée.

    Chaque alvéole est une ``LinkedList``. L'insertion vérifie d'abord que la
    clé est absente (comme un ``dict`` Python) : son coût est donc celui
    d'une recherche infructueuse, puis O(1) pour l'insertion en tête.
    C'est exactement ce qui rend l'attaque HashDoS quadratique : n
    insertions dans une même chaîne coûtent 0 + 1 + … + (n−1) = n(n−1)/2.
    """

    hash_fn: HashFunction
    buckets: list[LinkedList] = field(init=False)
    n: int = field(init=False, default=0)
    insertions: int = field(init=False, default=0)
    insert_collisions: int = field(init=False, default=0)
    total_comparisons: int = field(init=False, default=0)
    insert_comparisons: int = field(init=False, default=0)
    search_comparisons: int = field(init=False, default=0)
    searches: int = field(init=False, default=0)
    log: list[OperationRecord] = field(init=False, default_factory=list)
    log_limit: int = 200

    def __post_init__(self) -> None:
        self.buckets = [LinkedList() for _ in range(self.hash_fn.m)]

    # ----------------------------------------------------------- propriétés
    @property
    def m(self) -> int:
        """Nombre d'alvéoles."""
        return self.hash_fn.m

    def _record(self, rec: OperationRecord) -> None:
        self.log.append(rec)
        if len(self.log) > self.log_limit:
            del self.log[: len(self.log) - self.log_limit]

    # ----------------------------------------------------------- opérations
    def insert(self, key: int, record: bool = True) -> OperationRecord:
        """Insère ``key`` si elle est absente (coût = parcours de la chaîne)."""
        if key < 0:
            raise ValueError("les clés doivent être des entiers naturels")
        bucket = self.hash_fn(key)
        chain = self.buckets[bucket]
        found, comps = chain.find(key)
        self.total_comparisons += comps
        self.insert_comparisons += comps
        collision = False
        if not found:
            collision = len(chain) > 0
            chain.push_front(key)
            self.n += 1
            self.insertions += 1
            self.insert_collisions += int(collision)
        rec = OperationRecord("insertion", key, bucket, comps, not found,
                              len(chain), collision)
        if record:
            self._record(rec)
        return rec

    def insert_many(self, keys: Iterable[int]) -> None:
        """Insère une suite de clés (sans encombrer le journal)."""
        for key in keys:
            self.insert(key, record=False)

    def search(self, key: int, record: bool = True) -> OperationRecord:
        """Recherche ``key`` ; renvoie la trace (dont le nombre de comparaisons)."""
        bucket = self.hash_fn(key)
        chain = self.buckets[bucket]
        found, comps = chain.find(key)
        self.searches += 1
        self.total_comparisons += comps
        self.search_comparisons += comps
        rec = OperationRecord("recherche", key, bucket, comps, found, len(chain))
        if record:
            self._record(rec)
        return rec

    def delete(self, key: int) -> OperationRecord:
        """Supprime ``key`` si elle est présente."""
        bucket = self.hash_fn(key)
        chain = self.buckets[bucket]
        removed, comps = chain.remove(key)
        self.total_comparisons += comps
        if removed:
            self.n -= 1
        rec = OperationRecord("suppression", key, bucket, comps, removed, len(chain))
        self._record(rec)
        return rec

    def chain_path(self, key: int) -> list[int]:
        """Clés comparées, dans l'ordre, lors de la recherche de ``key``."""
        path: list[int] = []
        for other in self.buckets[self.hash_fn(key)]:
            path.append(other)
            if other == key:
                break
        return path

    def keys(self) -> list[int]:
        """Toutes les clés stockées, alvéole par alvéole."""
        return [k for chain in self.buckets for k in chain]

    def rehash(self, new_hash: Optional[HashFunction] = None,
               rng: Optional[random.Random] = None) -> None:
        """Re-tire la fonction (ou en installe une nouvelle) et réinsère tout.

        La taille m est conservée si ``new_hash`` n'est pas fourni.
        """
        keys = self.keys()
        if new_hash is None:
            self.hash_fn.redraw(rng)
        else:
            self.hash_fn = new_hash
        self.buckets = [LinkedList() for _ in range(self.hash_fn.m)]
        self.n = 0
        self.insert_many(keys)

    def clear(self) -> None:
        """Vide la table et remet les compteurs à zéro."""
        self.buckets = [LinkedList() for _ in range(self.m)]
        self.n = self.insertions = self.insert_collisions = 0
        self.total_comparisons = self.insert_comparisons = 0
        self.search_comparisons = self.searches = 0
        self.log.clear()

    # ------------------------------------------------------------ mesures
    def chain_lengths(self) -> list[int]:
        """Longueur de chaque chaîne (la « répartition »)."""
        return [len(chain) for chain in self.buckets]

    def stats(self) -> TableStats:
        """Calcule toutes les métriques de la table en O(m)."""
        lengths = self.chain_lengths()
        n, m = self.n, self.m
        nonempty = [x for x in lengths if x > 0]
        pairs = sum(x * (x - 1) // 2 for x in lengths)
        # Recherche fructueuse : la i-ème clé d'une chaîne coûte i comparaisons,
        # donc Σ_chaînes L(L+1)/2 au total, soit 1 + pairs/n en moyenne.
        success = (1 + pairs / n) if n else 0.0
        # Recherche infructueuse d'une clé uniforme : on parcourt toute la chaîne.
        failure = n / m
        return TableStats(
            n=n,
            m=m,
            load_factor=n / m,
            max_chain=max(lengths) if lengths else 0,
            mean_nonempty_chain=(sum(nonempty) / len(nonempty)) if nonempty else 0.0,
            empty_ratio=(m - len(nonempty)) / m,
            colliding_pairs=pairs,
            insert_collisions=self.insert_collisions,
            expected_success_cost=success,
            expected_failure_cost=failure,
        )


def build_table(keys: Iterable[int], hash_fn: HashFunction) -> HashTableChaining:
    """Crée une table instrumentée et y insère ``keys``."""
    table = HashTableChaining(hash_fn)
    table.insert_many(keys)
    return table


# --------------------------------------------------------------------------
# Générateurs de jeux de données
# --------------------------------------------------------------------------

def keys_uniform(n: int, upper: int = UNIVERSE_DEFAULT,
                 rng: Optional[random.Random] = None) -> list[int]:
    """n clés DISTINCTES tirées uniformément dans [0, upper]."""
    r = rng or random.Random()
    if n > upper + 1:
        raise ValueError("univers trop petit pour n clés distinctes")
    return r.sample(range(upper + 1), n)


def keys_sequential(n: int, start: int = 1) -> list[int]:
    """1, 2, 3, …, n : cas FAVORABLE pour k mod m (répartition parfaite)."""
    return list(range(start, start + n))


def keys_adversarial_modulo(n: int, m: int, c: int = 0) -> list[int]:
    """Attaque contre k mod m : k_i = i·m + c, toutes dans l'alvéole c."""
    return [i * m + c for i in range(n)]


def keys_adversarial_knuth(n: int, m: int, c: int = 0) -> list[int]:
    """Attaque contre Knuth (fonction CONNUE) : n clés de l'alvéole c.

    h(k) = c  ⟺  k·A_w mod 2^w ∈ [c·2^w/m, (c+1)·2^w/m).
    A_w est impair donc inversible modulo 2^w : pour chaque cible t de cet
    intervalle, k = t · A_w^{-1} mod 2^w convient. Coût : O(n).
    """
    modulus = 1 << KNUTH_W
    inv = pow(KNUTH_A_FIXED, -1, modulus)
    lo = -(-c * modulus // m)                  # ⌈c·2^w/m⌉
    hi = -(-(c + 1) * modulus // m)            # ⌈(c+1)·2^w/m⌉ (exclu)
    if hi - lo < n:
        raise ValueError("pas assez de clés distinctes pour cette alvéole")
    return [(t * inv) % modulus for t in range(lo, lo + n)]


def keys_adversarial_known_universal(n: int, h: CarterWegmanHash,
                                     c: int = 0) -> list[int]:
    """Adversaire qui CONNAÎT (a, b) : n clés envoyées dans l'alvéole c.

    On vise les valeurs internes t_j = c + j·m < p, puis on résout
    a·k + b ≡ t_j (mod p), soit k = (t_j − b)·a^{-1} mod p.
    Démontre que la sécurité repose sur le SECRET du tirage.
    """
    inv_a = pow(h.a, -1, h.p)
    if c + (n - 1) * h.m >= h.p:
        raise ValueError("p trop petit pour produire autant de clés")
    return [((c + j * h.m - h.b) * inv_a) % h.p for j in range(n)]


def keys_adversarial_for(hash_fn: HashFunction, n: int, c: int = 0) -> list[int]:
    """Meilleure attaque connue contre ``hash_fn`` si l'attaquant la connaît."""
    if isinstance(hash_fn, ModuloHash):
        return keys_adversarial_modulo(n, hash_fn.m, c)
    if isinstance(hash_fn, KnuthHash):
        return keys_adversarial_knuth(n, hash_fn.m, c)
    if isinstance(hash_fn, CarterWegmanHash):
        return keys_adversarial_known_universal(n, hash_fn, c)
    raise TypeError(f"pas d'attaque prévue pour {type(hash_fn).__name__}")


DATASETS: tuple[str, ...] = (
    "Uniforme aléatoire",
    "Séquentielle (1, 2, …, n)",
    "Adverse (k_i = i·m + c)",
)


def generate_dataset(kind: str, n: int, m: int, c: int = 0,
                     upper: int = UNIVERSE_DEFAULT,
                     rng: Optional[random.Random] = None) -> list[int]:
    """Génère le jeu de clés ``kind`` (voir ``DATASETS``)."""
    if kind == DATASETS[0]:
        return keys_uniform(n, upper, rng)
    if kind == DATASETS[1]:
        return keys_sequential(n)
    if kind == DATASETS[2]:
        return keys_adversarial_modulo(n, m, c)
    raise ValueError(f"jeu de données inconnu : {kind}")


# --------------------------------------------------------------------------
# Théorie : borne 2-universelle
# --------------------------------------------------------------------------

def exact_collision_probability(p: int, m: int) -> float:
    """Probabilité EXACTE Pr[h_{a,b}(x) = h_{a,b}(y)] pour x ≠ y (Carter-Wegman).

    Preuve (CLRS, th. 11.5) : comme p est premier et x ≢ y (mod p),
    (a, b) ↦ (r, s) = (a·x + b mod p, a·y + b mod p) est une bijection de
    {1..p−1}×{0..p−1} sur les couples r ≠ s. Il y a collision ssi r ≡ s (mod m).
    Si n_c = #{r < p : r ≡ c mod m}, le nombre de couples favorables est
    Σ_c n_c(n_c − 1), d'où une probabilité INDÉPENDANTE de x et y,
    toujours ≤ 1/m.
    """
    q, rem = divmod(p, m)
    favourable = rem * (q + 1) * q + (m - rem) * q * (q - 1)
    return favourable / (p * (p - 1))


@dataclass
class CollisionExperiment:
    """Résultat d'une estimation Monte-Carlo de Pr[h(x) = h(y)]."""

    x: int
    y: int
    m: int
    p: int
    trials: int
    hits: int
    running_frequency: list[float]   # fréquence après 1, 2, …, trials tirages
    exact: float

    @property
    def frequency(self) -> float:
        return self.hits / self.trials if self.trials else 0.0


def collision_experiment(x: int, y: int, m: int, trials: int, p: int = P_MERSENNE_61,
                         seed: Optional[int] = None) -> CollisionExperiment:
    """Tire ``trials`` couples (a, b) indépendants et compte h(x) = h(y)."""
    if x == y:
        raise ValueError("il faut deux clés distinctes")
    if not (0 <= x < p and 0 <= y < p):
        raise ValueError("les clés doivent appartenir à {0, …, p−1}")
    rng = random.Random(seed)
    hits = 0
    running: list[float] = []
    for t in range(1, trials + 1):
        a = rng.randrange(1, p)
        b = rng.randrange(p)
        if ((a * x + b) % p) % m == ((a * y + b) % p) % m:
            hits += 1
        running.append(hits / t)
    return CollisionExperiment(x, y, m, p, trials, hits, running,
                               exact_collision_probability(p, m))


def random_shift_collision_frequency(x: int, y: int, m: int, trials: int,
                                     seed: Optional[int] = None) -> float:
    """Même expérience pour le contre-exemple h_b(k) = (k + b) mod m."""
    rng = random.Random(seed)
    hits = sum(1 for _ in range(trials)
               if (x + (b := rng.randrange(m))) % m == (y + b) % m)
    return hits / trials if trials else 0.0


# --------------------------------------------------------------------------
# Benchmarks
# --------------------------------------------------------------------------

@dataclass
class BenchmarkPoint:
    """Une mesure du benchmark de recherche."""

    scenario: str
    n: int
    m: int
    mean_comparisons: float    # comparaisons moyennes, recherches fructueuses
    max_chain: int
    insert_comparisons: int    # coût total de construction (vérif. des doublons)
    micros_per_search: float   # temps moyen mesuré (µs)


def measure_search(table: HashTableChaining, probes: Sequence[int]) -> tuple[float, float]:
    """Cherche réellement chaque sonde ; renvoie (comparaisons moy., µs moy.)."""
    if not probes:
        return 0.0, 0.0
    total = 0
    start = time.perf_counter()
    for key in probes:
        total += table.search(key, record=False).comparisons
    elapsed = time.perf_counter() - start
    return total / len(probes), 1e6 * elapsed / len(probes)


def run_benchmark(sizes: Sequence[int], alpha: float = 1.0, probes: int = 300,
                  seed: int = 0, draws: int = 5) -> list[BenchmarkPoint]:
    """Coût de recherche en fonction de n, à facteur de charge α constant.

    m = ⌈n/α⌉ (la table est redimensionnée comme dans une vraie implémentation).
    Quatre scénarios : {k mod m, Carter-Wegman} × {clés uniformes, clés adverses
    k_i = i·m}. Les sondes sont des clés présentes tirées au hasard.
    La garantie universelle porte sur une ESPÉRANCE : pour Carter-Wegman, on
    moyenne donc sur ``draws`` tirages indépendants de (a, b).
    """
    rng = random.Random(seed)
    points: list[BenchmarkPoint] = []
    for n in sizes:
        m = max(1, math.ceil(n / alpha))
        datasets = {
            "uniforme": keys_uniform(n, UNIVERSE_DEFAULT, rng),
            "adverse": keys_adversarial_modulo(n, m),
        }
        for data_name, keys in datasets.items():
            sample = rng.sample(keys, min(probes, len(keys)))
            for fn_name in ("k mod m", "Carter-Wegman"):
                repeats = 1 if fn_name == "k mod m" else max(1, draws)
                comps_sum = micros_sum = 0.0
                max_chain = insert_sum = 0
                for _ in range(repeats):
                    fn: HashFunction = (ModuloHash(m) if fn_name == "k mod m"
                                        else CarterWegmanHash(m, rng=rng))
                    table = build_table(keys, fn)
                    comps, micros = measure_search(table, sample)
                    comps_sum += comps
                    micros_sum += micros
                    max_chain = max(max_chain, max(table.chain_lengths()))
                    insert_sum += table.insert_comparisons
                points.append(BenchmarkPoint(
                    scenario=f"{fn_name} · clés {data_name}", n=n, m=m,
                    mean_comparisons=comps_sum / repeats, max_chain=max_chain,
                    insert_comparisons=insert_sum // repeats,
                    micros_per_search=micros_sum / repeats))
    return points


# --------------------------------------------------------------------------
# Coûts théoriques des structures comparées (pour la fiche soutenance)
# --------------------------------------------------------------------------

def theoretical_costs(n: int, alpha: float) -> dict[str, dict[str, float]]:
    """Comparaisons (ou sondages) attendues / garanties pour n clés, charge α.

    Colonnes : recherche fructueuse moyenne, infructueuse moyenne, pire cas.
    * chaînage : 1 + α/2 − 1/(2m) ≈ 1 + α/2 et α (Knuth ; CLRS th. 11.1-11.2) ;
      pour une fonction déterministe ces moyennes supposent des clés « au
      hasard », pour Carter-Wegman elles valent pour TOUT jeu de clés ;
    * sondage linéaire : ½(1 + 1/(1−α)) et ½(1 + 1/(1−α)²) (Knuth 1962) ;
    * double hachage (≈ hachage uniforme) : (1/α)·ln(1/(1−α)) et 1/(1−α) ;
    * arbres : profondeur ≈ log2 n ; hauteur AVL ≤ 1,44·log2(n+2),
      rouge-noir ≤ 2·log2(n+1) — garanties déterministes.
    Pour l'adressage ouvert, α est ramené à 0,99 au plus (il faut α < 1).
    """
    a = min(max(alpha, 1e-9), 0.99)
    log2n = math.log2(max(n, 2))
    return {
        "Chaînage + déterministe (k mod m)": {
            "fructueuse": 1 + alpha / 2, "infructueuse": alpha, "pire": float(n)},
        "Chaînage + universelle (Carter-Wegman)": {
            "fructueuse": 1 + alpha / 2, "infructueuse": alpha, "pire": float(n)},
        "Adressage ouvert — sondage linéaire": {
            "fructueuse": 0.5 * (1 + 1 / (1 - a)),
            "infructueuse": 0.5 * (1 + 1 / (1 - a) ** 2), "pire": float(n)},
        "Adressage ouvert — double hachage": {
            "fructueuse": (1 / a) * math.log(1 / (1 - a)),
            "infructueuse": 1 / (1 - a), "pire": float(n)},
        "Arbre AVL": {
            "fructueuse": log2n, "infructueuse": log2n,
            "pire": 1.44 * math.log2(n + 2)},
        "Arbre rouge-noir": {
            "fructueuse": log2n, "infructueuse": log2n,
            "pire": 2 * math.log2(n + 1)},
    }
