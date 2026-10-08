"""Tests du moteur Streamlit : lancer  python -m unittest  depuis la racine."""

import random
import unittest

import moteur as M


class TestPrimalite(unittest.TestCase):
    def test_is_prime(self):
        premiers = [p for p in range(200) if M.is_prime(p)]
        attendus = [p for p in range(2, 200) if all(p % d for d in range(2, p))]
        self.assertEqual(premiers, attendus)
        self.assertTrue(M.is_prime(M.P_MERSENNE_61))
        self.assertFalse(M.is_prime(M.P_MERSENNE_61 + 2))   # 2^61 + 1 divisible par 3

    def test_next_prime(self):
        self.assertEqual(M.next_prime(10), 11)
        self.assertEqual(M.next_prime(11), 13)
        self.assertEqual(M.next_prime(0), 2)


class TestTable(unittest.TestCase):
    def test_operations_et_instrumentation(self):
        t = M.HashTableChaining(M.ModuloHash(7))
        for k in (3, 10, 17, 4):
            self.assertTrue(t.insert(k).success)
        self.assertFalse(t.insert(10).success)            # doublon refusé
        self.assertEqual(t.n, 4)
        self.assertEqual(list(t.buckets[3]), [17, 10, 3])  # insertion en tête
        self.assertEqual(t.insert_collisions, 2)
        rec = t.search(3)
        self.assertTrue(rec.success)
        self.assertEqual(rec.comparisons, 3)
        self.assertEqual(t.search(24).comparisons, 3)      # échec : chaîne entière
        self.assertEqual(t.search(5).comparisons, 0)
        self.assertEqual(t.chain_path(10), [17, 10])
        self.assertTrue(t.delete(10).success)
        self.assertFalse(t.delete(10).success)
        self.assertEqual(sorted(t.keys()), [3, 4, 17])
        s = t.stats()
        self.assertEqual((s.n, s.max_chain, s.colliding_pairs), (3, 2, 1))

    def test_cout_moyen_exact(self):
        """1 + paires/n = moyenne mesurée en cherchant chaque clé."""
        rng = random.Random(3)
        for name in M.HASH_FAMILIES:
            keys = M.keys_uniform(400, rng=rng)
            t = M.build_table(keys, M.make_hash(name, 37, rng))
            mesure = sum(t.search(k, record=False).comparisons for k in keys) / len(keys)
            self.assertAlmostEqual(mesure, t.stats().expected_success_cost)

    def test_rehash_garde_les_cles_et_m(self):
        u = M.CarterWegmanHash(16, rng=random.Random(1))
        keys = M.keys_adversarial_known_universal(100, u)
        t = M.build_table(keys, u)
        self.assertEqual(t.stats().max_chain, 100)
        t.rehash(rng=random.Random(2))
        self.assertEqual((t.m, sorted(t.keys())), (16, sorted(keys)))
        self.assertLess(t.stats().max_chain, 30)


class TestFonctions(unittest.TestCase):
    def test_images_dans_la_table(self):
        rng = random.Random(4)
        for name in M.HASH_FAMILIES:
            for m in (1, 2, 10, 1000):
                h = M.make_hash(name, m, rng)
                for k in [0, 1, 2**32, M.P_MERSENNE_61 - 1] + rng.sample(range(10**12), 100):
                    self.assertTrue(0 <= h(k) < m)

    def test_knuth_proche_de_la_formule_reelle(self):
        h = M.KnuthHash(1000)
        for k in range(1, 5000):
            self.assertLessEqual(abs(h(k) - int(1000 * ((k * M.KNUTH_A) % 1))), 1)

    def test_attaques(self):
        m = 32
        self.assertEqual({k % m for k in M.keys_adversarial_modulo(200, m, 5)}, {5})
        kn = M.keys_adversarial_knuth(200, m, 7)
        self.assertEqual(len(set(kn)), 200)
        self.assertEqual({M.KnuthHash(m)(k) for k in kn}, {7})
        u = M.CarterWegmanHash(m)
        cw = M.keys_adversarial_known_universal(200, u, 9)
        self.assertEqual(len(set(cw)), 200)
        self.assertEqual({u(k) for k in cw}, {9})

    def test_parametres_invalides(self):
        with self.assertRaises(ValueError):
            M.CarterWegmanHash(10, p=15)
        with self.assertRaises(ValueError):
            M.CarterWegmanHash(10, p=7)


class TestTheorie(unittest.TestCase):
    def test_probabilite_exacte_egale_enumeration(self):
        for p, m in ((13, 4), (17, 5), (11, 10)):
            attendu = M.exact_collision_probability(p, m)
            self.assertLessEqual(attendu, 1 / m)
            for x, y in ((0, 1), (2, 9), (3, p - 1)):
                cnt = sum(((a * x + b) % p) % m == ((a * y + b) % p) % m
                          for a in range(1, p) for b in range(p))
                self.assertAlmostEqual(cnt / (p * (p - 1)), attendu)

    def test_experience_monte_carlo(self):
        e = M.collision_experiment(7, 17, 10, 20_000, seed=1)
        self.assertLess(abs(e.frequency - e.exact), 4 * (0.1 * 0.9 / 20_000) ** 0.5)
        self.assertEqual(len(e.running_frequency), 20_000)
        self.assertEqual(M.random_shift_collision_frequency(7, 17, 10, 500, 1), 1.0)

    def test_benchmark(self):
        pts = M.run_benchmark([200, 400], alpha=1.0, probes=100, draws=2)
        adv = {p.n: p for p in pts if p.scenario == "k mod m · clés adverse"}
        self.assertEqual(adv[400].max_chain, 400)
        self.assertEqual(adv[400].insert_comparisons, 400 * 399 // 2)
        uni = [p for p in pts if p.scenario == "Carter-Wegman · clés adverse"]
        self.assertTrue(all(p.mean_comparisons < 10 for p in uni))


if __name__ == "__main__":
    unittest.main()
