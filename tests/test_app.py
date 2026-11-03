"""Test de lancement du tableau de bord : python -m unittest depuis la racine."""

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"


def tiles(at: AppTest) -> str:
    return " ".join(md.value for md in at.markdown if "class='kpi'" in md.value)


class TestTableauDeBord(unittest.TestCase):
    def test_chiffres_de_la_soutenance(self):
        at = AppTest.from_file(str(APP), default_timeout=180).run()
        self.assertFalse(at.exception)
        t = tiles(at)
        for attendu in ("640", "11", "204 480", "2 892", "3 195", "A × 71", "A × 58", "0,0162"):
            self.assertIn(attendu, t)

    def test_fuite_puis_nouveau_tirage(self):
        at = AppTest.from_file(str(APP), default_timeout=180).run()
        at.segmented_control(key="scen").set_value("leak").run()
        self.assertFalse(at.exception)
        self.assertIn("B × ", tiles(at))                       # B s'effondre
        at.button(key="draw").click().run()                 # nouveau tirage (barre latérale)
        self.assertFalse(at.exception)
        self.assertNotIn("<i>B</i>640", tiles(at))             # B est réparée


if __name__ == "__main__":
    unittest.main()
