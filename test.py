import unittest
from pulp import value
import main  

class TestAllocation(unittest.TestCase):
    """
    Tests unitaires pour vérifier l'allocation des modèles IA sur les GPU.
    """

    def test_demande_respectee(self):
        """
        Vérifie que la demande minimale de chaque modèle est entièrement satisfaite.
        """
        models = main.models
        gpus = main.gpus
        x = main.x
        demand = main.demand

        for m in models:
            total = sum(value(x[m, g]) for g in gpus)
            self.assertEqual(int(total), demand[m],
                             f"Demande non respectée pour {m}")

    def test_gpu_surcharge(self):
        """
        Vérifie que les heures utilisées sur chaque GPU ne dépassent pas la disponibilité.
        """
        gpus = main.gpus
        x = main.x
        time_per_model = main.time_per_model
        hours_available = main.hours_available
        models = main.models

        for g in gpus:
            used_hours = sum(value(x[m, g]) * time_per_model[m, g] for m in models)
            self.assertLessEqual(used_hours, hours_available[g],
                                 f"GPU {g} surchargé !")

    def test_quotas_specifiques(self):
        """
        Vérifie que les quotas spécifiques sont respectés :
        - BERT sur A100+V100 >= 10
        - ResNet sur T4 <= 20
        - LSTM sur V100+T4 >= 15
        - Quota global A100 <= 25
        """
        x = main.x

        self.assertGreaterEqual(value(x["BERT", "A100"] + x["BERT", "V100"]), 10,
                                "Quota BERT sur A100+V100 non respecté")
        self.assertLessEqual(value(x["ResNet", "T4"]), 20,
                             "Quota ResNet sur T4 non respecté")
        self.assertGreaterEqual(value(x["LSTM", "V100"] + x["LSTM", "T4"]), 15,
                                "Quota LSTM sur V100+T4 non respecté")
        self.assertLessEqual(value(x["BERT", "A100"] + x["ResNet", "A100"] + x["LSTM", "A100"]), 25,
                             "Quota global A100 non respecté")


        
def test_goulot_detranglement(self):
    """
    Vérifie que le goulot d'étranglement (GPU le plus utilisé) ne dépasse pas 100% de sa capacité.
    """
    gpus = main.gpus
    x = main.x
    time_per_model = main.time_per_model
    hours_available = main.hours_available
    models = main.models

    # Calcul du pourcentage d'utilisation de chaque GPU
    utilization = {}
    for g in gpus:
        used_hours = sum(value(x[m, g]) * time_per_model[m, g] for m in models)
        utilization[g] = used_hours / hours_available[g] * 100

    # Identification du goulot d'étranglement
    bottleneck = max(utilization, key=utilization.get)

    # Vérifie que l'utilisation du GPU le plus sollicité <= 100%
    self.assertLessEqual(utilization[bottleneck], 100,
                         f"GPU {bottleneck} dépasse sa capacité ! ({utilization[bottleneck]:.2f}%)")


if __name__ == "__main__":
    unittest.main()

