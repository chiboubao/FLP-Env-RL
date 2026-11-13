"""
Writes the transportation demands of the CIE 2025 paper (Table 3) for the 4-workstation case:
training scenarios T1..T5 and their perturbed versions T1_P..T5_P.
Run once:  python make_paper_demands.py
"""
import os

import pandas as pd

TRAIN = {
    'T1': [[0, 20, 0, 30], [0, 0, 20, 0], [0, 0, 0, 20], [30, 0, 0, 0]],
    'T2': [[0, 0, 40, 0], [0, 0, 10, 10], [40, 0, 0, 0], [0, 0, 0, 0]],
    'T3': [[0, 20, 0, 10], [0, 0, 20, 40], [0, 0, 0, 20], [10, 0, 0, 0]],
    'T4': [[0, 0, 30, 0], [0, 0, 10, 20], [30, 0, 0, 0], [0, 20, 0, 0]],
    'T5': [[0, 3, 24, 2], [0, 0, 3, 23], [24, 0, 0, 3], [2, 8, 0, 0]],
}
PERTURBED = {
    'T1_P': [[0, 24, 0, 31], [0, 0, 22, 0], [0, 0, 0, 9], [28, 0, 0, 0]],
    'T2_P': [[0, 0, 33, 0], [0, 0, 11, 10], [33, 0, 0, 0], [0, 0, 0, 0]],
    'T3_P': [[0, 16, 0, 11], [0, 0, 16, 35], [0, 0, 0, 16], [11, 0, 0, 0]],
    'T4_P': [[0, 0, 29, 0], [0, 0, 9, 21], [28, 0, 0, 0], [0, 20, 0, 0]],
    'T5_P': [[0, 3, 23, 2], [0, 0, 3, 20], [23, 0, 0, 3], [2, 0, 0, 0]],
}

if __name__ == '__main__':
    root = os.path.dirname(os.path.abspath(__file__))
    for folder, scenarios in (('paper_M4_train', TRAIN), ('paper_M4_perturbed', PERTURBED)):
        out = os.path.join(root, 'transportation_demand', folder)
        os.makedirs(out, exist_ok=True)
        for name, matrix in scenarios.items():
            df = pd.DataFrame(matrix, index=[f'P{i}' for i in range(1, 5)],
                              columns=[f'D{i}' for i in range(1, 5)])
            df.to_csv(os.path.join(out, f'{name}.csv'))
        print('written', out)
