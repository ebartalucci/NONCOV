import pandas as pd

class EmptyDataFrameCreator:
    def __init__(self):
        self.columns = ['molecule', 'atom', 'noncov', 'x_coord', 'y_coord', 'z_coord',
                        'tot_shielding_11', 'tot_shielding_22', 'tot_shielding_33',
                        'dia_shielding_11', 'dia_shielding_22', 'dia_shielding_33',
                        'para_shielding_11', 'para_shielding_22', 'para_shielding_33',
                        'iso_shift', 'functional', 'basis_set', 'aromatic']

    def create_empty_dataframe(self):
        return pd.DataFrame(columns=self.columns)

def main():
    creator = EmptyDataFrameCreator()
    empty_df = creator.create_empty_dataframe()
    empty_csv_path = "empty_dataframe.csv"
    empty_df.to_csv(empty_csv_path, index=False)
    print(f"Empty DataFrame saved to {empty_csv_path}")

if __name__ == "__main__":
    main()
