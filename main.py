from examples.example_1 import run_example as ex_1
from examples.example_2 import run_example as ex_2
from hypercycle_numeric.one_mol_autocathalitic import run_model as one_molecule
from hypercycle_numeric.n_mol_hypercyclic import run_model as n_mol_hyper
from hypercycle_numeric.inf_hypercyclic import run_model as inf_hyper


def main():
    # ex_1()
    # ex_2()
    n_mol_hyper()
    # one_molecule()
    

if __name__ == "__main__":
    main()
