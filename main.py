from examples.example_1 import run_example as ex_1
from examples.example_2 import run_example as ex_2
from hypercycle_numeric.one_mol_autocathalitic import run_model as one_molecule
from hypercycle_numeric.n_mol_hypercyclic import run_model as n_mol_hyper
from hypercycle_numeric.inf_hypercyclic import run_model as inf_hyper

def main(model_to_run):
    model_to_run()

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main.py <model_to_run>")
        print("Available models:")
        print("- example_1")
        print("- example_2")
        print("- one_mol_auto")
        print("- n_mol_hyper")
        print("- inf_hyper")
        sys.exit(1)

    model_name = sys.argv[1]

    model_map = {
        "example_1": ex_1,
        "example_2": ex_2,
        "one_mol_auto": one_molecule,
        "n_mol_hyper": n_mol_hyper,
        "inf_hyper": inf_hyper
    }

    if model_name not in model_map:
        print(f"Error: {model_name} is not a valid model.")
        sys.exit(1)

    main(model_map[model_name])