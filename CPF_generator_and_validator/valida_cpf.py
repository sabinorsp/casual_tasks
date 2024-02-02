import argparse
from src.cpfvalidator import CPFValidator

def cpf_val(cpf):
    check_cpf = CPFValidator(cpf)
    check_cpf.validar_cpf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cpf', type=str, help='Digite o CPF')
    args = parser.parse_args()

    cpf_val(args.cpf)
