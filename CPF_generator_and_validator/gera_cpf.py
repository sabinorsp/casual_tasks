from src.cpfgerador import CPFGenerator

cpf_gerado = CPFGenerator()
print('CPF gerado:', ''.join(map(str, cpf_gerado.cpf)))
