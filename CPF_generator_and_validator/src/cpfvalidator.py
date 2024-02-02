class CPFValidator:
    COD_ESTADOS = {1: ['MT', 'GO', 'MS', 'MT', 'TO'],
                   2: ['AC', 'AM', 'PA', 'RO', 'RR'],
                   3: ['CE', 'MA', 'PI'],
                   4: ['AL', 'PB', 'PE', 'RN'],
                   5: ['BA', 'SE'],
                   6: ['MG'],
                   7: ['ES', 'RJ'],
                   8: ['SP'],
                   9: ['PR', 'SC'],
                   0: ['RS']}


    def __init__(self, cpf):
        self.cpf = cpf


    def extrair_primeiros_digitos_cpf(self):
        cpf_numerico = ''.join(filter(str.isdigit, self.cpf))
        cpf_list = [int(digito) for digito in cpf_numerico]
        return cpf_list


    def validar_digito(self, digito):
        cpf_list = self.extrair_primeiros_digitos_cpf()
        produto_digito = []
        prod_numeros = list(range(2, 12 if digito == 2 else 11))
        prod_numeros.reverse()
        for index in range(len(prod_numeros)):
            produto_digito.append(prod_numeros[index] * cpf_list[index])

        val_cpf = sum(produto_digito) * 10 % 11
        return val_cpf == cpf_list[9 if digito == 1 else 10]


    def validar_cpf(self):
        cod_digit1 = self.validar_digito(1)
        cod_digit2 = self.validar_digito(2)
        print(f'Digito 1 = {cod_digit1}')
        print(f'Digito 2 = {cod_digit2}')
        if cod_digit1:
            cod_digit2 = self.validar_digito(2)
            if cod_digit2:
                print('CPF Válido')
        else:
            print('CPF Inválido!')
        

            

            
        
    