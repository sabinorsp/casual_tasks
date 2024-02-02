import random

class CPFGenerator:
    NUM_DIGITS = 9
    MODULUS = 11

    def __init__(self) -> None:
        self.cpf = self.gerar_cpf()

    def gera_primeiros_digitos(self):
        return random.sample(range(10), self.NUM_DIGITS)

    def calculate_digit(self, digits):
        product_digito = [a * b for a, b in zip(digits, range(self.NUM_DIGITS + 1, 1, -1))]
        result = sum(product_digito) * 10 % self.MODULUS
        return result if result < 10 else 0

    def gerar_cpf(self):
        primeiros_digitos = self.gera_primeiros_digitos()
        digito1 = self.calculate_digit(primeiros_digitos)
        primeiros_digitos.append(digito1)
        digito2 = self.calculate_digit(primeiros_digitos)
        primeiros_digitos.append(digito2)
        return primeiros_digitos
