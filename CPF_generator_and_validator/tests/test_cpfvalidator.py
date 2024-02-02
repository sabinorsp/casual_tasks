try:
    import sys
    import os

    sys.path.append(
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                '../src'
            )
        )
    )
except:
    raise


import unittest
from cpfvalidator import CPFValidator


class TestCPFValidator(unittest.TestCase):

    def test_valid_cpf(self):
        valid_cpfs = ['123.456.789-09', '98765432109', '00000000000']
        for cpf in valid_cpfs:
            cpf_validator = CPFValidator(cpf)
            self.assertTrue(cpf_validator.validar_cpf())


    def test_invalid_cpf(self):
        invalid_cpfs = ['123.456.789-01', '98765432100', '00000000001']
        for cpf in invalid_cpfs:
            cpf_validator = CPFValidator(cpf)
            self.assertFalse(cpf_validator.validar_cpf())


    def test_state_from_cpf(self):
        state_mappings = {
            '123.456.789-09': 'SP',
            '98765432109': 'SP',
            '00000000000': 'RS',
            '333.444.555-96': 'CE',
            '11122334455': 'AC'
        }
        for cpf, state in state_mappings.items():
            cpf_validator = CPFValidator(cpf)
            self.assertEqual(cpf_validator.get_state_from_cpf(), state)


if __name__ == '__main__':
    unittest.main()