import unittest
from eval import extract_answer_from_pred

class TestExtractAnswerFromPred(unittest.TestCase):

    def test_extract_answer_valid(self):
        pred = "The answer is (42)."
        self.assertEqual(extract_answer_from_pred(pred), "42")

    def test_extract_answer_invalid_format(self):
        pred = "The answer is 42."
        self.assertIsNone(extract_answer_from_pred(pred))

    def test_extract_answer_no_parentheses(self):
        pred = "The answer is 42"
        self.assertIsNone(extract_answer_from_pred(pred))

    def test_extract_answer_multiple_numbers(self):
        pred = "Options are (1), (2), and the answer is (3)."
        self.assertEqual(extract_answer_from_pred(pred), "3")

    def test_longer_text(self):
        pred = """The question is asking for an opinion about World War II. 
(1) Dropping atomic bombs was not necessary to end the war - This is an opinion.
(2) The economies of many countries were damaged by the war - This is a factual statement.
(3) Many families suffered the loss of loved ones during the war - This is a factual statement.
(4) Italy and Germany were members of the Axis powers - This is a factual statement.

Therefore, the answer is (1) Dropping atomic bombs was not necessary to end the war."""
        self.assertEqual(extract_answer_from_pred(pred), "1")

    def test_yellow_journalism_purpose(self):
        pred = "The question is asking about the purpose of yellow journalism in the 1890s. \n\n(1) influence public opinion - This option aligns with the purpose of yellow journalism, which was to sway public opinion through sensationalized and exaggerated stories. \nTherefore, the answer is (1) influence public opinion."
        self.assertEqual(extract_answer_from_pred(pred), "1")

    def test_african_americans_job_opportunities(self):
        pred = "The question is asking where large numbers of African Americans from the South found better job opportunities during the early 1900s. \n\nOption (1) in northern cities correctly answers the question as many African Americans migrated from the South to northern cities like Chicago, Detroit, and New York in search of better job opportunities during the Great Migration. Therefore, the answer is (1) in northern cities."
        self.assertEqual(extract_answer_from_pred(pred), "1")

    def test_vapor_pressure_propanone(self):
        pred = "The question is asking for the vapor pressure of propanone at 45°C. \n\nOption (4) 79 kPa is the correct answer as it is the closest to the typical vapor pressure of propanone at 45°C, which is around 79 kPa."
        self.assertEqual(extract_answer_from_pred(pred), "4")

    def test_vapor_pressure_propanone_at_45C(self):
        pred = "The question is asking for the vapor pressure of propanone at 45°C. \n\nOption (4) 79 kPa is the correct answer as it is the closest to the typical vapor pressure of propanone at 45°C, which is around 79 kPa."
        self.assertEqual(extract_answer_from_pred(pred), "4")

if __name__ == '__main__':
    unittest.main()