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

    def test_chinese_political_ideology(self):
        pred = "A. 习近平新时代中国特色社会主义思想"
        self.assertEqual(extract_answer_from_pred(pred), "A")

    def test_ngerti_marang_kahanane_wong_tuwa(self):
        pred = "a. ngerti marang kahanane wong tuwa"
        self.assertEqual(extract_answer_from_pred(pred), "a")

    def test_fan_brand_comparison(self):
        pred = "4. พัดลมยี่ห้อ B จะประหยัดค่าไฟฟ้าได้มากกว่ายี่ห้อ A 44 บาท และความถี่การหมุนใบพัดเท่ากันทั้งสองยี่ห้อ"
        self.assertEqual(extract_answer_from_pred(pred), "4")

    def test_correct_option_indonesian(self):
        pred = "Jadi, pilihan yang benar adalah: a."
        self.assertEqual(extract_answer_from_pred(pred), "a")

    def test_criminal_responsibility_vietnamese(self):
        pred = "D. Trách nhiệm hình sự được áp dụng cho những cá nhân có hành vi nguy hiểm hoặc vi phạm pháp luật, tức là người có hành vi vi phạm các nguyên tắc quản lý nhà nước không áp dụng trách nhiệm hình sự, mà có thể chịu trách nhiệm hành chính hoặc dân sự tuỳ thuộc vào mức độ vi phạm. Vì vậy, lựa chọn đúng là D."
        self.assertEqual(extract_answer_from_pred(pred), "D")

    def test_criminal_responsibility_vietnamese_extended(self):
        pred = "Trách nhiệm hình sự được áp dụng cho những cá nhân có hành vi nguy hiểm hoặc vi phạm pháp luật, đe dọa đến xã hội. Do đó, lựa chọn đúng là B: Người có hành vi nguy hiểm cho xã hội."
        self.assertEqual(extract_answer_from_pred(pred), "B")

    def test_chinese_answer(self):
        pred = "答案：A"
        self.assertEqual(extract_answer_from_pred(pred), "A")

    def test_expansionist_foreign_policy(self):
        pred = "A) Was 'n ekspansionistiese buitelandse beleid."
        self.assertEqual(extract_answer_from_pred(pred), "A")

    def test_chinese_political_ideology_extended(self):
        pred = "正确答案是 A。 \n\n分析：中国共产党第十九次全国代表大会强调，中国共产党在执政过程中，对共产党执政规律、社会主义建设规律和人类社会发展规律有了更深入的认识和理解，并取得了重大的理论创新成果。这些创新成果构成了 '习近平新时代中国特色社会主义思想'。因此，选项 A 是正确答案。"
        self.assertEqual(extract_answer_from_pred(pred), "A")

if __name__ == '__main__':
    unittest.main()