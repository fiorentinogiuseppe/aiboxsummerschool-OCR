from UFUtils import utils

text = utils.load_text("../data/Rapunzel_250_test.txt")
resultdo = utils.correct_spell(text)
print(resultdo)
