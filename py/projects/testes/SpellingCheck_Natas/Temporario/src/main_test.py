from UFUtils import utils
import codecs
import random
import string


def insert_error(sentence):
    do = random.randint(0,10)
    if do==0:
        new_sentence = []
        for palavra in sentence.split():
            do = random.randint(0, 10)
            if do==0:
                new_word = []
                for letra in palavra:
                    do = random.randint(0, 10)
                    if do==0:
                        new_word.append(random.choice(string.ascii_lowercase))
                    else:
                        new_word.append(letra)
                new_sentence.append(''.join(new_word))
            else:
                new_sentence.append(palavra)
        return ' '.join(new_sentence)
    else:
        return sentence

texts = utils.load_text("../data/Rapunzel_250_original.txt").split('\n')
final_text_original = ''
final_text_corrected = ''
for text in texts:
    print("++++++++++")
    print("Doing...")
    resultado = insert_error(text)
    final_text_original += resultado + '\n'
    resultado = utils.correct_spell(resultado)
    final_text_corrected += resultado + '\n'
    print("Done...")


file = codecs.open("../data/Rapunzel_250_test_albert_corredtec.txt", "w", "utf-8")
file.write(final_text_corrected)
file.close()

file = codecs.open("../data/Rapunzel_250_test_albert_modified.txt", "w", "utf-8")
file.write(final_text_original)
file.close()