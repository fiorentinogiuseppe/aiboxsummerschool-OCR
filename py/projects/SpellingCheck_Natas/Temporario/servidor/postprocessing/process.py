'''def handle_file_upload(f):
    from os import path
    text_received = f.read().decode('utf-8')
    title = generate_string()
    file_path = 'outputs/' + title + '.txt'
    while path.exists(file_path):
        title = generate_string()
        file_path = 'outputs/' + title + '.txt'
    with open(file_path, 'w+') as destination:
        destination.write(text_received)

    return title
    
def generate_string(string_length = 10):
    import string, random
    lettersAndDigits = string.ascii_letters + string.digits
    return ''.join(random.choice(lettersAndDigits) for i in range(string_length))'''
from .UFUtils import utils

def handle_file_upload(f):
    text_received = f.read().decode('utf-8')
    lines = text_received.split('\n')
    text_return = []
    for line in lines:
        text_return.append(utils.correct_spell(line)+'\n')
    return '<p>' + ''.join(text_return) + '</p>'
