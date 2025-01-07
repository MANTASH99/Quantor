with open('summed_document_generalized.csv') as f:
    lines = f.readlines()
    count = 2
    offset = 0
    output = ''

    id = lines[1].split('-')[0]
    output += f'{id},{count}'

    for i in range(len(lines)):
        if i == 0:
            pass
        elif lines[i].startswith(id):
            count += 1
        else:
            output += f',{count-1}\n'
            id = lines[i].split('-')[0]
            output += f'{id},{count}'
            count += 1
    output += f',{count-1}'

print(output)

