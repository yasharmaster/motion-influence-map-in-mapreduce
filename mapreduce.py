
R = 0
C = 0

def mapper(input):
    sum = 0
    for row in range(len(input)):
        for col in range(len(input[0])):
            sum += input[row][col]
    sum /= 400.0
    return sum


def generate_key_value(x):
    return (( (int(x[0]/20))*100000 + int(x[1]/20) ), x[2]/400.0)
