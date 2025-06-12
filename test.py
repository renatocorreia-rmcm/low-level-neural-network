from random import randint

cases = 5

for i in range(2*cases):
    # input matrix
    print('\"\"\"')
    for i in range (28):
        for j in range(28):
            x = str(randint(0,99))
            x = " "*(2-len(x)) + x 
            print(x, end=", ")
        print()
    print('\"\"\"')
    # output vector
    for i in range(9):
        x = str(randint(0,10))
        x = " "*(2-len(x)) + x 
        print(x, end=",\n")
