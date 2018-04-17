

try:
    print(2/0)
except NameError:
    print("variable a not defined")
except:
    print("Something is not right")
finally:
    print("yo.")
