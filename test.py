import matplotlib.pyplot as plt



class testClass():
    def __init__(self, data1):
        self.data1 = data1

    def double(self):
        self.data1 = self.data1*2

    def getData1(self):
        return self.data1


newClass1 = testClass(5)
newClass2 = testClass(10)

print(type(newClass1))

print(newClass1.data1) #5
print(newClass2.data1) #10

newClass1.double()

print(newClass1.data1) #10
print(newClass2.data1) #10

newClass1.double()

print(newClass1.getData1()) #20
print(newClass2.getData1()) #10


