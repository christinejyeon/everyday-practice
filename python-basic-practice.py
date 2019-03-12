#1 Using List
print("#1 Using List")

dogList = []
dogName = 0

while dogName != "":
    dogName = input("Write your dog's name!: ")
    if dogName != "":
        dogList.append(dogName)
    else:
        print(dogList)


#2 Counting Words
print("\n#2 Counting Words")

file = open('textdata.txt', 'r')

data = file.read()

words = data.split(" ")

counter = []

for word in words:
    count = words.count(word)
    if count != 1:
        counter.append(word + ":" + str(count))
        for repetition in range(count):
            words.remove(word)
    elif count == 1:
        counter.append(word + ":" + str(count))

counter.sort()

print(counter)


#3 Class
print("\n#3 Class")

rectW=[]
rectH=[]

class Rect:
    def __init__(self, width, height):
        self.width=width
        self.height=height
        rectW.append(self.width)
        rectH.append(self.height)
def main():
    allArea=[]
    summm=0
    for i in range(len(rectW)):
            allArea.append(int(rectW[i]*rectH[i]))
    for i in range(len(allArea)):
            summm += allArea[i]
    print(summm)
