class MyPet:
    def __init__(self, pet_name):
        self.name = pet_name

    def pet(self):
        print("Your pet " + self.name + " seemed happy that you pet them.")

    #def make_noise(self):
        #print("Noise noise noise.")

orange_cat = MyPet("f")
small_dog = MyPet("m")

print(orange_cat.name)
#small_dog.play_fetch()

small_dog.pet()


class MyDog(MyPet):

    def bark(self):
        print("Woof woof woof")





#class MyDog(MyPet):

    #def bark(self):
        #print("Woof woof woof")

#my_dog = MyDog("Spike")
#my_dog.pet()
#my_dog.bark()

