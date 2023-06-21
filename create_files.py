
from os import system

type_1 = ["train", "test"]
type_2 = {"An":["CNV", "DRUSEN","DME"], "C":["CNV"], "Dr":["DRUSEN"], "D":["DME"]}

directory = "OCT-DATABASE/"

for t2, c in type_2.items():
        location = directory + "Bin_Nx" + t2
        system("mkdir " + location)
        for t1 in type_1:
                location_t = location + "/" + t1
                system("mkdir " + location_t)
                system("ln -s " + directory + "Mult/" + t1 + "/NORMAL " + location_t)
                if len(c) > 1:
                        system("mkdir " + location_t + "/ANORMAL")
                        location_t += "/ANORMAL"
                for name in c:
                        system("ln -s " + directory + "Mult/" + t1 + "/" + name + "/ " + location_t)
