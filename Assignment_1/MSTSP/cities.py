import random


def cityCoordinates():
    return [(-271, -202), (132, 320), (-357, -389), (-76, -171), (259, 269), (-244, 302), (140, -222), (249, -252), (-178, 332), (-370, 272), (343, 286), (-367, 254), (-80, -322), (79, -365), (-249, 13)]
    # return [(28, -358), (65, -72), (212, 319), (386, 72), (337, -149), (-387, -325), (391, 220), (363, -359), (195, -343),
    #  (-107, -78), (66, -352), (191, -200), (-66, -86), (-269, 98), (351, 293), (-170, -183), (124, 76), (236, -246),
    #  (-188, 39), (-367, 156), (131, -210), (104, 132), (40, 103), (295, 34), (68, 296), (-84, 59), (192, -377),
    #  (-86, 366), (46, -352), (16, 251), (-27, -217), (63, 267), (-129, 18), (21, 344), (363, -164), (123, 278),
    #  (-222, 351), (-268, -381), (-148, -226), (-187, -331), (72, -177), (181, -393), (122, 13), (-343, 137), (68, -241),
    #  (-120, 195), (-242, 280), (45, -338), (-121, -342), (-125, -395)]
    # return [(-819, -847), (417, -725), (982, 143), (-302, -949), (533, 1903), (-238, -1826), (1749, 1359), (-1081, 267), (1618, 565), (654, -328), (1550, 250), (-731, 1653), (1557, -873), (-749, -97), (1225, 681), (-1361, -281), (32, 1505), (192, 1015), (822, 111), (287, 1755), (-630, -511), (1735, 1924), (-1630, -966), (-1439, 1105), (549, -1936), (-546, -81), (265, -5), (-1855, -752), (255, -414), (-1817, 1419), (-433, 1473), (-1648, -349), (1188, -1741), (-1835, 1554), (-771, -1911), (1646, 1552), (-539, -1585), (870, 666), (1549, 1685), (1731, -239), (-716, 797), (-1899, -1759), (1590, -1789), (1901, 1329), (189, 1839), (-877, 1298), (1749, 786), (-4, 437), (1787, -1252), (-708, 1997)]



def generateCities_dense(nr_cities):
    print([(random.randint(-400, 400), random.randint(-400, 400)) for _ in range(0, nr_cities)])

def generateCities(nr_cities):
    print([(random.randint(-2000, 2000), random.randint(-2000, 2000)) for _ in range(0, nr_cities)])