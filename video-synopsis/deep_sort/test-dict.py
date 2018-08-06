my_dict = {}

def add (key, value):
    if not key in my_dict:
    	my_dict[key] = value

if __name__ == '__main__':
    add(1, "5:30")
    add(2, "5:30")
    add(1, "5;45")
    # print(my_dict)

    # ts = [(1, "5:30"), (2, "5:40"), (1, "5:44")]


d = {}
l1 = [1, 2, 3, 4]
l2 = [1, 3, 5, 6]

for i in l1:
	if not i in d:
		d[i] = "5:30"

# print(d)

for i in l2:
	if not i in d:
		d[i] = "5:45"

# print(d)

d = {}
d = {i:"5:30" for i in l1 if not i in d}
d = {i:"5:40" for i in l2 if not i in d}