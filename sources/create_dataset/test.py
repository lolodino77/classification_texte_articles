

f = open("corpus_list_philosophy.txt", "r")
# lines = f.readlines()
lines = f.read().splitlines()
print("lines =", lines)
a = []
b = []
for line in lines:
    s = line.split(" ")
    print("s =", s)
    a.append(s[0])
    b.append(s[1])

print("a =", a)
print("b =", b)