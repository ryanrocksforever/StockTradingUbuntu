
paintbuckets = input("paint buckets: ") #n
paintbuckets= int(paintbuckets)
canvascapacity = input("total capacity: ") #c
canvascapacity = int(canvascapacity)

finalinput = []
for x in range(0, int(paintbuckets)):
    paintcapacity = input("Enter capacity for paint bucket"+ str(x) + " :")
    finalinput.append(int(paintcapacity))

finalinput = sorted(finalinput, reverse=False)
print(finalinput)

emptybuckets = 0
for x in range(0, paintbuckets):
    projectcanvascapacity = canvascapacity - finalinput[x]
    if projectcanvascapacity <= 0:
        print("full "+ str(x))
        emptybuckets = x

        break
    else:
        canvascapacity = projectcanvascapacity
print("answer: ")
print(emptybuckets)