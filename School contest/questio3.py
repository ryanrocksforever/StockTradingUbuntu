#inputN = input("enter positive int n:") #The input will be a positive integer n, followed by n lines consisting of n capital letters.
inputNlines = input("enter number of students:") # n lines
#inputNlinesletters = input("enter positive int n  Capital letters in line:") #capital letters.
finalinput = []
for x in range(0, int(inputNlines)):
    lettersinline = input("Enter student "+ str(x) + " Name:")
    lettersinline2 = input("Enter student "+ lettersinline + " Grade:")
    finalinput.append([lettersinline, int(lettersinline2)])

acceptedstudents = []
global mood


def main():
    mood = True
    moodcount = 0
    for x in finalinput:
        if moodcount == 0:
            mood = True
        print(x)
        if x[1] >= 90:
            acceptedstudents.append(x[0])
            continue
        if x[1] <= 70:
            #acceptedstudents.append(x[0])
            if x[1] <= 50:
            #acceptedstudents.append(x[0])
                mood= False
                moodcount = 5
            continue
        if mood is True and x[1] >= 80:
            acceptedstudents.append(x[0])
            continue
        else:
            if mood is False:
                if moodcount > 0:
                    moodcount =-1
                continue




print(finalinput)

main()
print("accepted students")
print(acceptedstudents)
