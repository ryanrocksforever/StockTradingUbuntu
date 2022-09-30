# Python3 program to search a word in a 2D grid
class findpali:

    def __init__(self):
        self.R = None
        self.C = None
        self.dir = [[-1, 0], [1, 0], [1, 1],
                    [1, -1], [-1, -1], [-1, 1],
                    [0, 1], [0, -1]]

    # This function searches in all 8-direction
    # from point(row, col) in grid[][]
    def search2D(self, grid, row, col, word):

        # If first character of word doesn't match
        # with the given starting point in grid.
        if grid[row][col] != word[0]:
            return False

        # Search word in all 8 directions
        # starting from (row, col)
        for x, y in self.dir:

            # Initialize starting point
            # for current direction
            rd, cd = row + x, col + y
            flag = True

            # First character is already checked,
            # match remaining characters
            for k in range(1, len(word)):

                # If out of bound or not matched, break
                if (0 <= rd <self.R and
                        0 <= cd < self.C and
                        word[k] == grid[rd][cd]):

                    # Moving in particular direction
                    rd += x
                    cd += y
                else:
                    flag = False
                    break

            # If all character matched, then
            # value of flag must be false
            if flag:
                return True
        return False

    # Searches given word in a given matrix
    # in all 8 directions
    def patternSearch(self, grid, word):

        # Rows and columns in given grid
        self.R = len(grid)
        self.C = len(grid[0])

        # Consider every point as starting point
        # and search given word
        for row in range(self.R):
            for col in range(self.C):
                if self.search2D(grid, row, col, word):
                    print("pattern found at " +
                          str(row) + ', ' + str(col))

# Driver Code
if __name__=='__main__':


    #inputN = input("enter positive int n:") #The input will be a positive integer n, followed by n lines consisting of n capital letters.
    inputNlines = input("enter positive int nLines:") # n lines
    inputNlinesletters = input("enter positive int n  Capital letters in line:") #capital letters.
    finalinput = []
    for x in range(0, int(inputNlines)):
        lettersinline = input("Enter letters in line "+ str(x) + " :")
        finalinput.append(lettersinline)

    grid = finalinput

    # grid = ["PDBCDDDDDDDDD",
    #         "DAFDDDDDDDDFF",
    #         "DDLDDFDFDFDFF",
    #         "DDDIFDFDFDFDF"]
    pali = findpali()
    pali.patternSearch(grid=grid, word='PALI')
    print('')

