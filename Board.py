import numpy as np

# A sudoku board is a list of 81 numbers (arranged by row) - 0 represents an empty cell
## To do: read puzzle from photo
def numbers():
    puzzle = np.array([1,0,7,0,0,9,0,2,0,8,0,0,0,0,0,0,0,7,0,0,0,0,1,0,0,0,0,0,0,1,6,0,7,0,0,5,0,2,0,0,0,0,6,3,0,0,0,9,0,0,5,0,0,2,4,0,0,0,0,0,0,0,0,0,0,3,0,0,1,5,0,0,0,0,0,0,3,0,0,0,8])
    return puzzle

# There are 9 rows and 9 columns in a standard sudoku
def rows():
    return 9
def columns():
    return 9

# Reshape the board so it looks like a 9x9 sudoku
def board(puzzle, rows, columns):
    return np.reshape(puzzle, (rows,columns)).T